"""Process functions."""
import logging
import shutil
from functools import partial
from multiprocessing.pool import Pool
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_validation_framework.result import ValidationResult
from data_validation_framework.result import ValidationResultSet
from diameter_synthesis.main import build_diameters
from diameter_synthesis.main import build_model
from matplotlib.backends.backend_pdf import PdfPages
from morph_tool.converter import convert
from morph_tool.exceptions import MorphToolException
from morph_tool.morphdb import MorphDB
from morph_tool.morphdb import MorphInfo
from morphio import Option
from morphio.mut import Morphology
from neurom import load_morphology
from neurom import view
from neuror.main import repair as _repair
from neuror.unravel import unravel as _unravel
from neuror.utils import RepairType
from neuror.zero_diameter_fixer import fix_zero_diameters as _fix_zero_diameters
from scipy.spatial import KDTree
from tqdm import tqdm

from morphology_workflows import MorphologyWorkflowsError
from morphology_workflows.marker_helper import MarkerSet
from morphology_workflows.utils import silent_loggers

L = logging.getLogger(__name__)
mpl.use("Agg")


class RepairError(MorphologyWorkflowsError):
    """Exception for Repair step."""


def write_neuron(neuron, filename):
    """Write a NEURON ordered version of the morphology."""
    Morphology(neuron, options=Option.nrn_order).write(filename)


def fix_zero_diameters(row, data_dir):
    """Assign mtype to morphologies."""
    neuron = Morphology(row.morph_path)
    _fix_zero_diameters(neuron)
    morph_path = data_dir / Path(row.morph_path).name
    write_neuron(neuron, morph_path)
    return ValidationResult(is_valid=True, morph_path=morph_path)


def _map_apical(apical_point, mapping):
    """Map apical points as cut-leaves."""
    t = KDTree(mapping[["x0", "y0", "z0"]])
    _, indices = t.query(apical_point)
    return mapping.iloc[indices][["x1", "y1", "z1"]].values


def _unravel_leaves(leaves, mapping):
    """Adapted from neuror.unraavel.unravel_plane."""
    t = KDTree(mapping[["x0", "y0", "z0"]])
    distances, indices = t.query(leaves)
    not_matching_leaves = np.where(distances > 1e-3)[0]
    if not_matching_leaves.size:
        raise RepairError(  # noqa: TRY003
            f"Cannot find the following leaves in the mapping:\n{leaves[not_matching_leaves]}"
        )
    return mapping.iloc[indices][["x1", "y1", "z1"]].values


@silent_loggers("neuror")
def unravel(row, data_dir, window_half_length=10):
    """Unravel morphologies and update cut plane."""
    neuron, mapping = _unravel(row.morph_path, window_half_length=window_half_length)
    morph_path = data_dir / Path(row.morph_path).name
    write_neuron(neuron, morph_path)

    # transform original cut leaves
    if not row.isnull()["cut_leaves_path"]:
        cut_leaves_marker = MarkerSet.from_file(row.cut_leaves_path)
        cut_leaves_marker.markers[0].data = _unravel_leaves(
            cut_leaves_marker.markers[0].data, mapping
        )
        new_cut_leaves_path = data_dir / (row.name + ".yaml")
        cut_leaves_marker.save(filename=new_cut_leaves_path)

        # transform original apical points
        new_apical_point_path = np.nan
        if not row.isnull()["apical_point_path"]:
            new_apical_point_path = (data_dir / ("apical_" + row.name)).with_suffix(".yaml")
            apical_point = MarkerSet.from_file(row.apical_point_path)
            apical_point.markers[0].data = _map_apical(
                np.array(apical_point.markers[0].data), mapping
            )
            apical_point.save(filename=new_apical_point_path)
    else:
        new_cut_leaves_path = None
        new_apical_point_path = None

    return ValidationResult(
        is_valid=True,
        morph_path=morph_path,
        unravelled_cut_leaves_path=new_cut_leaves_path,
        unravelled_apical_point_path=new_apical_point_path,
    )


def repair(row, data_dir, with_plot=False, repair_params=None):
    """Repair morphologies.

    TODO: understand the repair_flag correctly and how to set them.
    """
    morph_path = data_dir / Path(row.morph_path).name
    apical_point = -1
    if not row.isnull()["unravelled_apical_point_path"]:
        apical_point = (
            MarkerSet.from_file(row.unravelled_apical_point_path).markers[0].data.tolist()
        )

    if not row.isnull()["unravelled_cut_leaves_path"]:
        leaves = MarkerSet.from_file(row.unravelled_cut_leaves_path).markers[0].data
        _repair(  # pylint: disable=unexpected-keyword-arg
            row.morph_path,
            morph_path,
            axons=None,
            seed=0,
            cut_leaves_coordinates=leaves,
            legacy_detection=False,  # cut_leaves_coordinates are provided, so this is not used
            plot_file=(data_dir / row.name).with_suffix(".html") if with_plot else None,
            repair_flags={
                RepairType.axon: row.has_axon,
                RepairType.tuft: row.has_apical,
                RepairType.oblique: row.has_apical,
                RepairType.basal: row.has_basal,
            },
            apical_point=apical_point,
            params=repair_params,
        )
    else:
        shutil.copy(row.morph_path, morph_path)
    return ValidationResult(is_valid=True, morph_path=morph_path)


def plot_repair(row, data_dir, with_plotly=True):
    """Plotting cut leaves on morphologies."""
    markers = None
    if not row.isnull()["cut_leaves_path"]:
        markers = MarkerSet.from_file(row.cut_leaves_path)
    plot_path = None
    if with_plotly:
        if markers is not None:
            plot_path = (data_dir / row.name).with_suffix(".html")
            markers.plot(filename=plot_path)

    else:
        mpl.font_manager._get_font.cache_clear()  # pylint: disable=protected-access # noqa: SLF001
        plot_path = (data_dir / row.name).with_suffix(".pdf")
        neuron = load_morphology(row.morph_path)

        points = None
        if markers is not None:
            points = markers.markers[0].data.T

        with silent_loggers("matplotlib.font_manager", "matplotlib.backends.backend_pdf"), PdfPages(
            plot_path
        ) as pdf:
            for plane in ["xy", "xz", "yz"]:
                plt.figure()
                ax = plt.gca()

                view.plot_morph(
                    neuron, ax, realistic_diameters=True, plane=plane, soma_outline=False
                )
                if points is not None:
                    proj = {"xy": [0, 1], "xz": [0, 2], "yz": [1, 2]}
                    plt.scatter(*points[proj[plane]], color="g", s=2, marker="o")
                ax.autoscale()
                ax.axis("equal")
                plt.title(f"{row.name}")
                if hasattr(row, "mtype") and isinstance(row.mtype, str):
                    plt.suptitle(plane + "mtype:" + row.mtype)
                else:
                    plt.suptitle(plane)
                pdf.savefig()
                plt.close()

    return ValidationResult(is_valid=True, plot_repair_path=plot_path)


def smooth_diameters(df, data_dir, neurite_types=None, seed=42):
    """Smooth diameters using diameter-synthesis simpler algorithm."""
    if neurite_types is None:
        neurite_types = ["basal_dendrite", "apical_dendrite"]

    config = {"models": ["simpler"], "neurite_types": neurite_types, "seed": seed}

    if "with_diameters" not in df.columns:
        df["with_diameters"] = True

    morphologies = [
        load_morphology(df.loc[gid, "morph_path"])
        for gid in df.index
        if df.loc[gid, "with_diameters"]
    ]
    model_params = build_model(morphologies, config)

    for gid in df.index:
        morph = Morphology(df.loc[gid, "morph_path"])
        build_diameters(morph, neurite_types, model_params, diam_params=config)
        morph_path = data_dir / Path(df.loc[gid, "morph_path"]).name
        df.loc[gid, "morph_path"] = morph_path
        write_neuron(morph, morph_path)
    return ValidationResultSet(data=df, output_columns={"morph_path": None})


def plot_smooth_diameters(row, data_dir, shift=200):
    """Plot original morphology and smoother one next to each other."""
    orig_neuron = load_morphology(row.morph_path)
    smooth_neuron = load_morphology(row.smooth_morph_path)
    plt.figure()
    ax = plt.gca()
    view.plot_morph(
        orig_neuron.transform(lambda p: p - np.array([shift, 0, 0])),
        ax=ax,
        plane="xy",
        soma_outline=False,
        realistic_diameters=False,
    )
    view.plot_morph(
        smooth_neuron.transform(lambda p: p + np.array([shift, 0, 0])),
        ax=ax,
        plane="xy",
        soma_outline=False,
        realistic_diameters=False,
    )
    plot_path = (data_dir / row.name).with_suffix(".pdf")
    plt.autoscale()
    plt.axis("equal")
    plt.savefig(plot_path, bbox_inches="tight")
    return ValidationResult(is_valid=True, plot_smooth_path=plot_path)


# pylint: disable=too-many-arguments,too-many-locals
def make_collage(  # noqa: PLR0913
    df,
    data_dir,
    collage_path="collage.pdf",
    separation=500,
    n_morph_per_page=10,
    middle_panel_shift=1000,
    top_panel_shift=2000,
    layer_thickness=None,
    rasterized=False,
    dpi=1000,
    figsize=None,
):
    """Make collage plot of morphologies per mtype."""
    if layer_thickness is None:
        layer_thickness = [700, 525, 190, 353, 149, 165]

    if figsize is None:
        figsize = (12, 10)

    layer_boundaries = np.insert(np.cumsum(layer_thickness), 0, 0)
    layer_centers = 0.5 * (layer_boundaries[:-1] + layer_boundaries[1:])[::-1]
    middle_panel_shift += layer_boundaries[-1]
    top_panel_shift += layer_boundaries[-1]

    mtypes = sorted(df.mtype.unique())
    with silent_loggers("matplotlib.font_manager", "matplotlib.backends.backend_pdf"), PdfPages(
        data_dir.parent.parent / collage_path
    ) as pdf:
        for mtype in tqdm(mtypes):
            _df = df[df.mtype == mtype]
            name_batches = np.array_split(_df.index, max(1, len(_df.index) / n_morph_per_page))
            for page, batch in enumerate(name_batches):
                plt.figure(figsize=figsize)
                ax = plt.gca()
                for shift, morph_name in enumerate(batch):
                    try:
                        layer_pos = layer_centers[int(mtype[1]) - 1]
                    except (TypeError, IndexError, ValueError):
                        layer_pos = layer_centers.mean()
                    neuron = load_morphology(_df.loc[morph_name, "morph_path"])

                    translate = [shift * separation, layer_pos, 0.0]
                    _neuron = neuron.transform(lambda p, t=translate: p + t)
                    view.plot_morph(
                        _neuron,
                        ax,
                        plane="xy",
                        soma_outline=False,
                        realistic_diameters=False,
                    )
                    ax.axhline(middle_panel_shift, c="k", lw=0.5, ls="--")

                    translate = [shift * separation, 0, middle_panel_shift]
                    _neuron = neuron.transform(lambda p, t=translate: p + t)
                    view.plot_morph(
                        _neuron,
                        ax,
                        plane="xz",
                        soma_outline=False,
                        realistic_diameters=False,
                    )
                    ax.axhline(top_panel_shift, c="k", lw=0.5, ls="--")

                    translate = [0, shift * separation, top_panel_shift]
                    _neuron = neuron.transform(lambda p, t=translate: p + t)
                    view.plot_morph(
                        _neuron,
                        ax,
                        plane="yz",
                        soma_outline=False,
                        realistic_diameters=False,
                    )
                    ax.text(
                        shift * separation,
                        layer_boundaries[0] - 100,
                        morph_name,
                        fontsize="x-small",
                        rotation=45,
                        horizontalalignment="center",
                    )

                for lb in layer_boundaries:
                    ax.axhline(lb, color="r", ls="--", alpha=0.3)

                ax.autoscale_view()
                ax.set_rasterized(rasterized)
                ax.set_title(f"{mtype}, page {page + 1} / {len(name_batches)}")
                ax.set_ylim(layer_boundaries[0] - 150, top_panel_shift + 100)
                ax.set_aspect("equal")
                pdf.savefig(bbox_inches="tight", dpi=dpi)
                plt.close()


@silent_loggers("morph_tool.converter")
def _convert(input_file, output_file):
    """Handles crashes in conversion of writing of morphologies."""
    try:
        L.debug("Converting %s into %s", input_file, output_file)
        convert(input_file, output_file, nrn_order=True, sanitize=True)
    except MorphToolException as exc:
        return (
            f"Could not convert the file '{input_file}' into '{output_file}' because of the "
            f"following exception:\n{exc}"
        )
    except RuntimeError:  # This can happen if duplicates are being written at the same time
        pass
    return output_file


def _create_db_row(_data, zero_diameter_path, unravel_path, repair_path, extension):
    """Create a db row and convert morphology."""
    index, data = _data
    m = MorphInfo(
        name=data["morph_name"],
        mtype=data["mtype"] if isinstance(data["mtype"], str) else "",
        layer=data["layer"],
        use_dendrite=data["has_basal"],
        use_axon=data["has_axon"],
    )

    if zero_diameter_path is not None:
        zero_diameter_release_path = (
            str(zero_diameter_path / Path(data["zero_diameter_morph_path"]).stem) + extension
        )

        data[f"zero_diameter_release_morph_path_{extension[1:]}"] = _convert(
            data["zero_diameter_morph_path"], zero_diameter_release_path
        )

    if unravel_path is not None:
        unravel_release_path = str(unravel_path / Path(data["unravel_morph_path"]).stem) + extension
        data[f"unravel_release_morph_path_{extension[1:]}"] = _convert(
            data["unravel_morph_path"], unravel_release_path
        )

    if repair_path is not None:
        repair_release_path = str(repair_path / Path(data["repair_morph_path"]).stem) + extension
        data[f"repair_release_morph_path_{extension[1:]}"] = _convert(
            data["repair_morph_path"], repair_release_path
        )
    return index, data, m


def set_layer_column(df):
    """Set layer values from mtype name if no layer column is present."""
    for gid in df.index:
        if df.loc[gid, "layer"] is None:
            mtype = df.loc[gid, "mtype"]
            if isinstance(mtype, str):
                if len(mtype) > 1:
                    try:
                        layer = int(mtype[1])
                    except ValueError:
                        layer = 0
            else:
                layer = 0
            df.loc[gid, "layer"] = layer
            df.loc[gid, "mtype"] = mtype


def add_duplicated_layers(df):
    """Duplicate entries if layer name has mixed layers, i.e. L23_PC."""
    for mtype in df.mtype.unique():
        if pd.isnull(mtype):
            layer = "0"
        else:
            split = mtype.split("_")
            if len(split) > 0 and len(split[0]) > 0:
                layer = [0][1:]
            else:
                layer = "0"
        if len(layer) > 1:
            _df = df[df.mtype == mtype]
            _df.layer = int(layer[1])
            df = pd.concat([df, _df]).reset_index(drop=True)
    return df


def make_release(
    df,
    _,
    release_path,
    zero_diameter_path,
    unravel_path,
    repair_path,
    extensions,
    duplicate_layers=True,
    nb_processes=None,
):
    """Make morphology release."""
    set_layer_column(df)

    df_tmp = df.reset_index()
    if duplicate_layers:
        df_tmp = add_duplicated_layers(df_tmp)

    release_path = Path(release_path or "")

    L.debug("Exporting releases to %s using %s processes", release_path, nb_processes)

    for extension in extensions:
        _zero_diameter_path = None
        if zero_diameter_path is not None:
            _zero_diameter_path = release_path / f"{zero_diameter_path}/{extension[1:]}"
            _zero_diameter_path.mkdir(exist_ok=True, parents=True)

        _unravel_path = None
        if unravel_path is not None:
            _unravel_path = release_path / f"{unravel_path}/{extension[1:]}"
            _unravel_path.mkdir(exist_ok=True, parents=True)

        _repair_path = None
        if repair_path is not None:
            _repair_path = release_path / f"{repair_path}/{extension[1:]}"
            _repair_path.mkdir(exist_ok=True, parents=True)

        __create_db_row = partial(
            _create_db_row,
            zero_diameter_path=_zero_diameter_path,
            unravel_path=_unravel_path,
            repair_path=_repair_path,
            extension=extension,
        )

        _m = []
        written = set()
        with Pool(nb_processes) as pool:
            for index, row, m in tqdm(
                pool.imap(__create_db_row, df_tmp.loc[df_tmp["is_valid"]].iterrows()),
                total=len(df_tmp),
            ):
                if row["morph_name"] in df.index and row["morph_name"] not in written:
                    df.loc[row["morph_name"]] = pd.Series(row)
                    written.add(row["morph_name"])
                df_tmp.loc[index] = pd.Series(row)
                _m.append(m)

        db = MorphDB(_m)
        if _zero_diameter_path is not None:
            db.write(_zero_diameter_path / "neurondb.xml")
            col_name = f"zero_diameter_morph_db_path_{extension[:1]}"
            db_path = _zero_diameter_path / "neurondb.xml"
            df.loc[df["is_valid"], col_name] = db_path
            df_tmp.loc[df_tmp["is_valid"], col_name] = db_path

        if _unravel_path is not None:
            db.write(_unravel_path / "neurondb.xml")
            col_name = f"unravel_morph_db_path_{extension[1:]}"
            db_path = _unravel_path / "neurondb.xml"
            df.loc[df["is_valid"], col_name] = db_path
            df_tmp.loc[df_tmp["is_valid"], col_name] = db_path

        if _repair_path is not None:
            db.write(_repair_path / "neurondb.xml")
            col_name = f"repair_morph_db_path_{extension[1:]}"
            db_path = _repair_path / "neurondb.xml"
            df.loc[df["is_valid"], col_name] = db_path
            df_tmp.loc[df_tmp["is_valid"], col_name] = db_path
