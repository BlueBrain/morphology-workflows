"""Process functions."""
import logging
import shutil
from functools import partial
from multiprocessing.pool import Pool
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_validation_framework.result import ValidationResult
from diameter_synthesis.main import diametrize_single_neuron
from matplotlib.backends.backend_pdf import PdfPages
from morph_tool.converter import convert
from morph_tool.exceptions import MorphToolException
from morph_tool.morphdb import MorphDB
from morph_tool.morphdb import MorphInfo
from morph_tool.spatial import point_to_section_segment
from morphio.mut import Morphology
from neurom import load_morphology
from neurom import view
from neuror.main import RepairType
from neuror.main import repair as _repair
from neuror.unravel import unravel as _unravel
from neuror.zero_diameter_fixer import fix_zero_diameters as _fix_zero_diameters
from scipy.spatial.ckdtree import cKDTree
from tqdm import tqdm

from morphology_workflows.marker_helper import MarkerSet

L = logging.getLogger(__name__)
matplotlib.use("Agg")


def fix_zero_diameters(row, data_dir):
    """Assign mtype to morphologies."""
    neuron = Morphology(row.morph_path)
    _fix_zero_diameters(neuron)
    morph_path = data_dir / Path(row.morph_path).name
    neuron.write(morph_path)
    return ValidationResult(is_valid=True, morph_path=morph_path)


def _map_apical(apical_point, mapping):
    """Map apical points as cut-leaves."""
    t = cKDTree(mapping[["x0", "y0", "z0"]])
    _, indices = t.query(apical_point)
    return mapping.iloc[indices][["x1", "y1", "z1"]].values


def _unravel_leaves(leaves, mapping):
    """Adapted from neuror.unraavel.unravel_plane."""
    t = cKDTree(mapping[["x0", "y0", "z0"]])
    distances, indices = t.query(leaves)
    not_matching_leaves = np.where(distances > 1e-3)[0]
    if not_matching_leaves.size:
        raise Exception(
            f"Cannot find the following leaves in the mapping:\n{leaves[not_matching_leaves]}"
        )
    return mapping.iloc[indices][["x1", "y1", "z1"]].values


def unravel(row, data_dir, window_half_length=10):
    """Unravel morphologies and update cut plane."""
    neuron, mapping = _unravel(row.morph_path, window_half_length=window_half_length)
    morph_path = data_dir / Path(row.morph_path).name
    neuron.write(morph_path)

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
        matplotlib.font_manager._get_font.cache_clear()  # pylint: disable=protected-access
        plot_path = (data_dir / row.name).with_suffix(".pdf")
        neuron = load_morphology(row.morph_path)

        points = None
        if markers is not None:
            points = markers.markers[0].data.T[[0, 2]]

        with PdfPages(plot_path) as pdf:
            for plane in ["xy", "xz", "yz"]:
                plt.figure()
                ax = plt.gca()

                view.plot_morph(
                    neuron, ax, realistic_diameters=True, plane="xz", soma_outline=False
                )
                if points is not None:
                    plt.scatter(*points, color="g", s=2, marker="o")
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


def smooth_diameters(row, data_dir):
    """Smooth diameters using diameter-synthesis."""
    morph = Morphology(row.morph_path)

    config = {
        "model": {
            "taper": {"max": 1e-06, "min": -0.5},
            "terminal_threshold": 2.0,
            "models": ["generic"],
            "neurite_types": ["basal", "apical"],
        },
        "diameters": {
            "models": ["neurite_based"],
            "n_samples": 5,
            "neurite_types": ["basal", "apical"],
            "seed": 0,
            "trunk_max_tries": 200,
        },
    }
    if isinstance(row.apical_point_path, str):
        apical_point = MarkerSet.from_file(row.apical_point_path).markers[0].data
        apical_sec = [point_to_section_segment(morph, apical_point)[0]]
    else:
        apical_sec = None
    diametrize_single_neuron(morph, config, apical_sec)
    morph_path = data_dir / Path(row.morph_path).name
    morph.write(morph_path)
    return ValidationResult(is_valid=True, morph_path=morph_path)


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
    plot_path = (data_dir / row.name).with_suffix(".png")
    plt.autoscale()
    plt.axis("equal")
    plt.savefig(plot_path, bbox_inches="tight")
    return ValidationResult(is_valid=True, plot_smooth_path=plot_path)


# pylint: disable=too-many-arguments,too-many-locals
def make_collage(
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
    with PdfPages(data_dir.parent.parent / collage_path) as pdf:
        for mtype in tqdm(mtypes):
            _df = df[df.mtype == mtype]
            name_batches = np.array_split(_df.index, max(1, len(_df.index) / n_morph_per_page))
            for page, batch in enumerate(name_batches):
                plt.figure(figsize=figsize)
                ax = plt.gca()
                for shift, morph_name in enumerate(batch):
                    try:
                        layer_pos = layer_centers[int(mtype[1]) - 1]
                    except (TypeError, IndexError):
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
                        shift * separation - separation / 2.0,
                        layer_boundaries[0] - 100,
                        morph_name,
                        fontsize="x-small",
                        rotation=45,
                    )

                for lb in layer_boundaries:
                    ax.axhline(lb, color="r", ls="--", alpha=0.3)

                ax.set_rasterized(rasterized)
                ax.set_title(f"{mtype}, page {page + 1} / {len(name_batches)}")
                ax.set_ylim(layer_boundaries[0] - 150, top_panel_shift + 100)
                ax.set_aspect("equal")
                pdf.savefig(bbox_inches="tight", dpi=dpi)
                plt.close()


def _convert(input_file, output_file):
    """Handles crashes in convertion of writing of morphologies."""
    try:
        convert(input_file, output_file)
        return output_file
    except MorphToolException:
        return "cannot save"


def _get_layer_mtype(mtype_input):
    """Helper to get layer from mtype, if mtype exists as a str."""
    layer = "no_layer"
    mtype = "no_mtype"
    if isinstance(mtype_input, str):
        mtype = mtype_input
        if len(mtype_input) > 1:
            layer = mtype_input[1]
    return mtype, layer


def _create_db_row(_data, zero_diameter_path, unravel_path, repair_path, extension):
    """Create a db row and convert morphology."""
    name, data = _data
    mtype, layer = _get_layer_mtype(data["mtype"])
    m = MorphInfo(
        name=name,
        mtype=mtype,
        layer=layer,
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
    return name, data, m


def make_release(df, _, zero_diameter_path, unravel_path, repair_path, extensions):
    """Make morphology release."""
    for extension in extensions:
        _zero_diameter_path = None
        if zero_diameter_path is not None:
            _zero_diameter_path = Path(f"{zero_diameter_path}-{extension[1:]}")
            _zero_diameter_path.mkdir(exist_ok=True, parents=True)

        _unravel_path = None
        if unravel_path is not None:
            _unravel_path = Path(f"{unravel_path}-{extension[1:]}")
            _unravel_path.mkdir(exist_ok=True, parents=True)

        _repair_path = None
        if repair_path is not None:
            _repair_path = Path(f"{repair_path}-{extension[1:]}")
            _repair_path.mkdir(exist_ok=True, parents=True)

        __create_db_row = partial(
            _create_db_row,
            zero_diameter_path=_zero_diameter_path,
            unravel_path=_unravel_path,
            repair_path=_repair_path,
            extension=extension,
        )

        _m = []
        with Pool() as pool:
            for name, row, m in pool.imap(__create_db_row, df.loc[df["is_valid"]].iterrows()):
                df.loc[name] = pd.Series(row)
                _m.append(m)

        db = MorphDB(_m)
        if _zero_diameter_path is not None:
            db.write(_zero_diameter_path / "neurondb.xml")
            df.loc[df["is_valid"], f"zero_diameter_morph_db_path_{extension[:1]}"] = (
                _zero_diameter_path / "neurondb.xml"
            )

        if _unravel_path is not None:
            db.write(_unravel_path / "neurondb.xml")
            df.loc[df["is_valid"], f"unravel_morph_db_path_{extension[1:]}"] = (
                _unravel_path / "neurondb.xml"
            )

        if _repair_path is not None:
            db.write(_repair_path / "neurondb.xml")
            df.loc[df["is_valid"], f"repair_morph_db_path_{extension[1:]}"] = (
                _repair_path / "neurondb.xml"
            )
