"""Curation functions."""
import json
import logging
import shutil
from functools import partial
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_validation_framework.result import ValidationResult
from matplotlib.backends.backend_pdf import PdfPages
from morph_tool import transform
from morph_tool.resampling import resample_linear_density
from morph_tool.transform import align_morphology
from morph_tool.transform import rotate
from morph_tool.transform import rotation_matrix_from_vectors
from morphio import PointLevel
from morphio import SectionType
from morphio.mut import Morphology
from neurom import COLS
from neurom import load_morphology
from neurom import view
from neurom.check import CheckResult
from neurom.check import morphology_checks as nc
from neurom.core.morphology import iter_sections
from neurom.core.soma import SomaSinglePoint
from neurom.geom import bounding_box
from neuror.sanitize import _ZERO_LENGTH
from neuror.sanitize import CorruptedMorphology
from neuror.sanitize import annotate_neurolucida
from neuror.sanitize import fix_points_in_soma
from neuror.sanitize import sanitize as _sanitize
from plotly_helper.neuron_viewer import NeuronBuilder

from morphology_workflows.marker_helper import Marker
from morphology_workflows.marker_helper import MarkerSet
from morphology_workflows.utils import disable_loggers
from morphology_workflows.utils import is_morphology

L = logging.getLogger(__name__)
matplotlib.use("Agg")


def collect(row, data_dir, morph_path_col="morph_path"):
    """Collect morphologies."""
    L.debug("Collecting %s from %s", row.name, row[morph_path_col])
    if not Path(row[morph_path_col]).exists():
        return ValidationResult(
            is_valid=False,
            comment=f"The file {row[morph_path_col]} does not exist.",
            morph_path=None,
        )
    is_morph, ext = is_morphology(row[morph_path_col])
    if is_morph:
        if " " in row.name:
            return ValidationResult(
                is_valid=False,
                comment=f"{row[morph_path_col]} has spaces in its name, which is not supported.",
                morph_path=None,
            )
        full_new_morph_path = str(data_dir / row.name) + ext
        shutil.copy(row[morph_path_col], full_new_morph_path)
        return ValidationResult(is_valid=True, morph_path=full_new_morph_path)
    return ValidationResult(
        is_valid=False,
        comment=f"The file {row[morph_path_col]} is not a morphology file or is corrupted.",
        morph_path=None,
    )


def _get_markers(row, morph):
    """Get marker data from morphio object to list of dicts."""
    if not hasattr(morph, "markers"):
        return []
    markers = []
    for marker in morph.markers:
        if marker.label.startswith("INCOMPLETE"):
            points = [morph.sections[marker.section_id].points[-1]]
        else:
            points = marker.points
        markers.append(
            Marker(
                marker.label,
                "points",
                points,
                morph_name=row.name,
                morph_path=row.morph_path,
            )
        )
    return markers


def extract_marker(row, data_dir):
    """Extract marker data from morphology files."""
    markers = _get_markers(row, Morphology(row.morph_path))
    marker_path = None
    if len(markers) > 0:
        marker_path = (data_dir / row.name).with_suffix(".yaml")
        MarkerSet.from_markers(markers).save(filename=marker_path)
    return ValidationResult(is_valid=True, marker_path=marker_path)


def plot_markers(row, data_dir, with_plotly=True):
    """Plot markers on morphologies."""
    plot_path = None
    if not row.isnull()["marker_path"]:
        neuron = load_morphology(row.morph_path)
        if neuron.soma.center is None:
            return ValidationResult(
                is_valid=True,
                ret_code=2,
                comment="Can not plot markers for morphologies with no soma.",
                plot_marker_path=None,
            )
        if with_plotly:
            plot_path = (data_dir / row.name).with_suffix(".html")
            MarkerSet.from_file(row.marker_path).plot(filename=plot_path)
        else:
            plt.figure()
            ax = plt.gca()
            view.plot_morph(neuron, ax, realistic_diameters=True, soma_outline=False)
            markers = MarkerSet.from_file(row.marker_path).markers

            for marker in markers:
                points = marker.data
                if marker.label.startswith("INCOMPLETE"):
                    c = "r"
                    ms = 0.5
                else:
                    c = marker.plot_style.get("color", "r")
                    ms = marker.plot_style.get("width", 5) / 10
                if marker.marker_type == "points":
                    style = "o"
                if marker.marker_type == "line":
                    style = "-"
                plt.plot(points[:, 0], points[:, 1], style, c=c, ms=ms)
            plot_path = (data_dir / row.name).with_suffix(".pdf")
            plt.savefig(plot_path)
            plt.close()
    return ValidationResult(is_valid=True, plot_marker_path=plot_path)


def _reconnect_to_soma(m):
    """Reconnect a neurite to soma if it is not."""

    def _copy(section, section_base):
        """Recursively copy downstream from section_base to section."""
        for base_child in section_base.children:
            section.append_section(base_child)
        for child, base_child in zip(section.children, section_base.children):
            _copy(child, base_child)

    for root in m.root_sections:
        for section in root.iter():
            if section.type != root.type:
                L.warning("We found root sections not at soma, reconnecting.")
                new_sec = m.append_root_section(section)
                _copy(new_sec, section)
                m.delete_section(section, recursive=True)
                return True
    return False


def sanitize(row, data_dir, ensure_roots_at_soma=True):
    """Sanitize morphologies.

    With `ensure_roots_at_soma` we disconnect neurites not at the soma and connect
    at the soma. This may happen for dendrite bearing axons morphologies in .swc format.
    Weirdness may happen, for example upon converting to .asc, the axon will become a basal.
    """
    new_morph_path = data_dir / Path(row.morph_path).name
    m = Morphology(row.morph_path)
    if ensure_roots_at_soma:
        retry = True
        while retry:
            if not _reconnect_to_soma(m):
                # if sanitize fails for another reason, we let it crash after
                break
            try:
                _sanitize(m, new_morph_path)
                retry = False
            except CorruptedMorphology:
                retry = True

    try:
        _sanitize(m, new_morph_path)
    except CorruptedMorphology as exc:
        return ValidationResult(is_valid=False, comment=exc, morph_path=None)
    return ValidationResult(is_valid=True, morph_path=new_morph_path)


def _center_root_points(morph):
    root_points = np.array([section.points[0, COLS.XYZ] for section in morph.root_sections])
    center = np.mean(root_points, axis=0)
    dists = np.linalg.norm(root_points - center, axis=1)
    radius = max(1.0, dists.mean())
    return center, radius, root_points


def _add_soma(morph, soma_type="spherical"):
    """Add a mock soma centered around first points with radius mean distance to them."""
    if len(morph.soma.points) == 0:
        center, radius, root_points = _center_root_points(morph)
        if soma_type == "spherical":
            morph.soma.points = [center.tolist()]
            morph.soma.diameters = [2.0 * radius]
            L.info("Adding a spherical mock soma at %s of radius %s.", center, radius)
        elif soma_type == "contour":
            # Order contour points by polar angle
            relative_pts = root_points - center
            angles = np.arctan(relative_pts[:, COLS.Y] / relative_pts[:, COLS.X])
            angle_order = np.argsort(angles)

            morph.soma.points = root_points[angle_order]
            morph.soma.diameters = np.zeros(len(root_points), dtype=float)
            L.info("Adding a contour mock soma around %s with %s points.", center, len(root_points))

    return morph


def _has_axon(morph_path, n_section_min=5):
    """Check if neuron has axon with strictly more than n_section_min sections."""
    _morph = Morphology(morph_path)
    for root in _morph.root_sections:
        if root.type == SectionType.axon:
            if len(list(root.iter())) > n_section_min:
                return True
    return False


def _has_basal(morph_path):
    """Check if neuron has basal dendrites."""
    _morph = Morphology(morph_path)
    for root in _morph.root_sections:
        if root.type == SectionType.basal_dendrite:
            return True
    return False


def _has_apical(morph_path):
    """Check if neuron has apical dendrites.

    TODO: maybe here we can detect if multiple apicals
    """
    _morph = Morphology(morph_path)
    for root in _morph.root_sections:
        if root.type == SectionType.apical_dendrite:
            return True
    return False


def _add_stub_axon(morph, length=100, diameter=1.0):
    """Add a stub axon to a morphology."""
    stub_orig = morph.soma.points[np.argmin(morph.soma.points[:, COLS.Y])]
    stub = PointLevel([stub_orig, stub_orig - np.array([0, length, 0])], 2 * [diameter])
    morph.append_root_section(stub, SectionType.axon)


def _children_direction(
    section,
    min_length=_ZERO_LENGTH,
    starting_point=None,
    remove_intermediate_pts=True,
):
    """Compute the mean direction of children of a given section."""
    if starting_point is None:
        starting_point = section.points[-1]

    direction = np.zeros(3)
    for child in section.children:
        child_direction_norm = 0
        imax = 0
        for i in range(1, len(child.points)):
            child_direction = child.points[i] - starting_point
            child_direction_norm = np.linalg.norm(child_direction)
            if child_direction_norm >= min_length:
                child_direction /= child_direction_norm
                imax = i
                break

        if child_direction_norm < min_length:
            # If the child section is too small, then the direction is derived from the
            # grand-children
            child_direction = _children_direction(child, min_length, starting_point=starting_point)

        if remove_intermediate_pts and imax > 1:
            # Remove intermediate points that are in the min_length redius
            imax = min(len(child.points) - 1, imax)
            child.points = np.vstack([child.points[0], child.points[imax:]])
            if len(child.diameters) > 0:
                child.diameters = np.vstack([child.diameters[0], child.diameters[imax:]])
            if len(child.perimeters) > 0:
                child.perimeters = np.vstack([child.perimeters[0], child.perimeters[imax:]])
        direction += child_direction

    direction /= np.linalg.norm(direction)

    return direction


def _move_children(section, shift, min_length=_ZERO_LENGTH):
    """Move the children of a given section by a given shift."""
    for child in section.children:
        child_norm = np.linalg.norm(child.points[0] - child.points[-1])
        child_points = child.points
        child_points[0] += shift
        if child_norm < min_length:
            try:
                child_points[1] += shift
            except IndexError:
                # A section can have only 1 point!
                pass
            _move_children(child, shift, min_length)
        child.points = child_points


def _float_formatter(x):
    return str(x)


def fix_root_section(morph, min_length=_ZERO_LENGTH):
    """Ensures that each neurite has a root section with non-zero length."""
    if min_length is None:
        return

    to_delete = []

    for root_section in morph.root_sections:
        root_section_points = root_section.points
        if (
            len(root_section_points) == 2
            and np.linalg.norm(np.diff(root_section_points[:2], axis=0)[0]) < min_length
        ):
            if root_section.children:
                direction = _children_direction(root_section, min_length, root_section.points[-1])
            else:
                # In some cases a 0-length section has no child so the direction has NaN coordinates
                direction = root_section_points[0] - morph.soma.center
                if (direction == 0).all():
                    # If the direction is still not correct, the section is deleted
                    to_delete.append(root_section)
                    continue
                direction /= np.linalg.norm(direction)

            new_point = (root_section_points[0] + direction * min_length).astype(
                root_section_points.dtype
            )
            if (new_point == root_section_points[1]).all():
                # If the point was not moved because min_length is too small for the current
                # precision, the smallest movement is ensured
                _formatter = {"float": _float_formatter}

                new_point = np.nextafter(
                    root_section_points[1],
                    root_section_points[1] * (1 + direction),
                    dtype=root_section_points.dtype,
                )
                L.debug(
                    "The min_length was too small to move the point %s so it was moved to %s",
                    np.array2string(root_section_points[1], separator=", ", formatter=_formatter),
                    np.array2string(new_point, separator=", ", formatter=_formatter),
                )
            root_section_points[1] = new_point
            shift = root_section_points[1] - root_section.points[1]
            root_section.points = root_section_points

            _move_children(root_section, shift, min_length)

    for sec in to_delete:
        morph.delete_section(sec)


def check_neurites(
    row,
    data_dir,
    axon_n_section_min=5,
    mock_soma_type="spherical",
    ensure_stub_axon=False,
    min_length_first_section=_ZERO_LENGTH,
):
    """Check which neurites are present, add soma if missing and mock_soma_type is not None."""
    new_morph_path = data_dir / Path(row.morph_path).name
    morph = Morphology(row.morph_path)
    if mock_soma_type is not None:
        _add_soma(morph, mock_soma_type)
    if ensure_stub_axon:
        if not _has_axon(row.morph_path, n_section_min=0):
            _add_stub_axon(morph)
    fix_root_section(morph, min_length_first_section)
    morph.write(new_morph_path)
    has_axon = row.get("use_axon", _has_axon(row.morph_path, n_section_min=axon_n_section_min))
    has_basal = row.get("use_dendrites", _has_basal(row.morph_path))
    has_apical = row.get("use_dendrites", _has_apical(row.morph_path))

    return ValidationResult(
        is_valid=True,
        morph_path=new_morph_path,
        has_axon=has_axon,
        has_basal=has_basal,
        has_apical=has_apical,
    )


def fix_soma_radius(morph):
    """Fix the radius of a spherical soma.

    If all points of a neurite are located inside the soma, the radius is reduced such that at
    least the last point of each neurite is located outside the soma.
    """
    if isinstance(morph.soma, SomaSinglePoint):
        _init_points = []
        _before_last_points = []
        _last_points = []
        for section in morph.root_sections:
            _init_points.append(section.points[0])
            _before_last_points.append(section.points[-2])
            _last_points.append(section.points[-1])
        _init_points = np.array(_init_points)
        _before_last_points = np.array(_before_last_points)
        _last_points = np.array(_last_points)

        # Compute radius
        radius = np.max([1.0, morph.soma.radius])

        # If there are entire sections inside the soma, the radius is reduced
        last_pt_dists = np.linalg.norm(_last_points - morph.soma.center, axis=1)
        sec_in_soma = last_pt_dists <= radius
        if sec_in_soma.any():
            before_last_pt_dists = np.linalg.norm(_before_last_points - morph.soma.center, axis=1)
            radius = (
                np.min([last_pt_dists, before_last_pt_dists])
                + np.abs(
                    (last_pt_dists[sec_in_soma] - before_last_pt_dists[sec_in_soma]) * 0.5
                ).min()
            )

        if radius < morph.soma.radius or morph.soma.radius < 1.0:
            former_radius = morph.soma.radius
            morph.soma.radius = radius
            return former_radius, radius

    return morph.soma.radius, None


def fix_neurites_in_soma(row, data_dir):
    """Fix neurites whose points are located inside the soma.

    Method:
        - The radius of the soma is adapted to reduce the number of points inside the soma.
        - The remaining points inside the soma are removed.
    """
    new_morph_path = data_dir / Path(row.morph_path).name

    ret_code = 0
    comment = None
    morph = load_morphology(row.morph_path)
    if isinstance(morph.soma, SomaSinglePoint):
        former_radius, new_radius = fix_soma_radius(morph)
        if new_radius is not None:
            ret_code = 2
            comment = f"Soma radius was changed from {former_radius} to {new_radius}."
        if fix_points_in_soma(morph):
            ret_code += 20
            if comment is None:
                comment = ""
            else:
                comment += " - "
            comment += "Section points located in soma were removed."
        if comment:
            L.warning("%s - %s", row.name, comment)

    morph.write(new_morph_path)

    return ValidationResult(
        is_valid=True,
        ret_code=ret_code,
        comment=comment,
        morph_path=new_morph_path,
    )


def recenter(row, data_dir):
    """Recenter morphologies to place soma at [0, 0, 0]."""
    new_morph_path = data_dir / Path(row.morph_path).name
    morph = Morphology(row.morph_path)

    location = morph.soma.center
    transform.translate(morph, -1 * location)
    morph.write(new_morph_path)
    return ValidationResult(
        is_valid=True, morph_path=new_morph_path, soma_location=json.dumps(location.tolist())
    )


def orient(row, data_dir, pia_direction="y"):
    """Orient a morphology such that the original pia_direcion is along y."""
    new_morph_path = data_dir / Path(row.morph_path).name
    _convert = {"x": [1.0, 0, 0], "z": [0.0, 0.0, 1.0]}
    morph = Morphology(row.morph_path)
    if pia_direction != "y":
        rotation_matrix = rotation_matrix_from_vectors(_convert[pia_direction], [0.0, 1.0, 0.0])
        rotate(morph, rotation_matrix)
    morph.write(new_morph_path)
    return ValidationResult(is_valid=True, morph_path=new_morph_path)


def align(
    row,
    data_dir,
    method="whole",
    neurite_type="apical",
    direction=None,
    custom_orientation_json_path=None,
):
    """Align a morphology."""
    new_morph_path = data_dir / Path(row.morph_path).name
    morph = Morphology(row.morph_path)

    if method == "custom":
        if custom_orientation_json_path is None:
            raise ValueError(
                "Provide a custom_orientation_json_path parameter when method=='custom'"
            )
        with open(custom_orientation_json_path, "r", encoding="utf-8") as orient_file:
            orient_dict = json.load(orient_file)
        if row.name in orient_dict:
            direction = orient_dict[row.name]
            rotation_matrix = rotation_matrix_from_vectors(direction, [0.0, 1.0, 0.0])
            rotate(morph, rotation_matrix)
        else:
            rotation_matrix = np.eye(3)
    else:
        rotation_matrix = align_morphology(
            morph, method=method, neurite_type=neurite_type, direction=direction
        )
    morph.write(new_morph_path)
    return ValidationResult(
        is_valid=True,
        morph_path=new_morph_path,
        rotation_matrix=json.dumps(rotation_matrix.tolist()),
    )


def _convert_error_markers(row, error_markers):
    markers = []
    for error_marker in error_markers:
        markers.append(
            Marker(
                error_marker["name"],
                "points",
                np.array([p[:3] for _, _points in error_marker["data"] for p in _points]),
                morph_name=row.name,
                morph_path=row.morph_path,
                plot_style={"color": error_marker["color"].lower(), "width": 3},
            )
        )

    return markers


def z_range(neuron, min_range=50):
    """Checker for z-range."""
    max_id = None
    min_id = None
    _max = -1e10
    _min = 1e10
    for sec in iter_sections(neuron):
        _max_point = sec.points[sec.points[:, COLS.Z].argmax()]
        _min_point = sec.points[sec.points[:, COLS.Z].argmin()]
        if _max_point[COLS.Z] > _max:
            max_id = (sec.id, [_max_point])
            _max = _max_point[COLS.Z]
        if _min_point[COLS.Z] < _min:
            min_id = (sec.id, [_min_point])
            _min = _min_point[COLS.Z]
    return CheckResult(abs(_max - _min) > min_range, [min_id, max_id])


def detect_errors(row, data_dir, min_range=50):
    """Detect errors in morphologies.

    TODO: bypass dangling if only one axon/neurite
    """
    checkers = {
        nc.has_no_fat_ends: {"name": "fat end", "label": "Circle3", "color": "Blue"},
        partial(nc.has_no_jumps, axis="z"): {
            "name": "zjump",
            "label": "Circle2",
            "color": "Green",
        },
        nc.has_no_narrow_start: {"name": "narrow start", "label": "Circle1", "color": "Blue"},
        nc.has_no_dangling_branch: {"name": "dangling", "label": "Circle6", "color": "Magenta"},
        nc.has_multifurcation: {
            "name": "Multifurcation",
            "label": "Circle8",
            "color": "Yellow",
        },
        nc.has_unifurcation: {"name": "unifurcation", "label": "Circle8", "color": "Magenta"},
        partial(z_range, min_range=min_range): {
            "name": "z_range",
            "label": "Circle2",
            "color": "Red",
        },
    }

    annotations, error_summary, error_markers = annotate_neurolucida(
        row.morph_path, checkers=checkers
    )
    markers = _convert_error_markers(row, error_markers)
    error_marker_path = None
    if len(markers) > 0:
        error_marker_path = (data_dir / row.name).with_suffix(".yaml")
        MarkerSet.from_markers(markers).save(filename=error_marker_path)

    new_morph_path = data_dir / Path(row.morph_path).name
    shutil.copy(row.morph_path, new_morph_path)
    with open(new_morph_path, "a", encoding="utf-8") as morph_file:
        morph_file.write(annotations)

    return ValidationResult(
        is_valid=True,
        error_marker_path=error_marker_path,
        error_annotated_path=new_morph_path,
        error_summary=json.dumps(error_summary),
    )


def plot_errors(row, data_dir, with_plotly=True):
    """Plot error markers."""
    plot_path = None
    if not row.isnull()["error_marker_path"]:
        if with_plotly:
            plot_path = (data_dir / row.name).with_suffix(".html")
            MarkerSet.from_file(row.error_marker_path).plot(filename=plot_path)

    return ValidationResult(is_valid=True, plot_errors_path=plot_path)


def make_error_report(df, data_dir, error_report_path="error_report.csv"):
    """Make the error report as a df and save to .csv."""
    markers_df = pd.DataFrame()
    df = df[~df.isnull()["error_marker_path"]]
    for morph_name, row in df.iterrows():
        for marker in MarkerSet.from_file(row.error_marker_path).markers:
            if marker.label == "z_range":
                info = marker.data[1, 2] - marker.data[0, 2]
            else:
                info = len(marker.data)

            markers_df.loc[morph_name, marker.label] = info
    markers_df.to_csv(data_dir.parent.parent / error_report_path)


def resample(row, data_dir, linear_density=1.0):
    """Resample morphologies with fixed linear density."""
    new_morph_path = data_dir / Path(row.morph_path).name
    morph = Morphology(row.morph_path)

    resample_linear_density(morph, linear_density).write(new_morph_path)
    return ValidationResult(is_valid=True, morph_path=new_morph_path)


def plot_morphology(row, data_dir, with_plotly=True, realistic_diameters=True):
    """Plot a morphology."""
    neuron = load_morphology(row.morph_path)

    if with_plotly:
        plot_path = (data_dir / row.name).with_suffix(".html")
        builder = NeuronBuilder(neuron, "3d", line_width=4, title=f"{row.name} ({row.mtype})")
        builder.plot(filename=str(plot_path), auto_open=False)
        return ValidationResult(is_valid=True, plot_path=plot_path)

    try:
        bbox = bounding_box(neuron)
    except (ValueError, IndexError):
        bbox = np.array([[-1000, -1000, 1000], [1000, 1000, 1000]])

    plot_path = (data_dir / row.name).with_suffix(".pdf")
    with disable_loggers("matplotlib.font_manager", "matplotlib.backends.backend_pdf"), PdfPages(
        plot_path
    ) as pdf:
        for plane, axis in {"xy": [0, 1], "xz": [0, 2], "yz": [1, 2]}.items():
            plt.figure()
            view.plot_morph(
                neuron,
                plt.gca(),
                realistic_diameters=realistic_diameters,
                plane=plane,
                soma_outline=False,
            )
            plt.axis(bbox[:, axis].T.flatten())
            plt.axis("equal")
            if hasattr(row, "mtype"):
                plt.suptitle(plane + "mtype:" + row.mtype)
            else:
                plt.suptitle(plane)
            pdf.savefig()
            plt.close()

    return ValidationResult(is_valid=True, plot_path=plot_path)
