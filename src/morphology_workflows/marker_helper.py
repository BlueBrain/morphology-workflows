"""Module to read/write/plot morphology markers."""
import logging

import numpy as np
import yaml
from neurom import load_morphology
from neurom.geom import bounding_box
from plotly_helper.neuron_viewer import NeuronBuilder
from plotly_helper.object_creator import scatter
from plotly_helper.object_creator import scatter_line
from plotly_helper.object_creator import vector

logger = logging.getLogger(__name__)


class MarkerSet:
    """Container of several markers for a single morphology."""

    def __init__(self):
        """ """
        self.markers = None
        self.morph_name = None
        self.morph_path = None

    @classmethod
    def from_markers(cls, markers):
        """Load from a list of Marker object."""
        obj = cls()

        if len({marker.morph_name for marker in markers}) > 1:
            raise Exception("Markers of different morphologies were provided.")
        obj.morph_name = markers[0].morph_name

        if len({marker.morph_path for marker in markers}) > 1:
            raise Exception("Markers of different morphology paths were provided.")
        obj.morph_path = markers[0].morph_path

        obj.markers = cls.check_labels(markers)
        return obj

    @staticmethod
    def check_labels(markers):
        """Ensure label names are different."""
        labels = {}
        for marker in markers:
            if marker.label in labels:
                labels[marker.label] += 1
                marker.label += "_" + str(labels[marker.label])
            else:
                labels[marker.label] = 0
        return markers

    @classmethod
    def from_file(cls, marker_path):
        """Load maker yaml file."""
        with open(marker_path, "r", encoding="utf-8") as f:
            return MarkerSet.from_dicts(yaml.safe_load(f))

    @classmethod
    def from_dicts(cls, markers):
        """Load marker from dict, where markers["markers"] is a list of dict markers."""
        obj = cls()
        obj.markers = cls.check_labels([Marker(**marker) for marker in markers["markers"]])
        obj.morph_name = markers.get("morph_name", None)
        obj.morph_path = markers.get("morph_path", None)
        return obj

    @staticmethod
    def _plot_points_marker(builder, marker):
        """Plot point markers."""
        data = np.array(marker.data)
        if len(np.shape(data)) == 1:
            data = data[np.newaxis]
        plot_data = scatter(data[:3], name=f"{marker.label}", **marker.plot_style)
        builder.helper.add_data({f"{marker.label}": plot_data})

    @staticmethod
    def _plot_line_marker(builder, marker):
        """Plot line markers."""
        plot_data = scatter_line(marker.data, name=f"{marker.label}", **marker.plot_style)
        builder.helper.add_data({f"{marker.label}": plot_data})

    @staticmethod
    def _plot_axis_marker(builder, marker, margin=10):
        """Plot axis marker.

        This marker is a line crossing the whole bbox.
        """
        point_x = np.array(marker.data[0])
        point_y = np.array(marker.data[1])

        def _bbox(neuron, margin=10):
            """Get enlarged bbox of neuron for plotting."""
            bbox = bounding_box(neuron)
            bbox[0] -= margin * np.ones(3)
            bbox[1] += margin * np.ones(3)
            return bbox

        if not builder.neuron.neurites:
            raise ValueError("Can not plot axis marker for a neuron with no neurites.")
        bbox = _bbox(builder.neuron, margin=margin)

        def _get_lim(direction=1):
            """Find point near bbox limit in directiion of (point_y - point_x)."""
            fac = 1.0
            point = point_x.copy()
            while (point > bbox[0]).all() and (point < bbox[1]).all():
                fac += 10.0
                point = point_x - direction * fac * (point_y - point_x)
            return point

        plot_data = vector(_get_lim(1), _get_lim(-1), name=f"{marker.label}", **marker.plot_style)
        builder.helper.add_data({f"{marker.label}": plot_data})

    def plot(self, filename="markers.html", with_plotly=True, plane="3d", **kwargs):
        """Plot morphology with markers."""
        if with_plotly:

            neuron = load_morphology(self.morph_path)
            builder = NeuronBuilder(neuron, plane, line_width=4, title=f"{self.morph_name}")

            if "auto_open" not in kwargs:
                kwargs["auto_open"] = False

            for marker in self.markers:
                if marker.marker_type == "points":
                    self._plot_points_marker(builder, marker)
                elif marker.marker_type == "axis":
                    self._plot_axis_marker(builder, marker)
                elif marker.marker_type == "line":
                    self._plot_line_marker(builder, marker)
                else:
                    logger.info("marker type %s not understood", marker.type)
            builder.plot(filename=str(filename), **kwargs)
        else:
            raise NotImplementedError("Only plotly available for now")

    def to_dicts(self):
        """Create a list of dicts from marker class."""
        out_dict = {
            "morph_name": self.morph_name,
            "morph_path": self.morph_path,
            "markers": [marker.to_dict()["marker"] for marker in self.markers],
        }
        return out_dict

    def save(self, filename=None, mode="a"):
        """Save marker data.

        Args:
            filename (str): if provided,  path to marker file
            mode (str): writing mode, 'a' or 'w'

        TODO: if data is large, write in an h5/pickle for speedup
        """
        if filename is None:
            filename = self.morph_name.with_suffix(".yaml")
        with open(filename, mode, encoding="utf-8") as f:
            yaml.dump(self.to_dicts(), f)


class Marker:
    """Container of a single marker."""

    def __init__(
        self,
        label,
        marker_type,
        data,
        morph_name=None,
        morph_path=None,
        plot_style=None,
        extra_data=None,
    ):
        """Create a marker.

        Args:
            label (str): label of marker
            marker_type (str): marker types (points, line, plane, etc...)
            data (str): markere data
            morph_name (str): name of corresponding morphology
            morph_path (str): path to corresponding morphology
            plot_style (dict): custom dict for plotting arguments
            extra_data (any): additional marker information to keep (better to be yaml-friendly)
        """
        self.label = label
        self.marker_type = marker_type
        self.data = np.array(data, dtype=float)
        self.morph_name = morph_name
        self.morph_path = morph_path
        self._set_plot_style(plot_style)
        self.extra_data = extra_data

        self._check_valid()

    def _set_plot_style(self, plot_style):
        """Set plotting style."""
        self.plot_style = {"showlegend": True}
        if self.marker_type == "points":
            self.plot_style = {"color": "black", "width": 20}
        if self.marker_type == "line":
            self.plot_style = {"color": "black", "width": 1, "marker_size": 0.0001}
        if plot_style is not None:
            self.plot_style.update(plot_style)

    def _check_valid(self):
        """Check if the given marker data is valid."""
        valid = False
        if self.marker_type == "points":
            valid = len(np.shape(self.data)) == 2 or len(np.shape(self.data)) == 1
        if self.marker_type == "axis":
            valid = len(np.shape(self.data)) == 2
        if self.marker_type == "line":
            valid = len(np.shape(self.data)) == 2
        if not valid:
            raise Exception(f"Marker {self.marker_type} is not valid with data {self.data}.")

    @property
    def list_data(self):
        """Returns data without numpy types."""
        return self.data.tolist()

    def to_dict(self):
        """Create a dict from marker class."""
        return {
            "morph_name": self.morph_name,
            "morph_path": self.morph_path,
            "marker": {
                "label": self.label,
                "marker_type": self.marker_type,
                "data": self.list_data,
                "plot_style": self.plot_style,
                "extra_data": self.extra_data,
            },
        }
