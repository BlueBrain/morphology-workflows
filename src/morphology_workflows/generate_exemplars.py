"""Module to generate exemplars from a population of morphologies."""
from functools import partial
from multiprocessing.pool import Pool
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize

import numpy as np
from scipy.optimize import curve_fit
import pandas as pd

import neurom as nm
from morph_tool.neuron_surface import get_NEURON_surface
from morphio.mut import Morphology
from morph_tool.morphdb import MorphDB


def single_compartment_exemplar(morph_df, neurite_fraction=0.2):
    """We create a single compartment morphology representative of the perisomatic surface area.

    The effective radius is computed as a weighted sum of the radiuses of the different sections.
    """
    db = MorphDB()
    db.df = morph_df
    config = {
        "neurite": {"total_area": {"modes": ["sum"]}},
        "morphology": {"soma_surface_area": {"modes": ["mean"]}},
        "neurite_type": ["AXON", "APICAL_DENDRITE", "BASAL_DENDRITE"],
    }

    df_features = db.features(config)
    radius = np.sqrt(
        (
            df_features["morphology"]["mean_soma_surface_area"]
            + (
                df_features["basal_dendrite"]["sum_total_area"]
                + df_features["apical_dendrite"]["sum_total_area"]
                + df_features["axon"]["sum_total_area"]
            )
            * neurite_fraction
        )
        / (4.0 * np.pi)
    ).mean()

    m = Morphology()
    m.soma.points = [[-radius, 0, 0], [0, 0, 0], [radius, 0, 0]]
    m.soma.diameters = [2 * radius, 2 * radius, 2 * radius]
    return m


def get_bins(bin_params):
    """Compute path lengths bins from parameters."""
    _b = np.linspace(bin_params["min"], bin_params["max"], bin_params["n"] + 1)
    return [[_b[i], _b[i + 1]] for i in range(bin_params["n"] - 1)]


def bin_data(distances, data, path_bins, tpe="area"):
    """Bin data using distances."""
    for _bin in path_bins:
        bin_center = 0.5 * (_bin[0] + _bin[1])
        if tpe == "area":
            mean_data = data[(_bin[0] <= distances) & (distances < _bin[1])].sum() / (
                _bin[1] - _bin[0]
            )
        if tpe == "diameter":
            mean_data = data[(_bin[0] <= distances) & (distances < _bin[1])].mean()
        yield bin_center, mean_data


def get_surface_density(neuron_path, path_bins, neurite_type="basal", tpe="area"):
    """Compute the binned surface densities of a neuron."""

    neuron = nm.load_morphology(neuron_path)

    _types = {"apical": nm.NeuriteType.apical_dendrite, "basal": nm.NeuriteType.basal_dendrite}

    data, dists = [], []
    for neurite in neuron.neurites:
        if neurite.type == _types[neurite_type]:
            if tpe == "area":
                data += list(nm.get("segment_areas", neurite))
            if tpe == "diameter":
                for section in nm.iter_sections(neurite):
                    data += list(section.points[1:, 3])
            dists += list(nm.get("segment_path_lengths", neurite))
    return list(bin_data(np.array(dists), np.array(data), path_bins, tpe=tpe))


def get_surface_profile(df, path_bins, neurite_type="basal", morphology_path="path", tpe="area"):
    """Get surface profile."""
    surface_df = pd.DataFrame()
    with Pool() as pool:
        for gid, res in enumerate(
            pool.map(
                partial(
                    get_surface_density, path_bins=path_bins, neurite_type=neurite_type, tpe=tpe
                ),
                df[morphology_path],
            )
        ):
            for b, s in res:
                surface_df.loc[gid, b] = s
    surface_df[surface_df.isna()] = 0

    return surface_df


def get_dendrite_exemplar(df, bin_params=None, surface_percentile=50):
    """Get dendrite exemplar."""
    _bin_params = {"min": 0, "max": 500, "n": 50}
    if bin_params is not None:
        _bin_params.update(bin_params)

    path_bins = get_bins(_bin_params)
    surf_df = get_surface_profile(df, path_bins, "basal")
    surf_df += get_surface_profile(df, path_bins, "apical")
    df["dist"] = np.linalg.norm(
        (surf_df - np.percentile(surf_df, surface_percentile, axis=0)).to_numpy(), axis=1
    )
    return df.sort_values(by="dist").index[0], surf_df


def build_soma_model(morphology_paths):
    """Build soma model.

    Using only surface area for now.
    """
    soma_surfaces = [float(get_NEURON_surface(path)) for path in tqdm(morphology_paths)]
    soma_radii = [
        float(nm.get("soma_radius", nm.load_morphology(path))) for path in morphology_paths
    ]
    return {
        "soma_model": {
            "soma_surface": float(np.mean(soma_surfaces)),
            "soma_radius": float(np.mean(soma_radii)),
        },
        "soma_data": {
            "soma_radii": soma_radii,
            "soma_surfaces": soma_surfaces,
        },
    }


def get_ais(neuron):
    """Get the axon initial section of a neuron."""
    for neurite in neuron.root_sections:
        if neurite.type == nm.NeuriteType.axon:
            return neurite
    return None


def extract_ais_diameters(morphologies):
    """Produce an iterator on ais diameters."""
    for neuron in morphologies:
        ais = get_ais(neuron)
        if ais is not None:
            yield ais.diameters


def extract_ais_path_distances(morphologies):
    """Produce an iterator on ais diameters."""
    for neuron in morphologies:
        ais = get_ais(neuron)
        if ais is not None:
            yield np.insert(
                np.linalg.norm(np.cumsum(np.diff(ais.points, axis=0), axis=0), axis=1), 0, 0
            )


def taper_function(length, strength, taper_scale, terminal_diameter):
    """Function to model tappers AIS."""
    return strength * np.exp(-length / taper_scale) + terminal_diameter


def build_ais_diameter_model(morphology_paths, bin_size=2, total_length=60, with_taper=False):
    """Build the AIS model by fitting first sections of axons.

    Args:
        morphology_paths (list): list of paths to morphologies
        bin_size (float): size of bins (in unit length) for smoothing of diameters
        total_length (flow): length of AIS
    """
    morphologies = [Morphology(str(morphology)) for morphology in morphology_paths]

    distances, diameters = [], []
    for dist, diams in zip(
        extract_ais_path_distances(morphologies), extract_ais_diameters(morphologies)
    ):
        distances += list(dist)
        diameters += list(diams)

    all_bins = np.arange(0, total_length, bin_size)
    indices = np.digitize(np.array(distances), all_bins, right=False)

    means, stds, bins = [], [], []
    for i in list(set(indices)):
        diams = np.array(diameters)[indices == i]
        means.append(np.mean(diams))
        stds.append(np.std(diams))
        bins.append(all_bins[i - 1])
    bounds = [3 * [-np.inf], 3 * [np.inf]]
    if not with_taper:
        bounds[0][0] = 0.0
        bounds[1][0] = 0.000001

    popt, _ = curve_fit(taper_function, np.array(bins), np.array(means), bounds=bounds)[:2]

    model = {}
    # first value is the length of AIS
    model["ais_model"] = {
        "popt_names": ["length"] + list(taper_function.__code__.co_varnames[1:]),
        "popt": [total_length] + popt.tolist(),
    }

    model["ais_data"] = {
        "distances": np.array(distances).tolist(),
        "diameters": np.array(diameters).tolist(),
        "bins": np.array(bins).tolist(),
        "means": np.array(means).tolist(),
        "stds": np.array(stds).tolist(),
    }
    return model


def plot_soma_shape_models(models, pdf_filename="soma_shape_models.pdf"):
    """Plot soma shape models (surface area and radii)."""
    with PdfPages(pdf_filename) as pdf:
        plt.figure()
        plt.hist(models["soma_data"]["soma_surfaces"], bins=20, label="data")
        plt.axvline(models["soma_model"]["soma_surface"], label="model", c="k")
        plt.xlabel("soma surface (NEURON)")
        plt.legend()
        pdf.savefig()

        plt.figure()
        plt.hist(models["soma_data"]["soma_radii"], bins=20, label="data")
        plt.axvline(models["soma_model"]["soma_radius"], label="model", c="k")
        plt.xlabel("soma radii (NeuroM)")
        plt.legend()
        pdf.savefig()


def plot_ais_taper(data, model, ax=None):
    """Plot AIS taper."""
    if ax is None:
        fig = plt.figure(figsize=(5, 4))
        ax = plt.gca()
    else:
        fig = ax.get_figure()

    ax.scatter(data["distances"], data["diameters"], marker=".", c="0.5", s=2, rasterized=True)

    ax.plot(data["bins"], data["means"], "C0")
    ax.plot(data["bins"], np.array(data["means"]) - np.array(data["stds"]), "C0--")
    ax.plot(data["bins"], np.array(data["means"]) + np.array(data["stds"]), "C0--")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, np.max(data["diameters"]))
    ax.set_xlabel("Distance from soma")
    ax.set_ylabel("AIS diameter")

    ax.plot(
        data["bins"],
        taper_function(np.array(data["bins"]), *model["ais_model"]["popt"][1:]),
        c="C1",
    )
    d = model["ais_model"]["popt"][1:]
    ax.set_title(
        f"Taper strength: {d[0]}, \n scale: {d[1]},\n terminal diameter: {d[2]}",
        fontsize=10,
        loc="left",
    )
    if ax is None:
        fig.savefig("ais_diameters.png", bbox_inches="tight")


def plot_ais_taper_models(models, pdf_filename="AIS_models.pdf"):
    """Create a pdf with all models of AIS and datapoints."""
    plt.figure()
    plot_ais_taper(models["ais_data"], models, ax=plt.gca())
    plt.savefig(pdf_filename)


def plot_surface_comparison(surf_df, df, pdf_filename="surface_profile.pdf", surface_percentile=50):
    """Plot comparison of surface areas and median scores."""
    with PdfPages(pdf_filename) as pdf:
        mean = np.percentile(surf_df, surface_percentile, axis=0)
        if "median_score" in df.columns:
            cmappable = plt.cm.ScalarMappable(
                norm=Normalize(df.median_score.min(), df.median_score.max()), cmap="plasma"
            )

        plt.figure(figsize=(7, 4))
        for gid in surf_df.index:
            if "median_score" in df.columns:
                c = cmappable.to_rgba(df.loc[gid, "median_score"])
            else:
                c = "k"
            plt.plot(surf_df.columns, surf_df.loc[gid], c=c, lw=0.5)

        plt.plot(surf_df.columns, mean, c="r", lw=3, label="mean area")
        if "is_exemplar" in df.columns:
            plt.plot(
                surf_df.columns,
                surf_df[df["is_exemplar"] == 1].to_numpy()[0],
                c="g",
                lw=3,
                label="exemplar",
            )
        if "median_score" in df.columns:
            plt.colorbar(cmappable, label="median score")
        plt.xlabel("path distance")
        plt.ylabel("surface area")
        plt.legend()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        plt.figure(figsize=(7, 4))
        for gid in surf_df.index:
            if "median_score" in df.columns:
                c = cmappable.to_rgba(df.loc[gid, "median_score"])
            else:
                c = "k"
            df.loc[gid, "diffs"] = np.linalg.norm(surf_df.loc[gid] - mean)
            plt.plot(surf_df.columns, surf_df.loc[gid] - mean, c=c, lw=0.5)
        if "is_exemplar" in df.columns:
            plt.plot(
                surf_df.columns,
                surf_df.loc[df["is_exemplar"] == 1].to_numpy()[0] - mean,
                c="g",
                lw=3,
                label="exemplar",
            )
        if "median_score" in df.columns:
            plt.colorbar(cmappable, label="median score")
        plt.xlabel("path distance")
        plt.ylabel("surface area - mean(surface area)")
        plt.legend()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        if "median_score" in df.columns:
            clip = 5
            plt.figure(figsize=(5, 3))
            plt.scatter(np.clip(df.median_score, 0, clip), df.diffs, label="adapted AIS/soma", s=10)
            plt.scatter(
                np.clip(df.generic_median_score, 0, clip),
                df.diffs,
                label="original AIS/SOMA",
                s=20,
                marker="+",
            )
            plt.legend(loc="best")
            plt.xlabel("median score")
            plt.ylabel("norm(surface area - mean)")
            plt.tight_layout()
            pdf.savefig()

            clip = 20
            plt.figure(figsize=(5, 3))
            plt.scatter(np.clip(df.max_score, 0, clip), df.diffs, label="adapted AIS/soma", s=10)
            plt.scatter(
                np.clip(df.generic_max_score, 0, clip),
                df.diffs,
                label="original AIS/SOMA",
                s=20,
                marker="+",
            )
            plt.legend(loc="best")
            plt.xlabel("max score")
            plt.ylabel("norm(surface area - mean)")
            plt.tight_layout()
            pdf.savefig()

            plt.figure(figsize=(5, 3))
            clip = 5
            plt.plot([0, clip], [0, clip], "k", lw=0.5)
            plt.scatter(
                np.clip(df.generic_median_score, 0, clip), np.clip(df.median_score, 0, clip), s=3
            )
            plt.xlabel("median original score")
            plt.ylabel("median adapted score")
            plt.axvline(2, ls="--", c="k", lw=0.5)
            plt.axhline(2, ls="--", c="k", lw=0.5)
            plt.axis([0, clip + 0.5, 0, clip + 0.5])
            plt.tight_layout()
            pdf.savefig()

            plt.figure(figsize=(5, 3))
            clip = 20
            plt.plot([0, clip], [0, clip], "k", lw=0.5)
            plt.scatter(np.clip(df.generic_max_score, 0, clip), np.clip(df.max_score, 0, clip), s=3)
            plt.xlabel("max original score")
            plt.ylabel("max adapted score")
            plt.axvline(5, ls="--", c="k", lw=0.5)
            plt.axhline(5, ls="--", c="k", lw=0.5)
            plt.axis([0, clip + 0.5, 0, clip + 0.5])
            plt.tight_layout()
            pdf.savefig()

        plt.close("all")


def full_exemplar(df, surface_percentile=50, bin_params=None, figure_folder="full_exemplar"):
    """We create a full exemplar from the population."""
    best_gid, surf_df = get_dendrite_exemplar(
        df, surface_percentile=surface_percentile, bin_params=bin_params
    )

    soma_model = build_soma_model(df["path"])
    ais_model = build_ais_diameter_model(df["path"])

    plot_soma_shape_models(soma_model, pdf_filename=f"{figure_folder}/soma_shape_model.pdf")
    plot_ais_taper_models(ais_model, pdf_filename=f"{figure_folder}/AIS_model.pdf")
    df["is_exemplar"] = 0
    df.loc[best_gid, "is_exemplar"] = 1
    plot_surface_comparison(
        surf_df,
        df,
        pdf_filename=f"{figure_folder}/surface_profile.pdf",
        surface_percentile=surface_percentile,
    )

    return {
        "soma": soma_model["soma_model"],
        "ais": ais_model["ais_model"],
        "path": df.loc[best_gid, "path"].resolve(),
    }
