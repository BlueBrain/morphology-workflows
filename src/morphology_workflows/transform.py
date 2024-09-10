"""Transformation functions (mousification, etc...).

adapted from: /gpfs/bbp.cscs.ch/project/proj66/morphologies
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import neurom as nm
import numpy as np
import pandas as pd
import seaborn as sns
from data_validation_framework.result import ValidationResult
from matplotlib.backends.backend_pdf import PdfPages
from morphio import SectionType
from neurom import view
from neurom.apps.morph_stats import extract_dataframe
from neurom.core.types import tree_type_checker
from neurom.features.section import branch_order
from scipy.optimize import curve_fit
from tqdm import tqdm

mpl.use("Agg")

# pylint: disable=unused-argument


def learn_soma_transform(
    df,
    data_dir,
    method="ratio_of_means",
    per_mtype=False,
    target_dataset="target_dataset.csv",
    plot=True,
):
    """Estimate the scaling of soma size.

    based on https://bbpteam.epfl.ch/project/issues/browse/MOUSSCX-12
    """
    target_df = pd.read_csv(target_dataset)
    if per_mtype:
        # TODO: implement a filter and loop mtype
        pass

    if method == "ratio_of_means":
        config = {"morphology": {"soma_radius": ["mean"]}, "neurite_type": ["ALL"], "neurite": {}}
        source_mean_radii = extract_dataframe(df.morph_path.to_list(), config).morphology
        target_mean_radii = extract_dataframe(target_df.morph_path.to_list(), config).morphology
        mean_target_radius = target_mean_radii.mean_soma_radius.mean()
        mean_source_radius = source_mean_radii.mean_soma_radius.mean()
        df["soma_transform"] = mean_target_radius / mean_source_radius

        if plot:
            plt.figure()
            plt.hist(
                source_mean_radii.mean_soma_radius.to_list(),
                bins=30,
                histtype="step",
                label="source",
                color="C0",
                density=True,
            )
            plt.axvline(mean_source_radius, c="C0", ls="--")
            plt.hist(
                target_mean_radii.mean_soma_radius.to_list(),
                bins=30,
                histtype="step",
                label="target",
                color="C1",
                density=True,
            )
            plt.legend(loc="best")
            plt.axvline(mean_target_radius, c="C1", ls="--")
            plt.xlabel("soma radius")
            plt.suptitle(f"transform factor: {mean_target_radius / mean_source_radius}")
            plt.savefig(f"{list(data_dir.parents)[1]}/soma_transform.png", bbox_inches="tight")
            plt.close()


def learn_morphology_transform(
    df,
    data_dir,
    method="cortical",
    per_mtype=False,
    target_dataset="target_dataset.csv",
    extra_data=None,
    plot=True,
):
    """Estimate the global scaling of neurites.

    Either using predefined scaling with method='cortical',
    or estimate from dendrite extent with method='extents_no_axon'.

    based on https://bbpteam.epfl.ch/project/issues/browse/MOUSSCX-31
    """
    if per_mtype:
        # TODO: implement a filter and loop mtype
        pass

    if method == "cortical":
        df["morphology_transform"] = extra_data["scaling"]

    if method == "extents_no_axon":
        target_df = pd.read_csv(target_dataset)
        target_reach = []
        for gid in target_df.index:
            morph = nm.load_morphology(target_df.loc[gid, "morph_path"])
            for section in morph.root_sections:
                if section.type == SectionType.axon:
                    morph.delete_section(section, recursive=True)

            origin = morph.soma.center[np.newaxis, nm.COLS.XYZ]
            target_reach.append(
                np.median(np.linalg.norm(morph.points[:, nm.COLS.XYZ] - origin, axis=1))
            )

        source_reach = []
        for gid in df.index:
            morph = nm.load_morphology(df.loc[gid, "morph_path"])
            for section in morph.root_sections:
                if section.type == SectionType.axon:
                    morph.delete_section(section, recursive=True)
            origin = morph.soma.center[np.newaxis, nm.COLS.XYZ]
            source_reach.append(
                np.median(np.linalg.norm(morph.points[:, nm.COLS.XYZ] - origin, axis=1))
            )

        mean_source_reach = np.mean(source_reach)
        mean_target_reach = np.mean(target_reach)
        df["morphology_transform"] = mean_target_reach / mean_source_reach

        if plot:
            plt.figure()
            plt.hist(source_reach, bins=30, label="source", histtype="step", color="C0")
            plt.axvline(mean_source_reach, c="C0", ls="--")
            plt.hist(target_reach, bins=30, label="target", histtype="step", color="C1")
            plt.axvline(mean_target_reach, c="C1", ls="--")

            plt.legend(loc="best")
            plt.xlabel("median(radial distances)")
            plt.suptitle(f"transform factor: {mean_target_reach / mean_source_reach}")
            plt.savefig(
                f"{list(data_dir.parents)[1]}/morphology_transform.png", bbox_inches="tight"
            )
            plt.close()


def _get_mean_radius(m, func, neurite_type):
    """Compute mean radius and additional section feature from func."""
    df = pd.DataFrame(columns=["res", "mean_radius"], dtype=float)
    for i, section in enumerate(
        nm.iter_sections(m, neurite_filter=tree_type_checker(neurite_type))
    ):
        res = func(section)
        mean_radius = np.mean(section.points[:, nm.COLS.R])
        df.loc[i, "res"] = res
        df.loc[i, "mean_radius"] = mean_radius
    return df


def _get_population_radii(df, func, neurite_type):
    """Get mean radius from df with morphologies."""
    pop = nm.load_morphologies(df.morph_path.to_list())
    dfs = [_get_mean_radius(m, func, neurite_type) for m in pop]
    dfs = [df for df in dfs if len(df.index) > 0]
    if dfs:
        return pd.concat(dfs)
    return None


def _fit_exp(ratios, x):
    """Fit exponential to data and evaluate on x."""

    def func_exp(x, a, c, d):
        return a * np.exp(-c * x) + d

    def func_lin(x, a, b):
        return a * x + b

    try:
        try:
            popt = curve_fit(func_exp, ratios.index, ratios.mean_radius)[0]
            return func_exp(x, *popt)
        except RuntimeError:
            popt = curve_fit(func_lin, ratios.index, ratios.mean_radius)[0]
            return func_lin(x, *popt)
    except RuntimeError:
        return func_lin(x, 0, 0)


def learn_diameter_transform(
    df,
    data_dir,
    method="branch_order",
    target_dataset="target_dataset.csv",
    plot=True,
    max_x=30,
):
    """Fit diameter scaling with branch order.

    adapted from: /gpfs/bbp.cscs.ch/project/proj66/morphologies/fitting
    """
    target_df = pd.read_csv(target_dataset)

    if method == "branch_order":
        func = branch_order
    else:
        raise NotImplementedError
    fits = pd.DataFrame(index=np.arange(0, max_x))
    for neurite_type in [
        nm.NeuriteType.axon,
        nm.NeuriteType.basal_dendrite,
        nm.NeuriteType.apical_dendrite,
    ]:
        source_radii = _get_population_radii(df, func, neurite_type)
        target_radii = _get_population_radii(target_df, func, neurite_type)
        if source_radii is None or target_radii is None:
            continue

        ratio = (target_radii.groupby("res").mean() / source_radii.groupby("res").mean()).dropna()
        fits[neurite_type.name] = _fit_exp(ratio, fits.index)

        if plot:
            plt.figure(figsize=(12, 5))
            ax = plt.gca()
            source_radii["type"] = "source"
            target_radii["type"] = "target"
            data = pd.concat([source_radii, target_radii])
            sns.violinplot(
                ax=ax,
                data=data,
                x="res",
                y="mean_radius",
                hue="type",
                split=True,
                bw=0.1,
                inner="quartile",
            )
            source_radii.drop(columns="type").groupby("res").mean().plot(ax=ax)
            target_radii.drop(columns="type").groupby("res").mean().plot(ax=ax)
            plt.savefig(
                f"{list(data_dir.parents)[1]}/diameter_comp_{neurite_type.name}.png",
                bbox_inches="tight",
            )

            plt.figure()
            ax = plt.gca()
            ratio.plot(ax=ax, marker="+")
            fits[neurite_type.name].plot(ax=ax, marker="+")
            plt.axhline(1, ls="--", c="k")
            plt.xlabel(method)
            plt.ylabel("ratio or radii (target / source)")
            plt.savefig(
                f"{list(data_dir.parents)[1]}/diameter_transform_{neurite_type.name}.png",
                bbox_inches="tight",
            )

    df_path = data_dir / "fit.csv"
    fits.to_csv(df_path)
    df["diameter_transform"] = str(df_path)


def apply_transformation(row, data_dir):
    """Apply the full transformation on a morphology.

    adapted from: /gpfs/bbp.cscs.ch/project/proj66/morphologies/scale_morphs.py

    TODO: pass the various method options to be in sync with Learn Tasks.
    """
    morph = nm.load_morphology(row["morph_path"])
    morph.soma.points *= row["soma_transform"]
    diameter_scalings = pd.read_csv(row["diameter_transform"])
    for section in nm.iter_sections(morph):
        points = section.points
        points[:, nm.COLS.XYZ] *= row["morphology_transform"]

        if section.type.name in diameter_scalings.columns:
            bo = min(branch_order(section), diameter_scalings.index[-1])
            factor = diameter_scalings.loc[bo, section.type.name]
            points[:, nm.COLS.R] *= factor
        section.points = points

    transformed_morph_path = data_dir / Path(row["morph_path"]).name
    morph.write(transformed_morph_path)
    return ValidationResult(is_valid=True, transformed_morph_path=transformed_morph_path)


def compare_transformed(df, data_dir, shift=300, target_dataset="target_dataset.csv"):
    """Plot target, source and transformed morphos."""
    target_df = pd.read_csv(target_dataset)
    target_df = target_df.astype({"morph_path": str})
    with PdfPages(f"{list(data_dir.parents)[1]}/target_morphologies.pdf") as pdf:
        for gid in tqdm(target_df.index):
            plt.figure()
            ax = plt.gca()
            view.plot_morph(
                nm.load_morphology(target_df.loc[gid, "morph_path"]),
                ax,
                soma_outline=False,
                realistic_diameters=True,
            )
            plt.autoscale()
            plt.axis("equal")
            plt.title(Path(target_df.loc[gid, "morph_path"]).stem)
            ax.set_rasterized(True)
            pdf.savefig(bbox_inches="tight", dpi=300)
            plt.close()

    df = df.astype({"morph_path": str, "transformed_morph_path": str})
    with PdfPages(f"{list(data_dir.parents)[1]}/transformed_morphologies.pdf") as pdf:
        for gid in tqdm(df.index):
            morph = nm.load_morphology(df.loc[gid, "morph_path"])
            transformed_morph = nm.load_morphology(df.loc[gid, "transformed_morph_path"])
            plt.figure()
            ax = plt.gca()
            view.plot_morph(
                morph.transform(lambda p: p - [shift, 0, 0]),
                ax,
                soma_outline=False,
                realistic_diameters=True,
            )
            view.plot_morph(
                transformed_morph.transform(lambda p: p + [shift, 0, 0]),  # noqa: RUF005
                ax,
                soma_outline=False,
                realistic_diameters=True,
            )
            plt.autoscale()
            plt.axis("equal")
            plt.title(Path(df.loc[gid, "morph_path"]).stem)
            ax.set_rasterized(True)
            pdf.savefig(bbox_inches="tight", dpi=300)
            plt.close()
