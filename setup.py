"""Setup for the morphology-workflows package."""

from pathlib import Path

from setuptools import find_namespace_packages
from setuptools import setup

reqs = [
    "data-validation-framework>=0.7.0",
    "diameter-synthesis>=0.5.2,<0.6",
    "luigi>=3.2",
    "luigi-tools>=0.3.3",
    "matplotlib>=3.4",
    "morphapi>=0.3",
    "morph_tool>=2.10.1,<3.0",
    "morphio>=3.3.6,<4.0",
    "neurom>=3.2.10,<4.0",
    "neuror>=1.6.3,<2.0",
    "numpy>=1.23",
    "pandas>=1.5",
    "plotly-helper>=0.0.8,<1.0",
    "PyYAML>=6",
    "rst2pdf>=0.99",
    "scipy>=1.8",
    "tqdm>=4.44",
    "urllib3>=1.26,<2; python_version < '3.9'",
    "bluepyparallel>=0.2.2",
    "lxml>=4.8",
    "neuroc>=0.3.0,<1.0",
    "seaborn>=0.11",
]

doc_reqs = [
    "docutils<0.21",  # Temporary fix for m2r2
    "graphviz",
    "m2r2",
    "sphinx-argparse",
    "sphinx-autoapi",
    "sphinx-bluebrain-theme",
    "sphinx-jsonschema",
]

test_reqs = [
    "dictdiffer>=0.8",
    "diff-pdf-visually>=1.7",
    "dir-content-diff>=1.9",
    "morph_tool>=2.11.1,<3.0",
    "pytest>=7",
    "pytest-console-scripts>=1.4",
    "pytest-cov>=3",
    "pytest-html>=2",
    "pytest-xdist>=2",
]

setup(
    name="morphology-workflows",
    author="Blue Brain Project, EPFL",
    description="Workflows used for morphology processing.",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://morphology-workflows.readthedocs.io",
    project_urls={
        "Tracker": "https://github.com/BlueBrain/morphology-workflows/issues",
        "Source": "https://github.com/BlueBrain/morphology-workflows",
    },
    license="Apache License 2.0",
    packages=find_namespace_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    use_scm_version=True,
    setup_requires=[
        "setuptools_scm",
    ],
    install_requires=reqs,
    extras_require={
        "docs": doc_reqs,
        "graphviz": ["graphviz"],
        "mouselight": ["brainglobe_atlasapi>=2"],
        "test": test_reqs,
    },
    entry_points={
        "console_scripts": [
            "morphology-workflows=morphology_workflows.tasks.cli:main",
        ],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
