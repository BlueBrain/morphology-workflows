"""Setup for the morphology-workflows package."""
from pathlib import Path

from setuptools import find_namespace_packages
from setuptools import setup

reqs = [
    "data-validation-framework>=0.3.0",
    "diameter-synthesis>=0.5.2,<0.6",
    "luigi>=3.1",
    "luigi-tools>=0.0.18",
    "matplotlib",
    "morphapi",
    "morph_tool>=2.9.0,<3.0",
    "morphio>=3.1,<4.0",
    "neurom>=3.2.0,<4.0",
    "neuror>=1.5.0,<2.0",
    "numpy>=1.21",
    "pandas",
    "plotly-helper>=0.0.8,<1.0",
    "PyYAML",
    "rst2pdf",
    "scipy>=1.6",
    "sphinx<5",
    "tqdm",
]

doc_reqs = [
    "graphviz",
    "m2r2",
    "sphinx-argparse",
    "sphinx-autoapi",
    "sphinx-bluebrain-theme",
]

test_reqs = [
    "diff-pdf-visually>=1.5.1",
    "dir-content-diff>=0.2",
    "mock",
    "pytest",
    "pytest-cov",
    "pytest-html",
    "pytest-xdist",
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
    python_requires=">=3.8",
    use_scm_version=True,
    setup_requires=[
        "setuptools_scm",
    ],
    install_requires=reqs,
    extras_require={
        "docs": doc_reqs,
        "test": test_reqs,
        "allen_brain": ["allensdk>=2.13.5"],
        "mouselight": ["bg_atlasapi"],
    },
    entry_points={
        "console_scripts": [
            "morphology_workflows=morphology_workflows.tasks.cli:main",
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
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
