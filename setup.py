"""Setup for the morphology-workflows package."""
from setuptools import find_packages
from setuptools import setup

# Read the contents of the README file
with open("README.md", encoding="utf-8") as f:
    README = f.read()

reqs = [
    "data-validation-framework>=0.2.0",
    "diameter-synthesis>=0.2.5",
    "luigi",
    "luigi-tools>=0.0.7",
    "matplotlib",
    "morph_tool>=2.9.0,<3.0",
    "morphio>=3.1,<4.0",
    "neurom>=3.0,<4.0",
    "neuror>=1.5.0,<2.0",
    "numpy",
    "pandas",
    "plotly-helper>=0.0.8,<1.0",
    "PyYAML",
    "rst2pdf",
    "sphinx<4",
    "tqdm",
]

doc_reqs = [
    "graphviz",
    "m2r2",
    "mistune<2",
    "sphinx-autoapi<1.6",
    "sphinx-bluebrain-theme",
    "sphinx-argparse",
]

test_reqs = [
    "diff-pdf-visually>=1.5.1",
    "dir-content-diff>=0.2",
    "pytest",
    "pytest-cov",
    "pytest-html",
    "pytest-xdist",
]

setup(
    name="morphology-workflows",
    author="Blue Brain Project, EPFL",
    description="Workflows used for morphology processing.",
    long_description=README,
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    url="https://github.com/BlueBrain/morphology-workflows",
    project_urls={
        "Tracker": "https://github.com/BlueBrain/morphology-workflows/issues",
        "Source": "https://github.com/BlueBrain/morphology-workflows",
    },
    packages=find_packages("src", exclude=["tests"]),
    package_dir={"": "src"},
    python_requires=">=3.8",
    use_scm_version=True,
    setup_requires=[
        "setuptools_scm",
    ],
    install_requires=reqs,
    extras_require={"docs": doc_reqs, "test": test_reqs},
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
    entry_points={"console_scripts": ["morphology_workflows=morphology_workflows.tasks.cli:main"]},
    include_package_data=True,
)
