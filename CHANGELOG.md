# Changelog

## [0.9.4](https://github.com/BlueBrain/morphology-workflows/compare/0.9.3..0.9.4)

> 12 March 2024

### New Features

- Add sign flip in Orient (Alexis Arnaudon - [#144](https://github.com/BlueBrain/morphology-workflows/pull/144))

### Fixes

- The add_soma() function should work with no root point (Adrien Berchet - [#145](https://github.com/BlueBrain/morphology-workflows/pull/145))

## [0.9.3](https://github.com/BlueBrain/morphology-workflows/compare/0.9.2..0.9.3)

> 15 February 2024

### Fixes

- Use absolute paths in the dataset created by the Initialize workflow (Adrien Berchet - [#142](https://github.com/BlueBrain/morphology-workflows/pull/142))

## [0.9.2](https://github.com/BlueBrain/morphology-workflows/compare/0.9.1..0.9.2)

> 12 February 2024

### New Features

- Make the FixZeroDiameters, Unravel and RepairNeurites steps skippable (Adrien Berchet - [#135](https://github.com/BlueBrain/morphology-workflows/pull/135))

### Documentation Changes

- Make more clear that the dependency graphs are clickable (Adrien Berchet - [#139](https://github.com/BlueBrain/morphology-workflows/pull/139))
- Fix some more links (Adrien Berchet - [#138](https://github.com/BlueBrain/morphology-workflows/pull/138))
- Fix link to neuror.main.Repair (Adrien Berchet - [#137](https://github.com/BlueBrain/morphology-workflows/pull/137))

### CI Improvements

- Add CodeCov token (Adrien Berchet - [#136](https://github.com/BlueBrain/morphology-workflows/pull/136))

## [0.9.1](https://github.com/BlueBrain/morphology-workflows/compare/0.9.0..0.9.1)

> 22 December 2023

### New Features

- Add nb_processes to MakeRelease and improve log level when using the CLI (Adrien Berchet - [#131](https://github.com/BlueBrain/morphology-workflows/pull/131))

## [0.9.0](https://github.com/BlueBrain/morphology-workflows/compare/0.8.1..0.9.0)

> 21 December 2023

### Build

- Bump MorphIO and fix tests accordingly (Adrien Berchet - [#125](https://github.com/BlueBrain/morphology-workflows/pull/125))

### New Features

- Add logger entry to know which morphology could not be written in make_release (Adrien Berchet - [#129](https://github.com/BlueBrain/morphology-workflows/pull/129))

### Fixes

- Handle any number of points in _add_soma and add a specific test (Adrien Berchet - [#123](https://github.com/BlueBrain/morphology-workflows/pull/123))
- Create contour soma properly (Adrien Berchet - [#122](https://github.com/BlueBrain/morphology-workflows/pull/122))

## [0.8.1](https://github.com/BlueBrain/morphology-workflows/compare/0.8.0..0.8.1)

> 15 November 2023

### Fixes

- Create the dataset before the metadata to not raise a useless warning (Adrien Berchet - [#120](https://github.com/BlueBrain/morphology-workflows/pull/120))

## [0.8.0](https://github.com/BlueBrain/morphology-workflows/compare/0.7.0..0.8.0)

> 1 November 2023

### New Features

- Smooth diameters for all population (Alexis Arnaudon - [#117](https://github.com/BlueBrain/morphology-workflows/pull/117))
- Improve Placeholders task (Adrien Berchet - [#110](https://github.com/BlueBrain/morphology-workflows/pull/110))

### Fixes

- Some small doc/plot fixes (Alexis Arnaudon - [#116](https://github.com/BlueBrain/morphology-workflows/pull/116))
- Properly initialize logger (Adrien Berchet - [#108](https://github.com/BlueBrain/morphology-workflows/pull/108))
- Write morphologies with NRN order (Alexis Arnaudon - [#104](https://github.com/BlueBrain/morphology-workflows/pull/104))
- Fix axon stub for has_axon (Alexis Arnaudon - [#103](https://github.com/BlueBrain/morphology-workflows/pull/103))
- fix repair plotting (Alexis Arnaudon - [#99](https://github.com/BlueBrain/morphology-workflows/pull/99))

### Chores And Housekeeping

- Apply Copier template (Adrien Berchet - [#109](https://github.com/BlueBrain/morphology-workflows/pull/109))

### Changes to Test Assests

- Use monkeypatch fixture instead of os.chdir() (Adrien Berchet - [#114](https://github.com/BlueBrain/morphology-workflows/pull/114))
- Fix soma type in test_no_soma (Adrien Berchet - [#105](https://github.com/BlueBrain/morphology-workflows/pull/105))

### Tidying of Code eg Whitespace

- Setup Ruff config (Adrien Berchet - [#115](https://github.com/BlueBrain/morphology-workflows/pull/115))

## [0.7.0](https://github.com/BlueBrain/morphology-workflows/compare/0.6.1..0.7.0)

> 22 May 2023

### New Features

- Change entry point from morphology_workflows to morphology-workflows (Adrien Berchet - [#98](https://github.com/BlueBrain/morphology-workflows/pull/98))

### Fixes

- Compatibility with Pandas&gt;=2 and urllib3&gt;=2 (Adrien Berchet - [#96](https://github.com/BlueBrain/morphology-workflows/pull/96))

### Documentation Changes

- Add link to the schema of the Repair params (Adrien Berchet - [#91](https://github.com/BlueBrain/morphology-workflows/pull/91))

### General Changes

- Fix layers not int (Alexis Arnaudon - [#92](https://github.com/BlueBrain/morphology-workflows/pull/92))
- Use JSON schema from NeuroR.main.Repair and fix CLI command in docs (Adrien Berchet - [#90](https://github.com/BlueBrain/morphology-workflows/pull/90))

## [0.6.1](https://github.com/BlueBrain/morphology-workflows/compare/0.6.0..0.6.1)

> 27 March 2023

### New Features

- Update command and doc (Adrien Berchet - [#88](https://github.com/BlueBrain/morphology-workflows/pull/88))

## [0.6.0](https://github.com/BlueBrain/morphology-workflows/compare/0.5.3..0.6.0)

> 13 March 2023

### New Features

- Add helpers to create inputs (Adrien Berchet - [#85](https://github.com/BlueBrain/morphology-workflows/pull/85))

### Chores And Housekeeping

- Bump Copier template (Adrien Berchet - [#82](https://github.com/BlueBrain/morphology-workflows/pull/82))

### CI Improvements

- Do not export all test results as artifacts and fix dependency graphs in docs (Adrien Berchet - [#84](https://github.com/BlueBrain/morphology-workflows/pull/84))

### General Changes

- Add propagation between workflows, fix MakeCollage and fix empty mtypes (Adrien Berchet - [#83](https://github.com/BlueBrain/morphology-workflows/pull/83))

## [0.5.3](https://github.com/BlueBrain/morphology-workflows/compare/0.5.2..0.5.3)

> 31 January 2023

### Chores And Housekeeping

- Add JSON schemas to ListParameters (Adrien Berchet - [#79](https://github.com/BlueBrain/morphology-workflows/pull/79))

### CI Improvements

- Bump pre-commit hooks (Adrien Berchet - [#80](https://github.com/BlueBrain/morphology-workflows/pull/80))

## [0.5.2](https://github.com/BlueBrain/morphology-workflows/compare/0.5.1..0.5.2)

> 5 January 2023

### Build

- Bump Sphinx (Adrien Berchet - [#77](https://github.com/BlueBrain/morphology-workflows/pull/77))

## [0.5.1](https://github.com/BlueBrain/morphology-workflows/compare/0.5.0..0.5.1)

> 22 December 2022

### Chores And Housekeeping

- Use optional params for region and mtype and improve default logger (Adrien Berchet - [#75](https://github.com/BlueBrain/morphology-workflows/pull/75))

## [0.5.0](https://github.com/BlueBrain/morphology-workflows/compare/0.4.2..0.5.0)

> 22 December 2022

### New Features

- Placeholders can be computed without metadata.csv file (Adrien Berchet - [#72](https://github.com/BlueBrain/morphology-workflows/pull/72))

### Fixes

- Make the rerun parameter work with CLI (Adrien Berchet - [#73](https://github.com/BlueBrain/morphology-workflows/pull/73))

## [0.4.2](https://github.com/BlueBrain/morphology-workflows/compare/0.4.1..0.4.2)

> 13 December 2022

### Chores And Housekeeping

- Remove dead code in conf.py (Adrien Berchet - [#68](https://github.com/BlueBrain/morphology-workflows/pull/68))
- Bump AllenSDK and Sphinx and use proper optional parameters (Adrien Berchet - [#67](https://github.com/BlueBrain/morphology-workflows/pull/67))
- Fix language detection (Adrien Berchet - [#64](https://github.com/BlueBrain/morphology-workflows/pull/64))

### CI Improvements

- Use tox 4 (Adrien Berchet - [#70](https://github.com/BlueBrain/morphology-workflows/pull/70))
- Apply Copier template (Adrien Berchet - [#69](https://github.com/BlueBrain/morphology-workflows/pull/69))

### General Changes

- Update README.md (alex4200 - [53b70d4](https://github.com/BlueBrain/morphology-workflows/commit/53b70d41c4b271f47fef00ba24b27b48747bf68b))

## [0.4.1](https://github.com/BlueBrain/morphology-workflows/compare/0.4.0..0.4.1)

> 22 September 2022

### Refactoring and Updates

- Update from template (Adrien Berchet - [#58](https://github.com/BlueBrain/morphology-workflows/pull/58))
- Apply Copier template (Adrien Berchet - [#57](https://github.com/BlueBrain/morphology-workflows/pull/57))

### CI Improvements

- Setup pre-commit and format the files accordingly (Adrien Berchet - [#55](https://github.com/BlueBrain/morphology-workflows/pull/55))
- Fix action to publish new releases on Pypi (Adrien Berchet - [#54](https://github.com/BlueBrain/morphology-workflows/pull/54))

### General Changes

- Reconnect axons to soma (Alexis Arnaudon - [#62](https://github.com/BlueBrain/morphology-workflows/pull/62))
- Simpler smooth algorithm (Alexis Arnaudon - [#56](https://github.com/BlueBrain/morphology-workflows/pull/56))
- Add duplicate layers to releases (Alexis Arnaudon - [#60](https://github.com/BlueBrain/morphology-workflows/pull/60))
- Fail if spaces in morph name (Alexis Arnaudon - [#59](https://github.com/BlueBrain/morphology-workflows/pull/59))

## [0.4.0](https://github.com/BlueBrain/morphology-workflows/compare/0.3.2..0.4.0)

> 11 August 2022

### Chores And Housekeeping

- Set mtype_regex argument as optional (Adrien Berchet - [#52](https://github.com/BlueBrain/morphology-workflows/pull/52))

### Documentation Changes

- Improve doc for the Fetch workflow (Adrien Berchet - [#51](https://github.com/BlueBrain/morphology-workflows/pull/51))
- Add the Placeholders workflow in the doc (Adrien Berchet - [#49](https://github.com/BlueBrain/morphology-workflows/pull/49))
- Improve instructions to create inputs (Adrien Berchet - [#47](https://github.com/BlueBrain/morphology-workflows/pull/47))

### General Changes

- Remove warning from scipy (Adrien Berchet - [#48](https://github.com/BlueBrain/morphology-workflows/pull/48))
- Add workflow to compute placeholders (Adrien Berchet - [#22](https://github.com/BlueBrain/morphology-workflows/pull/22))

### CI Improvements

- Use commitlint to check PR titles (#45) (Adrien Berchet - [4d9c3df](https://github.com/BlueBrain/morphology-workflows/commit/4d9c3df738d0b715c5156cc3e4ce066738029bc9))

<!-- auto-changelog-above -->

## [0.3.2](https://github.com/BlueBrain/morphology-workflows/compare/0.3.1..0.3.2)

> 1 July 2022

- Improve test coverage (Adrien Berchet - [#40](https://github.com/BlueBrain/morphology-workflows/pull/40))
- Use luigi features moved from luigi-tools (Adrien Berchet - [#42](https://github.com/BlueBrain/morphology-workflows/pull/42))
- Fix task arguments in CLI and use new empty dependency graph feature of luigi-tools (Adrien Berchet - [#41](https://github.com/BlueBrain/morphology-workflows/pull/41))
- Improve MakeRelease and minor changes (Alexis Arnaudon - [#11](https://github.com/BlueBrain/morphology-workflows/pull/11))
- Constrain requirements for Jinja2 and Numpy (Adrien Berchet - [#38](https://github.com/BlueBrain/morphology-workflows/pull/38))

## [0.3.1](https://github.com/BlueBrain/morphology-workflows/compare/0.3.0..0.3.1)

> 17 June 2022

- Ensure 0-length sections are always fixed, even if the min length is too small (Adrien Berchet - [#35](https://github.com/BlueBrain/morphology-workflows/pull/35))

## [0.3.0](https://github.com/BlueBrain/morphology-workflows/compare/0.2.0..0.3.0)

> 14 June 2022

- Build and publish wheels (Adrien Berchet - [#32](https://github.com/BlueBrain/morphology-workflows/pull/32))
- Fix 0-length root sections in CheckNeurites task (arnaudon - [#31](https://github.com/BlueBrain/morphology-workflows/pull/31))
- Fix SSL certificate in example (Alexis Arnaudon - [#23](https://github.com/BlueBrain/morphology-workflows/pull/23))
- Fix optional imports (Adrien Berchet - [#27](https://github.com/BlueBrain/morphology-workflows/pull/27))
- Fetch morphologies workflow (Adrien Berchet - [#20](https://github.com/BlueBrain/morphology-workflows/pull/20))
- cleanup example (Alexis Arnaudon - [#17](https://github.com/BlueBrain/morphology-workflows/pull/17))
- Add link to the documentation into README.md (Adrien Berchet - [#16](https://github.com/BlueBrain/morphology-workflows/pull/16))
- Add banner to repo (alex4200 - [#13](https://github.com/BlueBrain/morphology-workflows/pull/13))
- Adding logo for Morphology-Workflows (alex4200 - [#12](https://github.com/BlueBrain/morphology-workflows/pull/12))
- Missed fix (#18) (Alexis Arnaudon - [3a9b56f](https://github.com/BlueBrain/morphology-workflows/commit/3a9b56f49fd4e01b2fb3abd2c7493e64912f9182))
- Ensure axon stub (#26) (arnaudon - [7e7f750](https://github.com/BlueBrain/morphology-workflows/commit/7e7f750391ef495c77219bd4e11d8abb978b1c3d))
- Updating copyright year (adietz - [9b662e5](https://github.com/BlueBrain/morphology-workflows/commit/9b662e538ffa598415a1a6a1640c03ef3354a91d))

## [0.2.0](https://github.com/BlueBrain/morphology-workflows/compare/0.1.3..0.2.0)

> 12 January 2022

- Use the new input mechanism for DVF inputs (Adrien Berchet - [#8](https://github.com/BlueBrain/morphology-workflows/pull/8))

## [0.1.3](https://github.com/BlueBrain/morphology-workflows/compare/0.1.2..0.1.3)

> 20 December 2021

- Use the SkippableMixin from DVF (Adrien Berchet - [#1](https://github.com/BlueBrain/morphology-workflows/pull/1))

## [0.1.2](https://github.com/BlueBrain/morphology-workflows/compare/0.1.1..0.1.2)

> 20 December 2021

- Fix Pypi variable (Adrien Berchet - [#3](https://github.com/BlueBrain/morphology-workflows/pull/3))

## [0.1.1](https://github.com/BlueBrain/morphology-workflows/compare/0.1.0..0.1.1)

> 20 December 2021

- Generate workflow images during sphinx build (Adrien Berchet - [#2](https://github.com/BlueBrain/morphology-workflows/pull/2))

## 0.1.0

> 16 December 2021

- Open source the Curate, Annotate and Repair workflows from the morphology-processing-workflow package (Adrien Berchet - [ccd4854](https://github.com/BlueBrain/morphology-workflows/commit/ccd4854b8c6126436f20faea4cf2c1488b30d5a8))
