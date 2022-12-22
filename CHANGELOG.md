# Changelog

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

### Uncategorized Changes

- Update README.md (alex4200 - [53b70d4](https://github.com/BlueBrain/morphology-workflows/commit/53b70d41c4b271f47fef00ba24b27b48747bf68b))

## [0.4.1](https://github.com/BlueBrain/morphology-workflows/compare/0.4.0..0.4.1)

> 22 September 2022

### Refactoring and Updates

- Update from template (Adrien Berchet - [#58](https://github.com/BlueBrain/morphology-workflows/pull/58))
- Apply Copier template (Adrien Berchet - [#57](https://github.com/BlueBrain/morphology-workflows/pull/57))

### CI Improvements

- Setup pre-commit and format the files accordingly (Adrien Berchet - [#55](https://github.com/BlueBrain/morphology-workflows/pull/55))
- Fix action to publish new releases on Pypi (Adrien Berchet - [#54](https://github.com/BlueBrain/morphology-workflows/pull/54))

### Uncategorized Changes

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

### Uncategorized Changes

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
