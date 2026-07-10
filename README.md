# TRIDENT NPP Workbench 3.7.0

TRIDENT is a marine primary-production workbench with two deliberately separated workflows:

1. **OSU Validation** — recalculates CAFE using archived OSU inputs and compares the result with the direct archived OSU CAFE NPP product.
2. **TRIDENT Native** — downloads/caches NASA and ESA ocean-colour products plus Copernicus physical forcing, identifies the required variables, harmonizes them to a common grid, and runs the same validated CAFE implementation only when all inputs pass readiness checks.

## Start on macOS

```bash
cd ~/Downloads
rm -rf TRIDENT
unzip -o TRIDENT_v3_7_0.zip
cd TRIDENT
./RUN_TRIDENT.command
```

The app lets the user choose the data directory, including an external drive. It only downloads data after the user chooses a region/date range and clicks the download button.

## Required CAFE inputs

- chlorophyll-a
- PAR
- aph443
- adg443
- bbp443
- bbp spectral slope
- SST
- mixed-layer depth

TRIDENT Native does not substitute undocumented constants for missing scientific inputs. The preparation report lists missing variables and prevents model execution until the input set is complete.

## Reproducible installation

TRIDENT 3.7.2 uses two pinned environments:

- `.venv` for the application, NASA Earthdata, ESA/Copernicus access, NetCDF/HDF5, and visualization.
- `.venv-hdf` only for OSU HDF4 reading.

On the first launch, dependencies are installed from `requirements.lock` and `requirements-hdf.lock`. Later launches compare lock-file hashes and verify imports; when nothing changed, pip is skipped entirely. Installation details are written to `logs/install.log`.
