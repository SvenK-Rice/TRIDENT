# TRIDENT native data sources

## NASA Earthdata

TRIDENT uses NASA `earthaccess` to search and download OB.DAAC ocean-colour collections by collection short name, date range and bounding box. Collection identifiers remain editable in the app because sensors and reprocessing streams can change.

## ESA Ocean Colour CCI

The ESA Ocean Colour Climate Change Initiative provides merged daily and monthly chlorophyll-a and remote-sensing reflectance products on a global 4 km grid. TRIDENT exposes ESA OC-CCI-derived products distributed through Copernicus Marine so users can request spatial and temporal subsets instead of downloading full global files.

## Copernicus Marine

TRIDENT uses the Copernicus Marine Toolbox for:

- ESA-CCI-derived and OLCI-containing global ocean-colour products;
- physical forcing such as sea-surface temperature and mixed-layer depth.

Dataset IDs and variable names are editable in the app and can be inspected through the installed Copernicus Marine client before downloading.

## Provenance rule

TRIDENT records source files and priorities in a manifest. It does not silently merge variables until units, scale factors, masks, coordinate orientation and temporal compositing have been checked.
