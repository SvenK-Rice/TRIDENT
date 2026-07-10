"""ESA / Copernicus ocean-colour data access."""

from .occi import (
    ESAOceanColourRequest,
    ESA_PRODUCTS,
    esa_status,
    download_esa_ocean_colour,
    inspect_copernicus_dataset,
)

__all__ = [
    "ESAOceanColourRequest",
    "ESA_PRODUCTS",
    "esa_status",
    "download_esa_ocean_colour",
    "inspect_copernicus_dataset",
]
