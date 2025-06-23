from typing import Tuple, List, Dict, Any, Optional

class DatasetMetadata:
    """Holds metadata for a single LIDAR dataset, potentially with multiple data products (bands)."""
    def __init__(self,
                 name: str,
                 resolution_m: float, # Native resolution for the primary product or common resolution
                 bounds: Tuple[float, float, float, float],  # (lat_min, lon_min, lat_max, lon_max)
                 access_method: str,
                 # provider_info now structured to hold band mappings
                 # For GEE: {"image_id/collection_id": "id", "products": {"DSM": "band_name1", "DTM": "band_name2"}}
                 provider_info: Dict[str, Any],
                 description: Optional[str] = None):
        self.name = name
        self.resolution_m = resolution_m
        self.bounds = bounds
        self.access_method = access_method
        self.provider_info = provider_info
        self.description = description

        # Dynamically determine available data types from provider_info
        if self.access_method == "GEE" and "products" in self.provider_info and isinstance(self.provider_info["products"], dict):
            self.available_data_types: List[str] = [dt.upper() for dt in self.provider_info["products"].keys()]
        else:
            # Fallback or for other access methods; might need adjustment
            # If no specific products are listed, assume it provides a DSM by default if it's GEE.
            # This part might need to be more robust based on other access methods.
            self.available_data_types: List[str] = ["DSM"] if self.access_method == "GEE" else []


    def get_band_for_datatype(self, data_type: str) -> Optional[str]:
        """Returns the specific band name for a given data_type (e.g., DSM, DTM) if configured."""
        if self.access_method == "GEE" and "products" in self.provider_info and isinstance(self.provider_info["products"], dict):
            return self.provider_info["products"].get(data_type.upper()) # Ensure case-insensitivity for lookup
        return None

    def __repr__(self):
        return f"DatasetMetadata(name='{self.name}', resolution={self.resolution_m}m, types={self.available_data_types})"

# Global registry of available LIDAR datasets
METADATA_REGISTRY: List[DatasetMetadata] = [
    DatasetMetadata(
        name="AHN4",
        resolution_m=0.5,
        bounds=(50.75, 3.2, 53.7, 7.22),  # Approximate bounds for Netherlands
        access_method="GEE",
        provider_info={
            "image_collection_id": "AHN/AHN4", # Key for GEE ImageCollection
            "products": {
                "DSM": "dsm" # AHN4's DSM is from the 'dsm' band
                # If AHN4 also offered a DTM, e.g., from a band named 'dtm_ahn', we would add:
                # "DTM": "dtm_ahn"
            }
        },
        description="Actueel Hoogtebestand Nederland, 0.5m Digital Surface Model."
    ),
    DatasetMetadata(
        name="SRTMGL1_003", # SRTM GL1 V003 (30m resolution)
        resolution_m=30.0,
        bounds=(-90, -180, 90, 180),  # Global coverage
        access_method="GEE",
        provider_info={
            "image_id": "USGS/SRTMGL1_003", # Key for GEE Image
            "products": {
                "DSM": "elevation" # SRTM's 'elevation' band is considered a DSM
            }
        },
        description="SRTM Global 1 arc-second, ~30m Digital Surface Model."
    ),
    # Example for a future dataset with both DSM and DTM:
    # DatasetMetadata(
    #     name="FutureLidarProduct",
    #     resolution_m=1.0, # Or perhaps a dict if resolutions vary significantly per product
    #     bounds=(10.0, 10.0, 20.0, 20.0), # Example bounds
    #     access_method="GEE",
    #     provider_info={
    #         "image_collection_id": "SOME_HYPOTHETICAL_COLLECTION/ID",
    #         "products": {
    #             "DSM": "surface_height_band",
    #             "DTM": "terrain_height_band"
    #         }
    #     },
    #     description="Hypothetical dataset with 1m DSM and DTM."
    # ),
]

def get_dataset_by_name(name: str) -> Optional[DatasetMetadata]:
    """Retrieve a dataset from the registry by its name."""
    for ds in METADATA_REGISTRY:
        if ds.name == name:
            return ds
    return None
