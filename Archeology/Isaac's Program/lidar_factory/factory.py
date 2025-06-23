import numpy as np
import logging
from typing import Dict, Optional, List, Any
from dataclasses import dataclass

from .registry import METADATA_REGISTRY, DatasetMetadata, get_dataset_by_name
from .connectors import CONNECTOR_MAP, LidarConnector
from .cloud_cache import LidarTileCache

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_DATA_TYPE = "DSM" # Default data type to fetch if not specified

@dataclass
class LidarPatchResult:
    """Result containing both elevation data and metadata from LiDAR fetch operation."""
    data: np.ndarray
    source_dataset: str
    resolution_m: float
    resolution_description: str  # e.g., "0.5m", "30m"
    is_high_resolution: bool
    data_type: str  # e.g., "DSM", "DTM"
    
    @property
    def shape(self) -> tuple:
        return self.data.shape

def _format_resolution(resolution_m: float) -> str:
    """Format resolution as a human-readable string."""
    if resolution_m < 1.0:
        if resolution_m == 0.5:
            return "0.5m"
        else:
            return f"{resolution_m:.1f}m"
    else:
        return f"{int(resolution_m)}m"

def _is_high_resolution(resolution_m: float, dataset_name: str) -> bool:
    """Determine if a dataset is considered high-resolution."""
    # High-resolution threshold: anything 2m or better
    # Also explicitly recognize known high-res datasets
    if resolution_m <= 2.0:
        return True
    
    # Known high-resolution datasets (even if resolution might be slightly > 2m)
    high_res_datasets = ['AHN4', 'AHN3', 'AHN2']
    if dataset_name in high_res_datasets:
        return True
    
    return False

# Global cache instance
_cache_instance: Optional[LidarTileCache] = None

def get_cache() -> LidarTileCache:
    """Get or create global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = LidarTileCache()
    return _cache_instance

class LidarMapFactory:
    """Factory to select and retrieve LIDAR data patches from various sources with cloud caching."""

    @staticmethod
    def _in_bounds(ds_meta: DatasetMetadata, lat: float, lon: float) -> bool:
        """Check if the given lat, lon is within the dataset's bounds."""
        lat_min, lon_min, lat_max, lon_max = ds_meta.bounds
        return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max

    @staticmethod
    def get_patch(lat: float, 
                  lon: float, 
                  size_m: int = 128, 
                  preferred_resolution_m: Optional[float] = None,
                  exact_dataset_name: Optional[str] = None,
                  preferred_data_type: Optional[str] = None,
                  use_cache: bool = True) -> Optional[LidarPatchResult]:
        """
        Fetches a LIDAR data patch for the given location and parameters with cloud caching.

        It selects the most suitable dataset based on availability, location,
        resolution, and requested data type (e.g., DSM, DTM). Results are automatically
        cached in Google Cloud Storage for faster subsequent access.

        Args:
            lat: Center latitude of the patch.
            lon: Center longitude of the patch.
            size_m: Desired edge size of the square patch in meters.
            preferred_resolution_m: If specified, the factory will try to find a dataset
                                    closest to this resolution. Otherwise, it typically
                                    prefers the highest available resolution.
            exact_dataset_name: If specified, only this dataset will be considered.
            preferred_data_type: The desired data product type (e.g., "DSM", "DTM").
                                 Defaults to DEFAULT_DATA_TYPE if None.
            use_cache: Whether to use cache for this request.
            cache_strategy: Cache strategy to use ("local", "gcs", "hybrid").

        Returns:
            A LidarPatchResult containing the elevation data and metadata (resolution, 
            source dataset, etc.), or None if no suitable data could be fetched.
        """
        
        data_type_to_fetch = (preferred_data_type or DEFAULT_DATA_TYPE).upper()
        logger.info(f"Requesting patch at ({lat:.4f}, {lon:.4f}), size: {size_m}m, type: {data_type_to_fetch}, pref_res: {preferred_resolution_m}m/px")

        # Initialize cloud cache
        cache = get_cache() if use_cache else None
        
        candidate_datasets: List[DatasetMetadata] = []

        if exact_dataset_name:
            dataset = get_dataset_by_name(exact_dataset_name)
            if dataset and LidarMapFactory._in_bounds(dataset, lat, lon):
                if data_type_to_fetch in dataset.available_data_types:
                    candidate_datasets.append(dataset)
                else:
                    logger.warning(f"Dataset '{exact_dataset_name}' found but does not offer data type '{data_type_to_fetch}'. Available: {dataset.available_data_types}")
            elif dataset:
                logger.warning(f"Dataset '{exact_dataset_name}' found but location ({lat:.4f}, {lon:.4f}) is out of its bounds.")
            else:
                logger.warning(f"Exact dataset '{exact_dataset_name}' not found in registry.")
        else:
            # Filter datasets that cover the requested coordinates and offer the data type
            for ds_meta in METADATA_REGISTRY:
                if LidarMapFactory._in_bounds(ds_meta, lat, lon) and data_type_to_fetch in ds_meta.available_data_types:
                    candidate_datasets.append(ds_meta)

        if not candidate_datasets:
            logger.error(f"No LIDAR datasets found in registry covering location ({lat:.4f}, {lon:.4f}) and offering data type '{data_type_to_fetch}'.")
            return None

        # Sort candidates
        if preferred_resolution_m:
            candidate_datasets.sort(key=lambda ds: (abs(ds.resolution_m - preferred_resolution_m), ds.resolution_m))
            logger.info(f"Found {len(candidate_datasets)} candidates for '{data_type_to_fetch}', sorted by preference for {preferred_resolution_m}m resolution.")
        else:
            candidate_datasets.sort(key=lambda ds: ds.resolution_m)
            logger.info(f"Found {len(candidate_datasets)} candidates for '{data_type_to_fetch}', sorted by finest resolution first.")

        # Attempt to fetch data using the sorted candidates
        for ds_meta in candidate_datasets:
            target_res_for_fetch = preferred_resolution_m if preferred_resolution_m is not None else ds_meta.resolution_m
            
            # Check cache first
            if cache:
                cached_data = cache.get(
                    lat=lat,
                    lon=lon, 
                    size_m=size_m,
                    resolution_m=target_res_for_fetch,
                    data_type=data_type_to_fetch,
                    source=ds_meta.name
                )
                if cached_data is not None:
                    # Cache returns raw numpy array, wrap it in LidarPatchResult
                    logger.info(f"✅ Using cached data from {ds_meta.name}. Shape: {cached_data.shape}")
                    result = LidarPatchResult(
                        data=cached_data,
                        source_dataset=ds_meta.name,
                        resolution_m=ds_meta.resolution_m,
                        resolution_description=_format_resolution(ds_meta.resolution_m),
                        is_high_resolution=_is_high_resolution(ds_meta.resolution_m, ds_meta.name),
                        data_type=data_type_to_fetch
                    )
                    return result
            
            logger.info(f"Attempting to use dataset: {ds_meta.name} (Resolution: {ds_meta.resolution_m}m) for data type '{data_type_to_fetch}'")
            ConnectorClass = CONNECTOR_MAP.get(ds_meta.access_method)

            if not ConnectorClass:
                logger.warning(f"No connector found for access method '{ds_meta.access_method}' of dataset '{ds_meta.name}'. Skipping.")
                continue

            try:
                connector_instance: LidarConnector = ConnectorClass(ds_meta)
                # GEEConnector initializes itself, check if successful
                if ds_meta.access_method == "GEE" and not connector_instance.ee_initialized:
                    logger.warning(f"GEE connector for {ds_meta.name} not initialized. Skipping.")
                    continue

                # Fetch from source
                patch_data = connector_instance.fetch_patch(lat, lon, size_m, target_res_for_fetch, data_type_to_fetch)

                if patch_data is not None and patch_data.size > 0:
                    logger.info(f"✅ Successfully fetched '{data_type_to_fetch}' patch from {ds_meta.name}. Shape: {patch_data.shape}")
                    
                    # Store in cache for future use
                    if cache:
                        cache.put(
                            lat=lat,
                            lon=lon,
                            size_m=size_m,
                            resolution_m=target_res_for_fetch,
                            data_type=data_type_to_fetch,
                            source=ds_meta.name,
                            tile_data=patch_data
                        )
                    
                    # Create result with metadata
                    result = LidarPatchResult(
                        data=patch_data,
                        source_dataset=ds_meta.name,
                        resolution_m=ds_meta.resolution_m,
                        resolution_description=_format_resolution(ds_meta.resolution_m),
                        is_high_resolution=_is_high_resolution(ds_meta.resolution_m, ds_meta.name),
                        data_type=data_type_to_fetch
                    )
                    
                    return result
                else:
                    logger.warning(f"Failed to fetch '{data_type_to_fetch}' patch from {ds_meta.name} or data was empty. Trying next candidate.")
            except Exception as e:
                logger.error(f"Error using connector for {ds_meta.name} to fetch '{data_type_to_fetch}': {e}. Trying next candidate.")
                continue

        logger.error(f"Exhausted all {len(candidate_datasets)} candidate datasets. Could not fetch '{data_type_to_fetch}' LIDAR data for location ({lat:.4f}, {lon:.4f}).")
        return None

    @staticmethod
    def get_patch_data_only(lat: float, 
                           lon: float, 
                           size_m: int = 128, 
                           preferred_resolution_m: Optional[float] = None,
                           exact_dataset_name: Optional[str] = None,
                           preferred_data_type: Optional[str] = None,
                           use_cache: bool = True) -> Optional[np.ndarray]:
        """
        Backward-compatible method that returns only the numpy array data.
        
        This method is for existing code that expects just the elevation data array.
        For new code, use get_patch() which returns LidarPatchResult with metadata.
        """
        result = LidarMapFactory.get_patch(lat, lon, size_m, preferred_resolution_m, 
                                         exact_dataset_name, preferred_data_type, use_cache)
        return result.data if result else None

    @staticmethod
    def get_cache_stats() -> Dict[str, Any]:
        """Get cache statistics and status."""
        cache = get_cache()
        return cache.get_cache_stats()
    
    @staticmethod
    def clear_cache(source_filter: Optional[str] = None) -> int:
        """
        Clear cache tiles, optionally filtered by source.
        
        Args:
            source_filter: If provided, only clear tiles from this source
            
        Returns:
            Number of tiles cleared
        """
        cache = get_cache()
        return cache.clear_cache(source_filter)

if __name__ == '__main__':
    print("--- LIDAR Map Factory Test (v4 with Multi-Strategy Cache) --- ")
    
    # Test different cache strategies
    strategies = ["local", "hybrid"]  # Skip GCS for now due to permissions
    
    for strategy in strategies:
        print(f"\n🗄️  Testing Cache Strategy: {strategy}")
        cache_stats = LidarMapFactory.get_cache_stats(strategy)
        print(f"Cache Status: {cache_stats}")

    # Test Case 1: AHN4 with local cache (fallback)
    zaanse_schans_lat, zaanse_schans_lon = 52.4746, 4.8163
    patch_size_m = 64

    print(f"\nAttempting to fetch AHN4 DSM data for Zaanse Schans ({zaanse_schans_lat}, {zaanse_schans_lon}), size={patch_size_m}m")
    
    # First request with local cache (should hit source)
    import time
    start_time = time.time()
    ahn4_patch_dsm_default = LidarMapFactory.get_patch(
        zaanse_schans_lat, zaanse_schans_lon, 
        size_m=patch_size_m, 
        cache_strategy="local"
    )
    first_fetch_time = time.time() - start_time
    
    if ahn4_patch_dsm_default is not None:
        print(f"AHN4 (First fetch - local cache) - Patch shape: {ahn4_patch_dsm_default.shape}, Min: {np.nanmin(ahn4_patch_dsm_default):.2f}, Max: {np.nanmax(ahn4_patch_dsm_default):.2f}, Mean: {np.nanmean(ahn4_patch_dsm_default):.2f}, Time: {first_fetch_time:.2f}s")
    else:
        print("AHN4 (First fetch) - Failed to get patch.")
    
    # Second request (should hit local cache)
    print(f"\nSecond request for same location (should hit local cache)...")
    start_time = time.time()
    ahn4_patch_dsm_cached = LidarMapFactory.get_patch(
        zaanse_schans_lat, zaanse_schans_lon, 
        size_m=patch_size_m,
        cache_strategy="local"
    )
    second_fetch_time = time.time() - start_time
    
    if ahn4_patch_dsm_cached is not None:
        print(f"AHN4 (Cached) - Patch shape: {ahn4_patch_dsm_cached.shape}, Min: {np.nanmin(ahn4_patch_dsm_cached):.2f}, Max: {np.nanmax(ahn4_patch_dsm_cached):.2f}, Mean: {np.nanmean(ahn4_patch_dsm_cached):.2f}, Time: {second_fetch_time:.2f}s")
        if first_fetch_time > 0 and second_fetch_time > 0:
            speedup = first_fetch_time / second_fetch_time
            print(f"📈 Local cache speedup: {speedup:.1f}x faster")
    else:
        print("AHN4 (Cached) - Failed to get patch.")

    # Test hybrid cache strategy
    print(f"\nTesting hybrid cache strategy...")
    start_time = time.time()
    ahn4_patch_hybrid = LidarMapFactory.get_patch(
        zaanse_schans_lat, zaanse_schans_lon, 
        size_m=patch_size_m,
        cache_strategy="hybrid"
    )
    hybrid_fetch_time = time.time() - start_time
    
    if ahn4_patch_hybrid is not None:
        print(f"AHN4 (Hybrid cache) - Time: {hybrid_fetch_time:.2f}s")

    # Request DSM explicitly
    ahn4_patch_dsm_explicit = LidarMapFactory.get_patch(zaanse_schans_lat, zaanse_schans_lon, size_m=patch_size_m, preferred_data_type="DSM")
    if ahn4_patch_dsm_explicit is not None:
        print(f"AHN4 (Explicit DSM) - Patch shape: {ahn4_patch_dsm_explicit.shape}, Min: {np.nanmin(ahn4_patch_dsm_explicit):.2f}, Max: {np.nanmax(ahn4_patch_dsm_explicit):.2f}, Mean: {np.nanmean(ahn4_patch_dsm_explicit):.2f}")
    else:
        print("AHN4 (Explicit DSM) - Failed to get patch.")

    # Test requesting a DTM (assuming AHN4 in registry only has DSM for now, this should fail gracefully or pick SRTM if it had DTM)
    print(f"\nAttempting to fetch DTM data for Zaanse Schans (expect failure or different dataset if DTM was available elsewhere)")
    ahn4_patch_dtm_request = LidarMapFactory.get_patch(zaanse_schans_lat, zaanse_schans_lon, size_m=patch_size_m, preferred_data_type="DTM")
    if ahn4_patch_dtm_request is not None:
        print(f"Zaanse Schans (DTM Request) - Patch shape: {ahn4_patch_dtm_request.shape}, Min: {np.nanmin(ahn4_patch_dtm_request):.2f}, Max: {np.nanmax(ahn4_patch_dtm_request):.2f}, Mean: {np.nanmean(ahn4_patch_dtm_request):.2f}")
    else:
        print("Zaanse Schans (DTM Request) - Failed to get patch as expected (or no DTM available).")

    # Test Case 2: SRTM (Global, lower resolution)
    print(f"\nAttempting to fetch SRTM DSM data for Zaanse Schans ({zaanse_schans_lat}, {zaanse_schans_lon}), size={patch_size_m}m")
    srtm_patch_specific_dsm = LidarMapFactory.get_patch(zaanse_schans_lat, zaanse_schans_lon, size_m=patch_size_m, exact_dataset_name="SRTMGL1_003", preferred_data_type="DSM")
    if srtm_patch_specific_dsm is not None:
        print(f"SRTM (Specific DSM) - Patch shape: {srtm_patch_specific_dsm.shape}, Min: {np.nanmin(srtm_patch_specific_dsm):.2f}, Max: {np.nanmax(srtm_patch_specific_dsm):.2f}, Mean: {np.nanmean(srtm_patch_specific_dsm):.2f}")
    else:
        print("SRTM (Specific DSM) - Failed to get patch.")

    # Test Case 3: Location outside AHN4 coverage (e.g., somewhere in Germany)
    german_lat, german_lon = 51.1657, 10.4515
    print(f"\nAttempting to fetch DSM data for a location in Germany ({german_lat}, {german_lon}), size={patch_size_m}m")
    german_patch_dsm_default = LidarMapFactory.get_patch(german_lat, german_lon, size_m=patch_size_m, preferred_data_type="DSM") # Explicitly DSM
    if german_patch_dsm_default is not None:
        print(f"Germany (Default DSM) - Patch shape: {german_patch_dsm_default.shape}, Min: {np.nanmin(german_patch_dsm_default):.2f}, Max: {np.nanmax(german_patch_dsm_default):.2f}, Mean: {np.nanmean(german_patch_dsm_default):.2f}")
    else:
        print("Germany (Default DSM) - Failed to get patch.")
    
    # Final cache stats for all strategies
    print(f"\n🗄️  Final Cache Statistics:")
    for strategy in strategies:
        final_stats = LidarMapFactory.get_cache_stats(strategy)
        print(f"{strategy.upper()}: {final_stats}")

    print("\n--- Test Complete ---")
