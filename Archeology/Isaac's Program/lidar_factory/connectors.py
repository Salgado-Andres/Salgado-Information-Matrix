import ee
import numpy as np
import logging
from typing import Dict, Optional, Any
from .registry import DatasetMetadata # Assuming registry.py is in the same directory

# Configure logging
logger = logging.getLogger(__name__)

class LidarConnector:
    """Abstract base class for LIDAR data connectors."""
    def __init__(self, dataset_metadata: DatasetMetadata):
        self.dataset_metadata = dataset_metadata
        self.ee_initialized = False # Specific to GEE, but useful to track for connectors that need init

    def initialize(self):
        """Initialize the connector, e.g., authenticate with a service."""
        raise NotImplementedError

    def fetch_patch(self, lat: float, lon: float, size_m: int, target_resolution_m: float, data_type_to_fetch: str) -> Optional[np.ndarray]:
        """
        Fetch a patch of LIDAR data for a specific data type (e.g., DSM, DTM).

        Args:
            lat: Center latitude of the patch.
            lon: Center longitude of the patch.
            size_m: Desired size of the patch in meters (square).
            target_resolution_m: Desired resolution in meters per pixel.
            data_type_to_fetch: The type of data product to fetch (e.g., "DSM", "DTM").

        Returns:
            A NumPy array containing the elevation data, or None if fetching fails.
        """
        raise NotImplementedError

class GEEConnector(LidarConnector):
    """Connects to Google Earth Engine to fetch LIDAR data."""
    _ee_initialized = False  # Class-level flag to track GEE initialization

    def __init__(self, dataset_metadata: DatasetMetadata):
        super().__init__(dataset_metadata)
        self._initialize_ee()

    def _initialize_ee(self):
        """Initializes Earth Engine if not already initialized."""
        if not GEEConnector._ee_initialized:
            try:
                # First check if Earth Engine is already initialized by the shared utility
                try:
                    from backend.utils.earth_engine import is_earth_engine_available
                    if is_earth_engine_available():
                        logger.info("✅ Earth Engine already initialized by shared utility")
                        GEEConnector._ee_initialized = True
                        self.ee_initialized = True
                        return
                except ImportError:
                    logger.debug("Shared Earth Engine utility not available, using local initialization")
                
                import os
                
                # Fallback: Try to use service account authentication locally
                service_account_key = os.getenv('GOOGLE_APPLICATION_CREDENTIALS') or os.getenv('GOOGLE_EE_SERVICE_ACCOUNT_KEY')
                project_id = os.getenv('GOOGLE_EE_PROJECT_ID', 'sage-striker-294302')
                
                if service_account_key and os.path.exists(service_account_key):
                    logger.info(f"🔐 Initializing Earth Engine with service account: {service_account_key}")
                    credentials = ee.ServiceAccountCredentials(None, key_file=service_account_key)
                    ee.Initialize(credentials, project=project_id)
                else:
                    logger.info(f"🔐 Initializing Earth Engine with default credentials for project: {project_id}")
                    ee.Initialize(project=project_id)
                    
                GEEConnector._ee_initialized = True
                logger.info(f"✅ Earth Engine initialized successfully for {self.dataset_metadata.name if self.dataset_metadata else 'GEE Connector'}")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Earth Engine: {e}")
                # Optionally, re-raise or handle more gracefully
                raise
        
        # Set the instance flag to match the class flag
        self.ee_initialized = GEEConnector._ee_initialized

    def fetch_patch(self, lat: float, lon: float, size_m: int, target_resolution_m: float, data_type_to_fetch: str) -> Optional[np.ndarray]:
        """
        Fetch a specific LIDAR data product (e.g., DSM, DTM) from Google Earth Engine.
        """
        if not self.ee_initialized:
            logger.error(f"Earth Engine not initialized for {self.dataset_metadata.name}. Cannot fetch data.")
            return None

        band_name = self.dataset_metadata.get_band_for_datatype(data_type_to_fetch)
        if not band_name:
            logger.error(f"Data type '{data_type_to_fetch}' not configured or band not found for dataset '{self.dataset_metadata.name}'. Available: {self.dataset_metadata.available_data_types}")
            return None

        try:
            logger.info(f"Fetching GEE data for {self.dataset_metadata.name} ({data_type_to_fetch} using band '{band_name}') at ({lat:.4f}, {lon:.4f}), size: {size_m}m, target_res: {target_resolution_m}m/px")

            center = ee.Geometry.Point([lon, lat])
            
            # Create a proper square bounds in meters using buffer
            # This should create a more square result than using degrees
            square_bounds = center.buffer(size_m / 2.0).bounds()

            # Get the GEE image or image collection ID from provider_info
            image_asset_id: Optional[str] = None
            is_collection = False
            if "image_collection_id" in self.dataset_metadata.provider_info:
                image_asset_id = self.dataset_metadata.provider_info["image_collection_id"]
                is_collection = True
            elif "image_id" in self.dataset_metadata.provider_info:
                image_asset_id = self.dataset_metadata.provider_info["image_id"]
            else:
                logger.error(f"No image_id or image_collection_id in provider_info for {self.dataset_metadata.name}")
                return None
            
            if is_collection:
                collection = ee.ImageCollection(image_asset_id).select(band_name)
                image = collection.median() # Or .mosaic(), depending on the dataset needs
            else:
                image = ee.Image(image_asset_id).select(band_name)
            
            # Use a projected coordinate system for more consistent pixel spacing
            # UTM zone is better for square patches than lat/lon
            image_reprojected = image.reproject(crs='EPSG:3857', scale=target_resolution_m)  # Web Mercator for better square sampling

            # Calculate expected dimensions for validation
            expected_pixels_dim = int(size_m / target_resolution_m)
            expected_total_pixels = expected_pixels_dim * expected_pixels_dim
            
            # Check if we'll exceed Earth Engine's pixel limit (262,144)
            if expected_total_pixels > 262144:
                logger.warning(f"Requested resolution too high for {self.dataset_metadata.name}: {expected_total_pixels} pixels > 262,144 limit. Reducing resolution.")
                # Automatically reduce resolution to stay under limit
                max_dim = int(np.sqrt(262144))  # ~512 pixels per side
                new_resolution = size_m / max_dim
                logger.info(f"Auto-adjusting resolution from {target_resolution_m}m to {new_resolution:.1f}m for {self.dataset_metadata.name}")
                image_reprojected = image.reproject(crs='EPSG:3857', scale=new_resolution)
            
            try:
                rect_data = image_reprojected.sampleRectangle(
                    region=square_bounds,
                    defaultValue=-9999
                )
            except Exception as e:
                if "Too many pixels" in str(e):
                    logger.warning(f"Still too many pixels for {self.dataset_metadata.name}. Skipping this dataset.")
                    return None
                else:
                    logger.error(f"GEE Error fetching {self.dataset_metadata.name} (band: {band_name}) data: {e}")
                    return None
            
            elev_block = rect_data.get(band_name).getInfo()
            
            if elev_block is None:
                logger.error(f"No data returned from sampleRectangle for band '{band_name}' in {self.dataset_metadata.name}")
                return None

            elevation_array = np.array(elev_block, dtype=np.float32)
            elevation_array = np.where(elevation_array == -9999, np.nan, elevation_array)

            if elevation_array.size == 0 or np.isnan(elevation_array).all():
                logger.warning(f"No valid data for {self.dataset_metadata.name} (band: {band_name}) at location {lat:.4f}, {lon:.4f}. Array is empty or all NaN.")
                return None

            if np.isnan(elevation_array).any():
                mean_val = np.nanmean(elevation_array)
                if not np.isnan(mean_val):
                    elevation_array = np.where(np.isnan(elevation_array), mean_val, elevation_array)
                else:
                    logger.warning(f"All values were NaN for {self.dataset_metadata.name} (band: {band_name}) at {lat:.4f}, {lon:.4f}. Filled with 0.")
                    elevation_array = np.nan_to_num(elevation_array, nan=0.0)

            # Log shape comparison with expected
            logger.info(f"✅ Fetched {self.dataset_metadata.name} (band: {band_name}) data: shape {elevation_array.shape}, expected ~({expected_pixels_dim}x{expected_pixels_dim})")
            
            # Optional: Resize to exact square if very close but not perfect
            if abs(elevation_array.shape[0] - expected_pixels_dim) <= 5 and abs(elevation_array.shape[1] - expected_pixels_dim) <= 5:
                from scipy.ndimage import zoom
                if 'scipy' in globals():  # Only if scipy is available
                    target_shape = (expected_pixels_dim, expected_pixels_dim)
                    zoom_factors = (target_shape[0] / elevation_array.shape[0], target_shape[1] / elevation_array.shape[1])
                    elevation_array = zoom(elevation_array, zoom_factors, order=1)
                    logger.info(f"Resized patch to exact square: {elevation_array.shape}")
            
            return elevation_array

        except ee.EEException as e:
            logger.error(f"GEE Error fetching {self.dataset_metadata.name} (band: {band_name}) data: {e}")
            if hasattr(e, 'args') and len(e.args) > 0 and isinstance(e.args[0], str):
                if "Too many concurrent aggregations" in e.args[0]:
                    logger.error("GEE concurrent aggregation limit hit. Consider retrying or reducing request frequency.")
                elif "computation timed out" in e.args[0].lower():
                    logger.error("GEE computation timed out. Consider smaller area or coarser resolution.")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching {self.dataset_metadata.name} (band: {band_name}) data: {e}")
            return None

# Example of how you might add other connectors:
# class OpenTopoConnector(LidarConnector):
#     def initialize(self):
#         # Check API keys, etc.
#         logger.info(f"OpenTopoConnector initialized for {self.dataset_metadata.name}")
#         self.ee_initialized = True # Using this flag generically for 'initialized'
#         return True

#     def fetch_patch(self, lat: float, lon: float, size_m: int, target_resolution_m: float) -> Optional[np.ndarray]:
#         if not self.ee_initialized: # self.initialized would be better name
#             logger.error("OpenTopoConnector not initialized.")
#             return None
#         logger.info(f"Fetching OpenTopo data for {self.dataset_metadata.name} (simulated)")
#         # Actual implementation would use requests library to hit OpenTopography API
#         # For now, returning a dummy array
        
#         # Calculate dimensions based on size_m and target_resolution_m
#         num_pixels_dim = int(size_m / target_resolution_m)
#         if num_pixels_dim <= 0:
#             logger.error(f"Calculated pixel dimension is zero or negative: {num_pixels_dim}. Check size_m and target_resolution_m.")
#             return None
            
#         # Simulate fetching data, e.g. a plane tilted or with some noise
#         dummy_array = np.ones((num_pixels_dim, num_pixels_dim), dtype=np.float32) * 10 # Base elevation 10m
#         # Add some variation
#         x = np.linspace(-1, 1, num_pixels_dim)
#         y = np.linspace(-1, 1, num_pixels_dim)
#         xx, yy = np.meshgrid(x,y)
#         dummy_array += xx * 2 + yy * 1 # Add a slope
#         dummy_array += np.random.rand(num_pixels_dim, num_pixels_dim) * 0.5 # Add some noise
        
#         logger.info(f"✅ Simulated OpenTopo data: shape {dummy_array.shape}")
#         return dummy_array

CONNECTOR_MAP = {
    "GEE": GEEConnector,
    # "OpenTopo": OpenTopoConnector, # Uncomment when implemented
}
