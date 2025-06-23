"""
Multi-strategy cache for LIDAR Factory with fallback options.

Supports both individual patch caching and aggregated tile caching.
"""

import os
import json
import hashlib
import numpy as np
import logging
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime
import io
from pathlib import Path

logger = logging.getLogger(__name__)

class LidarCacheStrategy:
    """Base class for cache strategies."""
    
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._initialized = True
    
    def get(self, cache_key: str) -> Optional[np.ndarray]:
        raise NotImplementedError
    
    def put(self, cache_key: str, data: np.ndarray, metadata: Dict) -> bool:
        raise NotImplementedError
    
    def exists(self, cache_key: str) -> bool:
        raise NotImplementedError
    
    def get_stats(self) -> Dict[str, Any]:
        raise NotImplementedError

class LocalFileCache(LidarCacheStrategy):
    """Local filesystem cache strategy."""
    
    def __init__(self, cache_dir: str = "./lidar_cache"):
        super().__init__(cache_dir)
        logger.info(f"✅ Local file cache initialized: {self.cache_dir}")
    
    def _cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.npz"
    
    def get(self, cache_key: str) -> Optional[np.ndarray]:
        cache_file = self._cache_path(cache_key)
        if not cache_file.exists():
            return None
        
        try:
            data = np.load(cache_file, allow_pickle=True)
            tile_data = data['elevation']
            metadata = data['metadata'].item()
            logger.info(f"✅ Local cache hit: {cache_key} | Shape: {tile_data.shape}")
            return tile_data
        except Exception as e:
            logger.warning(f"Error loading from local cache: {e}")
            return None
    
    def put(self, cache_key: str, data: np.ndarray, metadata: Dict) -> bool:
        try:
            cache_file = self._cache_path(cache_key)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            np.savez_compressed(
                cache_file,
                elevation=data,
                metadata=metadata
            )
            
            size_kb = cache_file.stat().st_size / 1024
            logger.info(f"✅ Cached locally: {cache_key} | Size: {size_kb:.1f}KB")
            return True
        except Exception as e:
            logger.error(f"Error caching locally: {e}")
            return False
    
    def exists(self, cache_key: str) -> bool:
        return self._cache_path(cache_key).exists()
    
    def get_stats(self) -> Dict[str, Any]:
        try:
            cache_files = list(self.cache_dir.glob("**/*.npz"))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            return {
                "strategy": "local_file",
                "enabled": True,
                "cache_dir": str(self.cache_dir),
                "tile_count": len(cache_files),
                "total_size_mb": round(total_size / (1024 * 1024), 2)
            }
        except Exception as e:
            return {"strategy": "local_file", "enabled": False, "error": str(e)}
    
    def clear(self, source_filter: Optional[str] = None) -> int:
        """Clear local cache files, optionally filtered by source."""
        try:
            if source_filter:
                # Clear only files matching the source pattern
                pattern = f"**/*{source_filter}*.npz"
            else:
                pattern = "**/*.npz"
            
            cache_files = list(self.cache_dir.glob(pattern))
            count = 0
            
            for cache_file in cache_files:
                try:
                    cache_file.unlink()
                    count += 1
                except Exception as e:
                    logger.warning(f"Could not delete {cache_file}: {e}")
            
            logger.info(f"Cleared {count} local cache files")
            return count
            
        except Exception as e:
            logger.error(f"Error clearing local cache: {e}")
            return 0

class GCSCache(LidarCacheStrategy):
    """Google Cloud Storage cache strategy."""
    
    def __init__(self, bucket_name: str = "lidar_cache", auto_create: bool = False):
        self.bucket_name = bucket_name
        self.auto_create = auto_create
        self.client = None
        self.bucket = None
        
        # Use same credentials as Earth Engine
        credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 
                                   'sage-striker-294302-b89a8b7e205b.json')
        project_id = os.getenv('GOOGLE_EE_PROJECT_ID', 'sage-striker-294302')
        
        self.credentials_path = credentials_path
        self.project_id = project_id
        self._initialized = self._initialize_gcs()
    
    def _initialize_gcs(self) -> bool:
        try:
            from google.cloud import storage
            from google.oauth2 import service_account
            from google.cloud import exceptions
            
            if os.path.exists(self.credentials_path):
                credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_path
                )
                self.client = storage.Client(
                    credentials=credentials,
                    project=self.project_id
                )
            else:
                self.client = storage.Client(project=self.project_id)
            
            self.bucket = self.client.bucket(self.bucket_name)
            
            # Test bucket access without listing (requires fewer permissions)
            try:
                # Try to get bucket metadata instead of listing contents
                self.bucket.reload()
                logger.info(f"✅ Connected to GCS bucket: {self.bucket_name}")
                return True
            except exceptions.NotFound:
                if self.auto_create:
                    try:
                        self.bucket = self.client.create_bucket(self.bucket_name, location="US")
                        logger.info(f"✅ Created and connected to GCS bucket: {self.bucket_name}")
                        return True
                    except Exception as create_error:
                        logger.warning(f"Cannot create bucket {self.bucket_name}: {create_error}")
                        return False
                else:
                    logger.warning(f"Bucket {self.bucket_name} does not exist")
                    return False
            except exceptions.Forbidden:
                logger.warning(f"No permission to access bucket {self.bucket_name}")
                return False
                
        except ImportError:
            logger.warning("google-cloud-storage not available")
            return False
        except Exception as e:
            logger.warning(f"GCS initialization failed: {e}")
            return False
    
    def _blob_path(self, cache_key: str) -> str:
        return f"tiles/{cache_key}.npz"
    
    def get(self, cache_key: str) -> Optional[np.ndarray]:
        if not self._initialized:
            return None
        
        try:
            blob = self.bucket.blob(self._blob_path(cache_key))
            if not blob.exists():
                return None
            
            blob_data = blob.download_as_bytes()
            with io.BytesIO(blob_data) as buffer:
                data = np.load(buffer)
                tile_data = data['elevation']
                metadata = data['metadata'].item()
            
            logger.info(f"✅ GCS cache hit: {cache_key} | Shape: {tile_data.shape}")
            return tile_data
        except Exception as e:
            logger.warning(f"Error loading from GCS cache: {e}")
            return None
    
    def put(self, cache_key: str, data: np.ndarray, metadata: Dict) -> bool:
        if not self._initialized:
            return False
        
        try:
            with io.BytesIO() as buffer:
                np.savez_compressed(
                    buffer,
                    elevation=data,
                    metadata=metadata
                )
                buffer.seek(0)
                blob_data = buffer.getvalue()
            
            blob = self.bucket.blob(self._blob_path(cache_key))
            blob.upload_from_string(
                blob_data,
                content_type='application/octet-stream'
            )
            
            logger.info(f"✅ Cached to GCS: {cache_key} | Size: {len(blob_data)/1024:.1f}KB")
            return True
        except Exception as e:
            logger.error(f"Error caching to GCS: {e}")
            return False
    
    def exists(self, cache_key: str) -> bool:
        if not self._initialized:
            return False
        
        try:
            blob = self.bucket.blob(self._blob_path(cache_key))
            return blob.exists()
        except Exception:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        if not self._initialized:
            return {"strategy": "gcs", "enabled": False, "reason": "not_initialized"}
        
        try:
            # Try to list a few tiles to test access
            blobs = list(self.client.list_blobs(self.bucket, prefix="tiles/", max_results=10))
            
            return {
                "strategy": "gcs",
                "enabled": True,
                "bucket": self.bucket_name,
                "project_id": self.project_id,
                "sample_tile_count": len(blobs)
            }
        except Exception as e:
            return {"strategy": "gcs", "enabled": False, "error": str(e)}

class HybridCache(LidarCacheStrategy):
    """Hybrid cache using both local and cloud storage."""
    
    def __init__(self, local_cache_dir: str = "./lidar_cache", gcs_bucket: str = "lidar_cache"):
        self.local_cache = LocalFileCache(local_cache_dir)
        self.gcs_cache = GCSCache(gcs_bucket)
        self._initialized = True
        
        logger.info(f"✅ Hybrid cache initialized (Local: {self.local_cache._initialized}, GCS: {self.gcs_cache._initialized})")
    
    def get(self, cache_key: str) -> Optional[np.ndarray]:
        # Try local cache first (fastest)
        data = self.local_cache.get(cache_key)
        if data is not None:
            return data
        
        # Try GCS cache
        data = self.gcs_cache.get(cache_key)
        if data is not None:
            # Store in local cache for next time
            metadata = {"retrieved_from": "gcs", "timestamp": datetime.utcnow().isoformat()}
            self.local_cache.put(cache_key, data, metadata)
        
        return data
    
    def put(self, cache_key: str, data: np.ndarray, metadata: Dict) -> bool:
        # Store in both caches
        local_success = self.local_cache.put(cache_key, data, metadata)
        gcs_success = self.gcs_cache.put(cache_key, data, metadata)
        
        return local_success or gcs_success  # Success if at least one works
    
    def exists(self, cache_key: str) -> bool:
        return self.local_cache.exists(cache_key) or self.gcs_cache.exists(cache_key)
    
    def get_stats(self) -> Dict[str, Any]:
        local_stats = self.local_cache.get_stats()
        gcs_stats = self.gcs_cache.get_stats()
        
        return {
            "strategy": "hybrid",
            "local": local_stats,
            "gcs": gcs_stats
        }
    
    def clear(self, source_filter: Optional[str] = None) -> int:
        """Clear cache from both local and GCS."""
        local_cleared = 0
        gcs_cleared = 0
        
        # Clear local cache
        if hasattr(self.local_cache, 'clear'):
            local_cleared = self.local_cache.clear(source_filter)
        
        # Clear GCS cache  
        if hasattr(self.gcs_cache, 'clear'):
            gcs_cleared = self.gcs_cache.clear(source_filter)
        
        return local_cleared + gcs_cleared

class MultiTileCache:
    """
    Main cache interface supporting multiple strategies and cache key generation.
    
    Supports both patch-by-patch and aggregated caching strategies.
    """
    
    STRATEGY_MAP = {
        "local": LocalFileCache,
        "gcs": GCSCache, 
        "hybrid": HybridCache
    }
    
    def __init__(self, strategy: str = "hybrid", **kwargs):
        self.strategy_name = strategy
        
        if strategy not in self.STRATEGY_MAP:
            logger.warning(f"Unknown strategy '{strategy}', falling back to 'local'")
            strategy = "local"
        
        self.cache_strategy = self.STRATEGY_MAP[strategy](**kwargs)
        logger.info(f"✅ Cache initialized with strategy: {strategy}")
    
    def _make_cache_key(self, 
                       lat: float, 
                       lon: float, 
                       size_m: int, 
                       resolution_m: float,
                       data_type: str,
                       source: str) -> str:
        """Generate cache key for individual patch."""
        lat_rounded = round(lat, 6)
        lon_rounded = round(lon, 6)
        res_rounded = round(resolution_m, 3)
        
        # Create readable but unique key
        params = f"{source}_{data_type}_{lat_rounded}_{lon_rounded}_{size_m}m_{res_rounded}m"
        # Add hash to handle any edge cases with special characters
        key_hash = hashlib.md5(params.encode()).hexdigest()[:8]
        
        return f"{params}_{key_hash}"
    
    def get(self, lat: float, lon: float, size_m: int, resolution_m: float,
            data_type: str, source: str) -> Optional[np.ndarray]:
        """Get cached tile."""
        cache_key = self._make_cache_key(lat, lon, size_m, resolution_m, data_type, source)
        return self.cache_strategy.get(cache_key)
    
    def put(self, lat: float, lon: float, size_m: int, resolution_m: float,
            data_type: str, source: str, tile_data: np.ndarray) -> bool:
        """Store tile in cache."""
        cache_key = self._make_cache_key(lat, lon, size_m, resolution_m, data_type, source)
        
        metadata = {
            "source": source,
            "data_type": data_type,
            "lat": lat,
            "lon": lon,
            "size_m": size_m,
            "resolution_m": resolution_m,
            "shape": tile_data.shape,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "stats": {
                "min": float(np.nanmin(tile_data)),
                "max": float(np.nanmax(tile_data)),
                "mean": float(np.nanmean(tile_data))
            }
        }
        
        return self.cache_strategy.put(cache_key, tile_data, metadata)
    
    def exists(self, lat: float, lon: float, size_m: int, resolution_m: float,
               data_type: str, source: str) -> bool:
        """Check if tile exists in cache."""
        cache_key = self._make_cache_key(lat, lon, size_m, resolution_m, data_type, source)
        return self.cache_strategy.exists(cache_key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache_strategy.get_stats()

# Global cache instance
_cache_instance: Optional[MultiTileCache] = None

def get_cache(strategy: str = "hybrid") -> MultiTileCache:
    """Get or create global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = MultiTileCache(strategy=strategy)
    return _cache_instance
