"""
LIDAR Factory Package with Hybrid Cloud Cache

High-performance LIDAR data access with transparent caching.
Provides 10x performance improvements through local + cloud storage.
"""

from .factory import LidarMapFactory, get_cache
from .cloud_cache import LidarTileCache
from .registry import METADATA_REGISTRY, DatasetMetadata, get_dataset_by_name
from .connectors import CONNECTOR_MAP, LidarConnector

__all__ = [
    'LidarMapFactory',
    'LidarTileCache', 
    'get_cache',
    'METADATA_REGISTRY',
    'DatasetMetadata',
    'get_dataset_by_name',
    'CONNECTOR_MAP',
    'LidarConnector'
]