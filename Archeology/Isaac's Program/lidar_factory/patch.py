import numpy as np
import threading
import queue
import time
import logging
from typing import Dict, Tuple, List, Optional, Any
from collections import OrderedDict

from .factory import LidarMapFactory, DEFAULT_DATA_TYPE # Assuming factory.py is in the same directory

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LidarRoamCache:
    """
    A roaming-aware cache for LIDAR data patches with prefetching capabilities.
    It uses LidarMapFactory to fetch patches that are not in the cache.
    """
    def __init__(self, 
                 default_patch_size_m: int = 128, 
                 max_cache_size: int = 100, # Max number of patches to store
                 prefetch_distance_m: Optional[float] = None, # How far ahead to prefetch (e.g., 2*patch_size_m)
                 prefetch_steps: int = 2 # Number of steps in cardinal directions to prefetch
                 ):
        """
        Initializes the LidarRoamCache.

        Args:
            default_patch_size_m: The default edge size of square patches in meters if not specified in get().
            max_cache_size: Maximum number of patches to keep in the LRU cache.
            prefetch_distance_m: If set, defines a radius for prefetching. Overrides prefetch_steps if both are set.
            prefetch_steps: Number of patch_size_m steps to prefetch in N, S, E, W directions.
        """
        self.default_patch_size_m = default_patch_size_m
        self.cache: OrderedDict[Tuple[float, float, int, str, Optional[float]], np.ndarray] = OrderedDict()
        self.max_cache_size = max_cache_size
        
        self.prefetch_queue = queue.Queue()
        self.prefetch_thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self.prefetch_active = threading.Event()
        self.stop_event = threading.Event()

        self.prefetch_distance_m = prefetch_distance_m
        self.prefetch_steps = prefetch_steps

        logger.info(f"LidarRoamCache initialized: patch_size={default_patch_size_m}m, max_cache={max_cache_size}, prefetch_steps={prefetch_steps}")

    def _make_cache_key(self, lat: float, lon: float, size_m: int, data_type: str, resolution_m: Optional[float]) -> Tuple[float, float, int, str, Optional[float]]:
        """Creates a unique key for caching. Resolution is rounded to avoid floating point issues."""
        # Round lat/lon to a certain precision to make keys more robust to tiny floating point variations if needed
        # For now, using them directly. Resolution is critical for uniqueness.
        rounded_resolution_m = round(resolution_m, 3) if resolution_m is not None else None
        return (round(lat, 6), round(lon, 6), size_m, data_type.upper(), rounded_resolution_m)

    def get(self, 
            lat: float, 
            lon: float, 
            size_m: Optional[int] = None, 
            preferred_resolution_m: Optional[float] = None, 
            preferred_data_type: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Retrieves a LIDAR patch. Checks cache first, then fetches using LidarMapFactory.
        Also enqueues prefetch requests for surrounding areas.
        """
        current_size_m = size_m if size_m is not None else self.default_patch_size_m
        current_data_type = (preferred_data_type or DEFAULT_DATA_TYPE).upper()

        cache_key = self._make_cache_key(lat, lon, current_size_m, current_data_type, preferred_resolution_m)

        if cache_key in self.cache:
            # Move accessed item to the end to mark it as recently used (for LRU)
            self.cache.move_to_end(cache_key)
            logger.info(f"Cache HIT for key: {cache_key}")
            patch_data = self.cache[cache_key]
            # Still enqueue prefetches even on cache hit, as user might be moving to a new area
            if self.prefetch_active.is_set():
                self._enqueue_surrounding_prefetches(lat, lon, current_size_m, preferred_resolution_m, current_data_type)
            return patch_data
        
        logger.info(f"Cache MISS for key: {cache_key}. Fetching from LidarMapFactory.")
        patch_data = LidarMapFactory.get_patch(
            lat=lat, 
            lon=lon, 
            size_m=current_size_m, 
            preferred_resolution_m=preferred_resolution_m, 
            preferred_data_type=current_data_type
        )

        if patch_data is not None:
            self._add_to_cache(cache_key, patch_data)
            if self.prefetch_active.is_set():
                self._enqueue_surrounding_prefetches(lat, lon, current_size_m, preferred_resolution_m, current_data_type)
            return patch_data
        else:
            logger.warning(f"Failed to fetch patch for key: {cache_key} from factory.")
            return None

    def _add_to_cache(self, key: Tuple[float, float, int, str, Optional[float]], patch_data: np.ndarray):
        """Adds a patch to the cache, enforcing LRU policy if max_cache_size is exceeded."""
        if key in self.cache:
            self.cache.move_to_end(key) # Mark as recently used
        self.cache[key] = patch_data
        if len(self.cache) > self.max_cache_size:
            oldest_key = self.cache.popitem(last=False) # Remove the least recently used item
            logger.info(f"Cache full. Evicted oldest item: {oldest_key[0]}")

    def _enqueue_surrounding_prefetches(self, lat: float, lon: float, size_m: int, resolution_m: Optional[float], data_type: str):
        """
        Enqueues prefetch requests for patches surrounding the given center point.
        Prefetches in N, S, E, W directions by `self.prefetch_steps` * `size_m`.
        Note: Latitude/Longitude degree changes for a given meter distance vary with latitude.
        This is a simplified prefetch based on adding/subtracting to lat/lon degrees.
        A more accurate method would convert meters to degrees at that specific latitude.
        For simplicity, using an approximation: 1 degree lat ~= 111km, 1 degree lon ~= 111km * cos(lat).
        """
        if self.prefetch_distance_m is not None:
            # TODO: Implement prefetch_distance_m logic (more complex due to geo calculations)
            logger.warning("prefetch_distance_m not yet fully implemented for _enqueue_surrounding_prefetches. Using step based.")

        # Approximate conversion: 1 meter in degrees (latitude is simpler)
        m_to_deg_lat = 1.0 / 111000.0 
        m_to_deg_lon = 1.0 / (111000.0 * np.cos(np.radians(lat))) 

        step_dist_lat = self.prefetch_steps * size_m * m_to_deg_lat
        step_dist_lon = self.prefetch_steps * size_m * m_to_deg_lon

        # Prefetch N, S, E, W
        prefetch_locations = [
            (lat + step_dist_lat, lon), # North
            (lat - step_dist_lat, lon), # South
            (lat, lon + step_dist_lon), # East
            (lat, lon - step_dist_lon), # West
            # Optional: Add diagonals
            # (lat + step_dist_lat, lon + step_dist_lon), # NE
            # ... and so on for NW, SE, SW
        ]

        for pf_lat, pf_lon in prefetch_locations:
            self.enqueue_prefetch(pf_lat, pf_lon, size_m, resolution_m, data_type)

    def enqueue_prefetch(self, lat: float, lon: float, size_m: int, resolution_m: Optional[float], data_type: str):
        """Enqueues a specific location for prefetching if not already cached or being fetched."""
        if not self.prefetch_active.is_set():
            return
        
        cache_key = self._make_cache_key(lat, lon, size_m, data_type, resolution_m)
        if cache_key not in self.cache: # Could also check if it's already in prefetch_queue but queue doesn't support `in` efficiently
            try:
                self.prefetch_queue.put_nowait((lat, lon, size_m, resolution_m, data_type, cache_key))
                logger.debug(f"Enqueued prefetch for: {cache_key}")
            except queue.Full:
                logger.warning("Prefetch queue is full. Skipping prefetch.")

    def _prefetch_loop(self):
        """Background thread loop that fetches items from the prefetch_queue."""
        logger.info("Prefetch thread started.")
        while not self.stop_event.is_set():
            if not self.prefetch_active.is_set():
                time.sleep(0.5) # Sleep if prefetching is paused
                continue
            try:
                lat, lon, size_m, resolution_m, data_type, cache_key = self.prefetch_queue.get(timeout=1) # Timeout to allow checking stop_event
                
                if cache_key not in self.cache: # Double check, might have been fetched by a direct get() call
                    logger.info(f"Prefetching: {cache_key}")
                    patch_data = LidarMapFactory.get_patch(
                        lat=lat, 
                        lon=lon, 
                        size_m=size_m, 
                        preferred_resolution_m=resolution_m, 
                        preferred_data_type=data_type
                    )
                    if patch_data is not None:
                        self._add_to_cache(cache_key, patch_data)
                        logger.info(f"Prefetched and cached: {cache_key}")
                    else:
                        logger.warning(f"Prefetch failed for: {cache_key}")
                else:
                    logger.debug(f"Skipping prefetch for already cached item: {cache_key}")
                self.prefetch_queue.task_done()
            except queue.Empty:
                continue # No items to prefetch, loop again
            except Exception as e:
                logger.error(f"Error in prefetch loop: {e}", exc_info=True)
                # Avoid continuous error loops for certain errors if needed
                time.sleep(5) # Wait a bit after an unexpected error
        logger.info("Prefetch thread stopped.")

    def start_prefetching(self):
        """Starts the prefetching background thread and enables prefetching."""
        if not self.prefetch_thread.is_alive():
            self.stop_event.clear()
            self.prefetch_active.set()
            self.prefetch_thread = threading.Thread(target=self._prefetch_loop, daemon=True) # Recreate if stopped
            self.prefetch_thread.start()
            logger.info("LidarRoamCache prefetching started.")
        else:
            self.prefetch_active.set()
            logger.info("LidarRoamCache prefetching resumed.")

    def pause_prefetching(self):
        """Pauses prefetching without stopping the thread (can be resumed)."""
        self.prefetch_active.clear()
        logger.info("LidarRoamCache prefetching paused.")

    def stop_prefetching(self):
        """Stops the prefetching background thread gracefully."""
        logger.info("Stopping LidarRoamCache prefetching...")
        self.prefetch_active.clear() # Signal to pause adding new items
        self.stop_event.set() # Signal the loop to terminate
        # Drain the queue to allow thread to exit if it's waiting on get()
        # This is a bit aggressive, might lose pending prefetches but ensures shutdown.
        while not self.prefetch_queue.empty():
            try:
                self.prefetch_queue.get_nowait()
                self.prefetch_queue.task_done()
            except queue.Empty:
                break
        if self.prefetch_thread.is_alive():
             self.prefetch_thread.join(timeout=5) # Wait for thread to finish
        if self.prefetch_thread.is_alive():
            logger.warning("Prefetch thread did not stop in time.")
        else:
            logger.info("LidarRoamCache prefetching stopped.")

    def clear_cache(self):
        """Clears all items from the cache."""
        self.cache.clear()
        logger.info("LidarRoamCache cleared.")

# Example Usage (for testing directly)
if __name__ == '__main__':
    print("--- LidarRoamCache Test --- ")
    # Ensure GEE is authenticated before running this test.

    # Initialize the cache
    # Using a small cache size for testing eviction
    roam_cache = LidarRoamCache(default_patch_size_m=64, max_cache_size=5, prefetch_steps=1)
    roam_cache.start_prefetching()

    # Test coordinates (Zaanse Schans)
    lat1, lon1 = 52.4746, 4.8163
    lat2, lon2 = 52.4750, 4.8160 # Slightly different location
    lat3, lon3 = 52.4755, 4.8157 # Another one

    # First get - should be a cache miss, then fetch
    print(f"\n1. Getting patch for ({lat1:.4f}, {lon1:.4f})")
    patch1 = roam_cache.get(lat1, lon1, preferred_resolution_m=0.5, preferred_data_type="DSM")
    if patch1 is not None:
        print(f"   Got patch1. Shape: {patch1.shape}. Cache size: {len(roam_cache.cache)}")
    else:
        print("   Failed to get patch1.")

    time.sleep(2) # Give some time for potential prefetches to be processed

    # Second get - for the same location, should be a cache hit
    print(f"\n2. Getting patch again for ({lat1:.4f}, {lon1:.4f}) - expecting cache HIT")
    patch1_again = roam_cache.get(lat1, lon1, preferred_resolution_m=0.5, preferred_data_type="DSM")
    if patch1_again is not None:
        print(f"   Got patch1_again. Shape: {patch1_again.shape}. Cache size: {len(roam_cache.cache)}")
    else:
        print("   Failed to get patch1_again.")

    # Third get - new location, should be a miss
    print(f"\n3. Getting patch for new location ({lat2:.4f}, {lon2:.4f})")
    patch2 = roam_cache.get(lat2, lon2, preferred_resolution_m=0.5, preferred_data_type="DSM")
    if patch2 is not None:
        print(f"   Got patch2. Shape: {patch2.shape}. Cache size: {len(roam_cache.cache)}")
    else:
        print("   Failed to get patch2.")

    time.sleep(3) # More time for prefetches
    print(f"   Current cache keys after some time: {list(roam_cache.cache.keys())}")

    # Fill up the cache to test eviction
    print("\n4. Filling cache to test eviction...")
    for i in range(roam_cache.max_cache_size + 2):
        current_lat = lat1 + (i * 0.001) # Create slightly different locations
        print(f"   Getting patch for ({current_lat:.4f}, {lon1:.4f})")
        p = roam_cache.get(current_lat, lon1, preferred_resolution_m=0.5, preferred_data_type="DSM")
        if p is not None:
            print(f"      Got patch. Shape: {p.shape}. Cache size: {len(roam_cache.cache)}")
            # Small delay to allow prefetch queue to be processed and not overwhelm GEE
            if i < roam_cache.max_cache_size: time.sleep(1) 
        else:
            print("      Failed to get a patch during fill.")
        if len(roam_cache.cache) == roam_cache.max_cache_size:
            print(f"   Cache reached max size: {roam_cache.max_cache_size}. Next get should evict.")
    
    print(f"\nFinal cache keys: {list(roam_cache.cache.keys())}")
    print(f"Final cache size: {len(roam_cache.cache)}")

    # Test pausing and resuming prefetching
    print("\n5. Testing pause and resume prefetching")
    roam_cache.pause_prefetching()
    # This get should not trigger new prefetches in the queue while paused
    patch_paused = roam_cache.get(lat3, lon3, preferred_resolution_m=0.5, preferred_data_type="DSM") 
    print(f"   Queue size while paused (approx): {roam_cache.prefetch_queue.qsize()}")
    roam_cache.start_prefetching() # Resume
    # This get should trigger prefetches again
    patch_resumed = roam_cache.get(lat3 + 0.0001, lon3 + 0.0001, preferred_resolution_m=0.5, preferred_data_type="DSM") 
    time.sleep(2)
    print(f"   Queue size after resume and get (approx): {roam_cache.prefetch_queue.qsize()}")


    # Stop prefetching and clear cache
    roam_cache.stop_prefetching()
    roam_cache.clear_cache()
    print(f"Cache size after clear: {len(roam_cache.cache)}")

    print("\n--- Test Complete ---")
