"""
Performance Monitoring and Caching for LangGraph Workflow
"""

import time
import logging
from typing import Dict, Any, Optional, Callable
from functools import wraps
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SimpleCache:
    """Simple in-memory cache with TTL support."""
    
    def __init__(self, default_ttl: int = 300):  # 5 minutes default
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key in self.cache:
            entry = self.cache[key]
            if datetime.now() < entry["expires"]:
                logger.debug(f"Cache hit for key: {key}")
                return entry["value"]
            else:
                del self.cache[key]
                logger.debug(f"Cache expired for key: {key}")
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with TTL."""
        ttl = ttl or self.default_ttl
        expires = datetime.now() + timedelta(seconds=ttl)
        self.cache[key] = {"value": value, "expires": expires}
        logger.debug(f"Cache set for key: {key}, TTL: {ttl}s")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)


class PerformanceMonitor:
    """Monitor performance metrics for workflow execution."""
    
    def __init__(self):
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.start_times: Dict[str, float] = {}
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self.start_times[operation] = time.time()
        logger.debug(f"Started timer for: {operation}")
    
    def end_timer(self, operation: str) -> float:
        """End timing and record duration."""
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            del self.start_times[operation]
            
            # Record metrics
            if operation not in self.metrics:
                self.metrics[operation] = {
                    "count": 0,
                    "total_time": 0.0,
                    "avg_time": 0.0,
                    "min_time": float('inf'),
                    "max_time": 0.0
                }
            
            metrics = self.metrics[operation]
            metrics["count"] += 1
            metrics["total_time"] += duration
            metrics["avg_time"] = metrics["total_time"] / metrics["count"]
            metrics["min_time"] = min(metrics["min_time"], duration)
            metrics["max_time"] = max(metrics["max_time"], duration)
            
            logger.debug(f"Operation {operation} completed in {duration:.3f}s")
            return duration
        return 0.0
    
    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all performance metrics."""
        return self.metrics.copy()
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        self.start_times.clear()
        logger.info("Performance metrics reset")


# Global instances
_cache = SimpleCache()
_monitor = PerformanceMonitor()


def get_cache() -> SimpleCache:
    """Get the global cache instance."""
    return _cache


def get_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return _monitor


def cached(ttl: int = 300, key_func: Optional[Callable] = None):
    """
    Decorator to cache function results.
    
    Args:
        ttl: Time to live in seconds
        key_func: Function to generate cache key from args/kwargs
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
            
            # Try to get from cache
            cached_result = _cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            _cache.set(cache_key, result, ttl)
            return result
        
        return wrapper
    return decorator


def timed(operation_name: Optional[str] = None):
    """
    Decorator to time function execution.
    
    Args:
        operation_name: Custom name for the operation
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            _monitor.start_timer(op_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                _monitor.end_timer(op_name)
        
        return wrapper
    return decorator


def memory_optimized(func: Callable) -> Callable:
    """
    Decorator for basic memory optimization.
    Clears local variables and forces garbage collection for large operations.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Basic memory cleanup for large operations
            import gc
            gc.collect()
    
    return wrapper