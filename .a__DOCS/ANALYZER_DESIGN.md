# Memory Fragmentation Analyzer for vLLM - Design Document

## Overview
This document outlines the design for a memory fragmentation analyzer for vLLM's KV cache system. The analyzer will monitor, measure, and report on memory fragmentation in the block-based KV cache.

## Key vLLM Memory System Components

### 1. **KVCacheBlock** (`vllm/v1/core/kv_cache_utils.py`)
- Basic unit of memory allocation
- Contains:
  - `block_id`: Unique identifier (0 to num_gpu_blocks - 1)
  - `ref_cnt`: Reference count (how many requests use this block)
  - `_block_hash`: Hash key when block is full and cached
  - `prev_free_block`/`next_free_block`: Doubly linked list pointers
  - `is_null`: Flag for null/placeholder blocks

### 2. **BlockPool** (`vllm/v1/core/block_pool.py`)
- Central manager for all KVCacheBlocks
- Key methods:
  - `get_new_blocks(num_blocks)`: Allocate blocks from free pool
  - `free_blocks(blocks)`: Return blocks to free pool
  - `cache_full_blocks()`: Cache blocks for prefix caching
  - `get_num_free_blocks()`: Get count of available blocks
  - `evict_blocks()`: Evict cached blocks
  - `get_usage()`: Get KV cache usage percentage
- Contains:
  - `free_block_queue`: Doubly linked list of free blocks (FreeKVCacheBlockQueue)
  - `cached_block_hash_to_block`: Hash map for prefix caching (BlockHashToBlockMap)
  - `blocks`: List of all blocks in the system

### 3. **FreeKVCacheBlockQueue** (`vllm/v1/core/kv_cache_utils.py`)
- Manages free blocks as a doubly linked list
- Ordered by eviction priority (LRU-based)
- Operations: `popleft()`, `append()`, `remove()`

### 4. **KVCacheManager** (`vllm/v1/core/kv_cache_manager.py`)
- High-level interface for block allocation
- Key methods:
  - `allocate_slots()`: Allocate blocks for new tokens
  - `free()`: Free blocks for a completed request
  - `cache_blocks()`: Cache computed blocks
  - `get_computed_blocks()`: Get prefix-cached blocks

### 5. **KVCacheMetricsCollector** (`vllm/v1/core/kv_cache_metrics.py`)
- Currently tracks block lifecycle metrics with sampling
- Monitors:
  - Block allocation/eviction events
  - Block lifetime and idle time
  - Access patterns and reuse gaps

### 6. **Scheduler** (`vllm/v1/core/sched/scheduler.py`)
- Decides which requests to schedule
- Calls KVCacheManager to allocate/free blocks
- Main entry point for request processing

## Memory Fragmentation in vLLM

### Types of Fragmentation to Track:

1. **External Fragmentation**
   - Free blocks scattered throughout memory (not contiguous)
   - Measured by: number of free block "islands" in the free queue
   - Impact: May prevent large allocations even with sufficient total free blocks

2. **Internal Fragmentation**
   - Partially filled blocks (wasted space within allocated blocks)
   - Measured by: unused token slots in allocated blocks
   - Impact: Inefficient memory utilization

3. **Temporal Fragmentation**
   - Blocks becoming free at different times, creating allocation gaps
   - Measured by: time distribution of block allocations/deallocations
   - Impact: Affects allocation efficiency over time

4. **Prefix Cache Fragmentation**
   - Cached blocks that are evicted before reuse
   - Measured by: eviction rate vs. hit rate
   - Impact: Reduced prefix caching effectiveness

## Proposed Analyzer Classes

### Class 1: **FragmentationSnapshot**
```python
class FragmentationSnapshot:
    """
    Captures a point-in-time snapshot of memory fragmentation state.
    """
    timestamp: float
    total_blocks: int
    free_blocks: int
    allocated_blocks: int
    cached_blocks: int
    
    # External fragmentation metrics
    num_free_block_runs: int  # Number of contiguous free block sequences
    largest_free_run: int     # Largest contiguous free block sequence
    average_free_run: float   # Average size of free runs
    
    # Internal fragmentation metrics
    partially_filled_blocks: int  # Blocks with unused slots
    total_unused_slots: int       # Total wasted token slots
    internal_frag_ratio: float    # Ratio of wasted space
    
    # Cache metrics
    num_cached_blocks: int
    cache_hit_rate: float
    cache_eviction_rate: float
```

### Class 2: **FragmentationTracker**
```python
class FragmentationTracker:
    """
    Continuously tracks fragmentation metrics over time.
    Hooks into BlockPool events to capture state changes.
    """
    
    def on_block_allocated(self, block: KVCacheBlock) -> None:
        """Called when a block is allocated."""
        
    def on_block_freed(self, block: KVCacheBlock) -> None:
        """Called when a block is freed."""
        
    def on_block_cached(self, block: KVCacheBlock) -> None:
        """Called when a block is cached."""
        
    def on_block_evicted(self, block: KVCacheBlock) -> None:
        """Called when a cached block is evicted."""
        
    def capture_snapshot(self) -> FragmentationSnapshot:
        """Capture current fragmentation state."""
        
    def get_history(self) -> list[FragmentationSnapshot]:
        """Get historical snapshots."""
```

### Class 3: **BlockLayoutAnalyzer**
```python
class BlockLayoutAnalyzer:
    """
    Analyzes the physical layout of blocks in memory.
    Identifies fragmentation patterns and hotspots.
    """
    
    def analyze_free_block_distribution(self, block_pool: BlockPool) -> dict:
        """Analyze distribution of free blocks."""
        
    def find_fragmentation_hotspots(self, block_pool: BlockPool) -> list[tuple[int, int]]:
        """Find regions with high fragmentation."""
        
    def calculate_defragmentation_potential(self, block_pool: BlockPool) -> float:
        """Calculate how much could be gained by defragmentation."""
        
    def visualize_memory_layout(self, block_pool: BlockPool) -> str:
        """Create ASCII visualization of memory layout."""
```

### Class 4: **RequestBlockAnalyzer**
```python
class RequestBlockAnalyzer:
    """
    Analyzes block allocation patterns per request.
    Tracks how requests contribute to fragmentation.
    """
    
    def track_request_allocation(self, request_id: str, blocks: list[KVCacheBlock]) -> None:
        """Track block allocation for a request."""
        
    def analyze_request_fragmentation(self, request_id: str) -> dict:
        """Analyze how fragmented a request's blocks are."""
        
    def get_request_block_map(self) -> dict[str, list[int]]:
        """Get mapping of requests to their blocks."""
```

### Class 5: **FragmentationReporter**
```python
class FragmentationReporter:
    """
    Generates reports and visualizations of fragmentation metrics.
    """
    
    def generate_summary_report(self, snapshots: list[FragmentationSnapshot]) -> str:
        """Generate human-readable summary report."""
        
    def export_metrics_csv(self, snapshots: list[FragmentationSnapshot], filepath: str) -> None:
        """Export metrics to CSV for analysis."""
        
    def plot_fragmentation_over_time(self, snapshots: list[FragmentationSnapshot]) -> None:
        """Create time-series plots of fragmentation metrics."""
        
    def generate_recommendations(self, analysis_results: dict) -> list[str]:
        """Generate recommendations to reduce fragmentation."""
```

### Class 6: **MemoryFragmentationAnalyzer** (Main Interface)
```python
class MemoryFragmentationAnalyzer:
    """
    Main interface for the memory fragmentation analyzer.
    Orchestrates all analysis components.
    """
    
    def __init__(self, block_pool: BlockPool, enable_tracking: bool = True):
        self.block_pool = block_pool
        self.tracker = FragmentationTracker()
        self.layout_analyzer = BlockLayoutAnalyzer()
        self.request_analyzer = RequestBlockAnalyzer()
        self.reporter = FragmentationReporter()
        
    def start_monitoring(self) -> None:
        """Start continuous monitoring."""
        
    def stop_monitoring(self) -> None:
        """Stop monitoring."""
        
    def analyze_current_state(self) -> dict:
        """Analyze current fragmentation state."""
        
    def generate_full_report(self, filepath: str) -> None:
        """Generate comprehensive analysis report."""
```

## Integration Points

### 1. Hook into BlockPool
- Modify `BlockPool` to call tracker methods on allocation/free events
- Alternatively, wrap BlockPool methods with monitoring decorators

### 2. Hook into KVCacheManager
- Track high-level allocation patterns
- Monitor request-level block usage

### 3. Periodic Snapshots
- Use a background thread or timer to capture snapshots
- Store snapshots in a time-series database or circular buffer

### 4. Export Metrics
- Integrate with existing metrics systems (Prometheus, etc.)
- Add custom logging for fragmentation events

## Key Metrics to Track

1. **Utilization Metrics**
   - Total blocks allocated vs. total blocks available
   - Percentage of free blocks
   - Percentage of cached blocks

2. **Fragmentation Metrics**
   - External fragmentation ratio
   - Internal fragmentation ratio
   - Number of free block runs
   - Average free run length

3. **Cache Metrics**
   - Cache hit rate
   - Cache eviction rate
   - Average block lifetime in cache
   - Prefix cache utilization

4. **Performance Metrics**
   - Allocation success rate
   - Average allocation latency
   - Number of failed allocations due to fragmentation

## Next Steps

1. Implement `FragmentationSnapshot` class
2. Implement `FragmentationTracker` with event hooks
3. Create `BlockLayoutAnalyzer` for pattern detection
4. Build `FragmentationReporter` for visualization
5. Integrate with vLLM's existing metrics infrastructure
6. Add testing and benchmarking
7. Document usage and best practices

## Notes

- vLLM uses a block-based system (similar to paging in OS)
- Default block size is typically 16 tokens
- Prefix caching allows block reuse across requests
- Free blocks managed in LRU order for efficient eviction
- Reference counting prevents premature block freeing
