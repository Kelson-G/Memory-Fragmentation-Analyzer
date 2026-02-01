# vLLM Memory Management - Key Findings Summary

## Architecture Overview

vLLM uses a **block-based memory management system** similar to operating system paging. The KV cache (which stores key-value tensors for attention) is divided into fixed-size blocks.

### Core Components Hierarchy

```
Scheduler
    └── KVCacheManager
            └── KVCacheCoordinator
                    └── SingleTypeKVCacheManager (per attention type)
                            └── BlockPool
                                    ├── FreeKVCacheBlockQueue (doubly linked list)
                                    ├── BlockHashToBlockMap (prefix cache lookup)
                                    └── blocks: list[KVCacheBlock]
```

## Critical Files to Understand

### 1. **vllm/v1/core/block_pool.py** (490 lines)
- **BlockPool**: Main class managing all blocks
- **BlockHashToBlockMap**: Cache lookup for prefix caching
- Key insight: Blocks are pre-allocated and reused (no dynamic allocation)

### 2. **vllm/v1/core/kv_cache_utils.py** (1645 lines)
- **KVCacheBlock**: The fundamental unit (dataclass)
- **FreeKVCacheBlockQueue**: Manages free blocks with LRU eviction
- **Block hashing functions**: For prefix caching
- Key insight: Uses doubly linked list for O(1) insertion/removal

### 3. **vllm/v1/core/kv_cache_manager.py** 
- **KVCacheManager**: High-level interface
- **KVCacheBlocks**: Wrapper for allocated blocks per request
- Main method: `allocate_slots()` - allocates blocks for new tokens
- Key insight: Handles both new allocations and prefix cache hits

### 4. **vllm/v1/core/kv_cache_metrics.py**
- **KVCacheMetricsCollector**: Existing metrics infrastructure
- **BlockMetricsState**: Per-block lifecycle tracking
- Uses sampling to reduce overhead
- Key insight: Already has some monitoring infrastructure we can extend

### 5. **vllm/v1/core/kv_cache_coordinator.py**
- **KVCacheCoordinator**: Coordinates multiple KV cache groups
- Handles different attention types (self-attention, cross-attention, mamba)
- Key insight: Different attention mechanisms may have different block sizes

## Memory Lifecycle

### Block States
1. **Free**: In free_block_queue, available for allocation (ref_cnt=0)
2. **Allocated**: Assigned to a request (ref_cnt>0)
3. **Cached**: Full block with hash, can be reused (has block_hash)
4. **Null**: Special placeholder block (is_null=True)

### Allocation Flow
```
1. Request arrives → Scheduler calls KVCacheManager.allocate_slots()
2. Check for prefix cache hit → find_longest_cache_hit()
3. If hit: touch cached blocks (increment ref_cnt, remove from free queue)
4. If miss: allocate new blocks from free_block_queue.popleft_n()
5. When block becomes full: compute hash, add to cached_block_hash_to_block
6. On eviction: reset hash, decrement ref_cnt, add back to free queue
```

### Deallocation Flow
```
1. Request completes → Scheduler calls KVCacheManager.free()
2. Decrement ref_cnt for all blocks
3. If ref_cnt reaches 0: append to free_block_queue
4. Blocks remain in cache (keep hash) until evicted
5. On next allocation, if free block has hash: evict from cache
```

## Prefix Caching System

**Purpose**: Reuse KV cache blocks across requests with common prefixes

**How it works**:
- Each block gets a hash based on its token IDs
- Full blocks are cached in `BlockHashToBlockMap`
- New requests check cache before allocating
- Cache hits increment ref_cnt (blocks stay alive)
- LRU eviction: oldest cached blocks with ref_cnt=0 are evicted first

**Important**: Blocks can be in BOTH the cache and the free queue simultaneously!
- This allows them to be reused OR evicted depending on what happens first

## Fragmentation Sources

### 1. **Interleaved Allocations/Deallocations**
   - Requests with different lifetimes create gaps
   - Example: Req1 allocates blocks 0-10, Req2 allocates 11-20, Req1 finishes
   - Result: Blocks 0-10 free, 11-20 allocated → fragmentation

### 2. **Partial Block Usage**
   - Last block of a request may not be full
   - Example: 35 tokens with block_size=16 → 3 blocks, last has 3/16 slots used
   - Result: 13 wasted slots in last block

### 3. **Cache Pollution**
   - Blocks cached but never reused before eviction
   - Takes up space in free queue without benefit
   - Can prevent fresh allocations

### 4. **Variable Request Sizes**
   - Small requests create small allocation gaps
   - Large requests may fail even with sufficient total free blocks

## Existing Metrics Infrastructure

### Current Metrics (KVCacheMetricsCollector)
- Block lifetime (birth to eviction)
- Idle time (last access to eviction)
- Reuse gaps (time between accesses)
- Sampling rate: configurable (default 1%)

### Missing Metrics (What We Need to Add)
- External fragmentation ratio
- Internal fragmentation (wasted slots)
- Free block distribution (contiguous runs)
- Allocation failure patterns
- Cache effectiveness (hit rate, eviction before reuse)

## Design Insights

### Why Block-Based?
1. **Fixed-size allocations**: No need for complex memory allocator
2. **Easy to track**: Each block has an ID, easy to reference
3. **Cache-friendly**: GPU memory accessed in block-sized chunks
4. **Prefix caching**: Hash entire blocks, not individual tokens

### Why Doubly Linked List?
1. **O(1) operations**: Insert/remove anywhere in queue
2. **LRU eviction**: Move blocks to back on access
3. **No Python object creation**: Manipulate existing blocks
4. **Cache-aware**: Remove blocks from free queue when cache hit

### Why Pre-allocate All Blocks?
1. **Avoid Python GC overhead**: No object creation/destruction
2. **Predictable memory usage**: Know total memory upfront
3. **Fast initialization**: Create all blocks once
4. **Stable block IDs**: IDs never change, simplifies tracking

## Memory Fragmentation Analyzer - Implementation Strategy

### Phase 1: Snapshot & Basic Metrics
1. Implement `FragmentationSnapshot` to capture state
2. Add hooks to BlockPool for allocation/free events
3. Calculate basic metrics: external frag, internal frag, free runs

### Phase 2: Continuous Tracking
1. Implement `FragmentationTracker` with event handlers
2. Store time-series snapshots
3. Track per-request fragmentation patterns

### Phase 3: Analysis & Visualization
1. Implement `BlockLayoutAnalyzer` for pattern detection
2. Build ASCII/visual memory layout display
3. Identify fragmentation hotspots

### Phase 4: Reporting & Optimization
1. Implement `FragmentationReporter` for reports
2. Generate recommendations (e.g., adjust block size)
3. Export metrics for external analysis

### Phase 5: Integration
1. Integrate with existing metrics collector
2. Add CLI flags for enabling/configuring analyzer
3. Minimize performance overhead (sampling, async)

## Important Considerations

### Performance Impact
- Minimize overhead: use sampling, avoid allocations
- Leverage existing metrics infrastructure
- Async/background processing for heavy analysis
- Don't block allocation/deallocation paths

### Compatibility
- Work with existing prefix caching system
- Support multiple KV cache groups (different attention types)
- Handle dynamic configurations (different block sizes)

### Testing
- Use `testing.py` to explore and validate
- Create synthetic fragmentation scenarios
- Measure overhead vs. baseline
- Validate metrics accuracy

## Quick Reference

### Block Size
- Typically 16 tokens per block (configurable)
- hash_block_size: granularity for hashing
- Actual block_size can be multiple of hash_block_size

### Reference Counting
- ref_cnt=0: Block is free (in free_block_queue)
- ref_cnt>0: Block is in use
- ref_cnt can be >1 if multiple requests share (prefix caching)

### Block Hashing
- BlockHash: bytes (hash of token IDs)
- BlockHashWithGroupId: BlockHash + 4-byte group_id
- Computed incrementally as tokens arrive
- Only full blocks are cached

### Null Block
- Special block at index 0
- Used as placeholder (sparse attention patterns)
- Never allocated or freed
- Always marked as used

## Next Steps

1. ✅ Understand the architecture (done)
2. ✅ Identify key components (done)
3. ✅ Design analyzer classes (done)
4. ✅ Create testing script (done)
5. → Run testing.py to validate understanding
6. → Implement FragmentationSnapshot class
7. → Add event hooks to BlockPool
8. → Build visualization tools
9. → Test with real workloads
10. → Optimize and deploy
