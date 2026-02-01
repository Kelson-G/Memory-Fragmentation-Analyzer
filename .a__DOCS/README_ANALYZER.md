# Memory Fragmentation Analyzer for vLLM

This project aims to build a comprehensive memory fragmentation analyzer for vLLM's KV cache system.

## Project Structure

```
Memory-Fragmentation-Analyzer/
├── KEY_FINDINGS.md           # Summary of vLLM's memory system architecture
├── ANALYZER_DESIGN.md        # Detailed design for the fragmentation analyzer
├── testing.py                # Testing script to explore vLLM locally
├── vllm/                     # vLLM source code
│   └── v1/
│       └── core/
│           ├── block_pool.py           # Block management
│           ├── kv_cache_utils.py       # Core data structures
│           ├── kv_cache_manager.py     # High-level interface
│           ├── kv_cache_metrics.py     # Existing metrics
│           └── kv_cache_coordinator.py # Multi-group coordination
└── README_ANALYZER.md        # This file
```

## Getting Started

### 1. Understand the System

Read these documents in order:

1. **KEY_FINDINGS.md** - Quick overview of how vLLM manages memory
   - Architecture overview
   - Core components and their roles
   - Memory lifecycle and allocation flows
   - Existing fragmentation sources

2. **ANALYZER_DESIGN.md** - Detailed design document
   - Proposed analyzer classes
   - Integration points
   - Key metrics to track
   - Implementation roadmap

### 2. Explore the Code

Run the testing script to interact with vLLM's memory system:

```bash
python testing.py
```

This script will:
- Explore KVCacheBlock structure
- Test FreeKVCacheBlockQueue operations
- Create and manipulate a BlockPool
- Simulate fragmentation scenarios
- Analyze memory layout patterns

### 3. Key Files to Study

Start with these files in the vLLM codebase:

1. **vllm/v1/core/kv_cache_utils.py**
   - `KVCacheBlock` dataclass (line 111)
   - `FreeKVCacheBlockQueue` class (line 157)
   - Block hashing functions

2. **vllm/v1/core/block_pool.py**
   - `BlockPool` class (line 129)
   - `BlockHashToBlockMap` class (line 28)
   - Allocation and eviction logic

3. **vllm/v1/core/kv_cache_manager.py**
   - `KVCacheManager` class (line 130)
   - `allocate_slots()` method (line 206)
   - Request lifecycle management

4. **vllm/v1/core/kv_cache_metrics.py**
   - `KVCacheMetricsCollector` class (line 40)
   - Existing metrics infrastructure to extend

## Proposed Analyzer Classes

### Core Classes

1. **FragmentationSnapshot**
   - Captures point-in-time memory state
   - Tracks free/allocated/cached blocks
   - Calculates fragmentation ratios

2. **FragmentationTracker**
   - Hooks into BlockPool events
   - Maintains time-series data
   - Captures snapshots periodically

3. **BlockLayoutAnalyzer**
   - Analyzes memory layout patterns
   - Identifies fragmentation hotspots
   - Visualizes memory state

4. **RequestBlockAnalyzer**
   - Tracks per-request allocation patterns
   - Analyzes request contribution to fragmentation
   - Maps requests to blocks

5. **FragmentationReporter**
   - Generates human-readable reports
   - Exports metrics to CSV/JSON
   - Creates visualizations

6. **MemoryFragmentationAnalyzer**
   - Main interface orchestrating all components
   - Provides start/stop monitoring
   - Generates comprehensive reports

## Key Metrics to Track

### External Fragmentation
- Number of contiguous free block runs
- Largest contiguous free run
- Average free run size
- External fragmentation ratio

### Internal Fragmentation
- Partially filled blocks
- Total unused token slots
- Internal fragmentation ratio

### Cache Metrics
- Cache hit rate
- Cache eviction rate
- Blocks evicted before reuse
- Average block lifetime in cache

### Performance Metrics
- Allocation success/failure rate
- Average allocation latency
- Blocks recycled vs. newly allocated

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
- [ ] Implement `FragmentationSnapshot`
- [ ] Add basic event hooks to `BlockPool`
- [ ] Calculate core fragmentation metrics
- [ ] Test with `testing.py` scenarios

### Phase 2: Tracking (Week 3-4)
- [ ] Implement `FragmentationTracker`
- [ ] Add time-series storage
- [ ] Integrate with existing metrics collector
- [ ] Add per-request tracking

### Phase 3: Analysis (Week 5-6)
- [ ] Implement `BlockLayoutAnalyzer`
- [ ] Build memory visualization
- [ ] Identify fragmentation patterns
- [ ] Add hotspot detection

### Phase 4: Reporting (Week 7-8)
- [ ] Implement `FragmentationReporter`
- [ ] Generate summary reports
- [ ] Export to CSV/JSON
- [ ] Create plots and graphs

### Phase 5: Integration (Week 9-10)
- [ ] Integrate with vLLM CLI
- [ ] Add configuration options
- [ ] Minimize performance overhead
- [ ] Write documentation

### Phase 6: Testing & Optimization (Week 11-12)
- [ ] Test with real workloads
- [ ] Benchmark overhead
- [ ] Optimize critical paths
- [ ] Validate metrics accuracy

## Development Workflow

### Step 1: Explore
```bash
# Run the testing script to understand the system
python testing.py
```

### Step 2: Implement
```bash
# Create a new file for your analyzer class
# Example: implementing FragmentationSnapshot
touch fragmentation_snapshot.py
```

### Step 3: Test
```bash
# Add tests to testing.py
# Run and validate
python testing.py
```

### Step 4: Integrate
```bash
# Modify BlockPool to call your hooks
# Test with real vLLM workloads
```

## Important Notes

### Memory System Characteristics
- **Block-based**: Fixed-size blocks (typically 16 tokens)
- **Pre-allocated**: All blocks created at initialization
- **Doubly-linked list**: O(1) insertion/removal for free blocks
- **Prefix caching**: Blocks can be shared across requests
- **Reference counting**: Prevents premature freeing

### Fragmentation Sources
1. **Interleaved allocations**: Different request lifetimes
2. **Partial blocks**: Last block may not be full
3. **Cache pollution**: Cached blocks never reused
4. **Variable sizes**: Small/large request mix

### Performance Considerations
- Use sampling to reduce overhead
- Avoid allocations in hot paths
- Leverage existing metrics infrastructure
- Consider async/background processing
- Measure and minimize impact

## Resources

### Documentation
- vLLM prefix caching design: `docs/design/prefix_caching.md`
- KV cache interface: `vllm/v1/kv_cache_interface.py`

### Related Code
- Scheduler: `vllm/v1/core/sched/scheduler.py`
- Request class: `vllm/v1/request.py`
- Metrics: `vllm/v1/metrics/stats.py`

## Questions to Answer

As you build the analyzer, aim to answer:

1. **What is the fragmentation pattern?**
   - External vs. internal fragmentation
   - Temporal patterns
   - Correlation with workload characteristics

2. **What causes fragmentation?**
   - Request size distribution
   - Request lifetime distribution
   - Cache hit/miss patterns

3. **What is the impact?**
   - Failed allocations
   - Reduced throughput
   - Inefficient memory usage

4. **How can we reduce it?**
   - Optimal block size
   - Better eviction policies
   - Request scheduling strategies

## Getting Help

If you encounter issues:

1. Check `KEY_FINDINGS.md` for architecture details
2. Review `ANALYZER_DESIGN.md` for design rationale
3. Run `testing.py` to validate your understanding
4. Examine the actual vLLM source code
5. Look at existing metrics in `kv_cache_metrics.py`

## License

This project follows the vLLM license (Apache 2.0).

---

**Last Updated**: January 31, 2026  
**Status**: Design Phase - Ready for Implementation
