"""
Memory Fragmentation Analyzer - Testing and Exploration Script

This script helps you explore and test vLLM's memory management system locally.
It demonstrates how to:
1. Access and inspect the BlockPool
2. Create and manipulate KVCacheBlocks
3. Simulate allocation and deallocation scenarios
4. Test fragmentation analysis logic

Run this script to experiment with the memory system before building the full analyzer.
"""

import sys
from pathlib import Path

# Add the project root to the path to allow imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def explore_block_structure():
    """Explore the KVCacheBlock data structure."""
    print("=" * 80)
    print("EXPLORING KVCacheBlock STRUCTURE")
    print("=" * 80)
    
    # Import here to avoid issues if vllm is not fully configured
    try:
        from vllm.v1.core.kv_cache_utils import KVCacheBlock
    except (ModuleNotFoundError, ImportError) as e:
        print(f"Could not import KVCacheBlock: {e}")
        print("This is expected if vLLM C++ extensions are not built.")
        print("To fix, run: pip install -e . from the vllm directory")
        return None
    
    # Create a sample block
    block = KVCacheBlock(block_id=42)
    
    print(f"\nCreated block: {block}")
    print(f"  - block_id: {block.block_id}")
    print(f"  - ref_cnt: {block.ref_cnt}")
    print(f"  - block_hash: {block.block_hash}")
    print(f"  - is_null: {block.is_null}")
    print(f"  - prev_free_block: {block.prev_free_block}")
    print(f"  - next_free_block: {block.next_free_block}")
    
    # Simulate usage
    block.ref_cnt += 1
    print(f"\nAfter incrementing ref_cnt: {block.ref_cnt}")
    
    return block


def explore_free_block_queue():
    """Explore the FreeKVCacheBlockQueue."""
    print("\n" + "=" * 80)
    print("EXPLORING FreeKVCacheBlockQueue")
    print("=" * 80)
    
    try:
        from vllm.v1.core.kv_cache_utils import KVCacheBlock, FreeKVCacheBlockQueue
    except (ModuleNotFoundError, ImportError) as e:
        print(f"Could not import: {e}")
        raise
    
    # Create a set of blocks
    num_blocks = 10
    blocks = [KVCacheBlock(block_id=i) for i in range(num_blocks)]
    
    print(f"\nCreated {num_blocks} blocks")
    
    # Create a free block queue
    free_queue = FreeKVCacheBlockQueue(blocks)
    
    print(f"Free queue size: {free_queue.num_free_blocks}")
    
    # Get all free blocks and print their connections
    all_free = free_queue.get_all_free_blocks()
    print(f"\nAll free blocks (total: {len(all_free)}):")
    for block in all_free[:5]:  # Print first 5
        next_id = block.next_free_block.block_id if block.next_free_block else None
        prev_id = block.prev_free_block.block_id if block.prev_free_block else None
        print(f"  Block {block.block_id}: prev={prev_id}, next={next_id}")
    
    # Pop some blocks
    print("\nPopping 3 blocks from the front:")
    for i in range(3):
        block = free_queue.popleft()
        print(f"  Popped block {block.block_id}")
    
    print(f"\nRemaining free blocks: {free_queue.num_free_blocks}")
    
    # Append a block back
    print("\nAppending block 0 back to the queue")
    free_queue.append(blocks[0])
    print(f"Free blocks after append: {free_queue.num_free_blocks}")
    
    return free_queue, blocks


def explore_block_pool():
    """Explore the BlockPool structure."""
    print("\n" + "=" * 80)
    print("EXPLORING BlockPool")
    print("=" * 80)
    
    try:
        from vllm.v1.core.block_pool import BlockPool
    except (ModuleNotFoundError, ImportError) as e:
        print(f"Could not import: {e}")
        raise
    
    # Create a block pool with 100 blocks
    num_gpu_blocks = 100
    enable_caching = True
    hash_block_size = 16  # Typical block size in tokens
    
    print(f"\nCreating BlockPool:")
    print(f"  - num_gpu_blocks: {num_gpu_blocks}")
    print(f"  - enable_caching: {enable_caching}")
    print(f"  - hash_block_size: {hash_block_size}")
    
    block_pool = BlockPool(
        num_gpu_blocks=num_gpu_blocks,
        enable_caching=enable_caching,
        hash_block_size=hash_block_size,
    )
    
    print(f"\nBlockPool created successfully!")
    print(f"  - Total blocks: {block_pool.num_gpu_blocks}")
    print(f"  - Free blocks: {block_pool.get_num_free_blocks()}")
    print(f"  - Usage: {block_pool.get_usage():.2%}")
    print(f"  - Null block ID: {block_pool.null_block.block_id}")
    
    # Allocate some blocks
    print("\n--- Allocating blocks ---")
    num_to_allocate = 20
    print(f"Allocating {num_to_allocate} blocks...")
    
    try:
        allocated_blocks = block_pool.get_new_blocks(num_to_allocate)
        print(f"Successfully allocated {len(allocated_blocks)} blocks")
        print(f"  Block IDs: {[b.block_id for b in allocated_blocks[:5]]}... (first 5)")
        print(f"  Free blocks remaining: {block_pool.get_num_free_blocks()}")
        print(f"  Usage: {block_pool.get_usage():.2%}")
        
        # Check reference counts
        print(f"\n  Reference counts:")
        for block in allocated_blocks[:3]:
            print(f"    Block {block.block_id}: ref_cnt={block.ref_cnt}")
        
        # Free some blocks
        print("\n--- Freeing blocks ---")
        blocks_to_free = allocated_blocks[:10]
        print(f"Freeing {len(blocks_to_free)} blocks...")
        block_pool.free_blocks(blocks_to_free)
        print(f"  Free blocks after freeing: {block_pool.get_num_free_blocks()}")
        print(f"  Usage: {block_pool.get_usage():.2%}")
        
    except Exception as e:
        print(f"Error during allocation: {e}")
    
    return block_pool


def simulate_fragmentation_scenario():
    """Simulate a scenario that creates memory fragmentation."""
    print("\n" + "=" * 80)
    print("SIMULATING FRAGMENTATION SCENARIO")
    print("=" * 80)
    
    try:
        from vllm.v1.core.block_pool import BlockPool
    except (ModuleNotFoundError, ImportError) as e:
        print(f"Could not import: {e}")
        raise
    
    block_pool = BlockPool(
        num_gpu_blocks=50,
        enable_caching=True,
        hash_block_size=16,
    )
    
    print(f"\nInitial state:")
    print(f"  Free blocks: {block_pool.get_num_free_blocks()}")
    print(f"  Usage: {block_pool.get_usage():.2%}")
    
    # Simulate multiple requests allocating and freeing blocks
    print("\n--- Simulating 3 requests with different lifetimes ---")
    
    # Request 1: Allocate 10 blocks
    print("\nRequest 1: Allocating 10 blocks")
    req1_blocks = block_pool.get_new_blocks(10)
    print(f"  Free blocks: {block_pool.get_num_free_blocks()}")
    
    # Request 2: Allocate 15 blocks
    print("\nRequest 2: Allocating 15 blocks")
    req2_blocks = block_pool.get_new_blocks(15)
    print(f"  Free blocks: {block_pool.get_num_free_blocks()}")
    
    # Request 3: Allocate 8 blocks
    print("\nRequest 3: Allocating 8 blocks")
    req3_blocks = block_pool.get_new_blocks(8)
    print(f"  Free blocks: {block_pool.get_num_free_blocks()}")
    print(f"  Usage: {block_pool.get_usage():.2%}")
    
    # Free Request 2 (middle request) - creates fragmentation
    print("\nRequest 2 completes: Freeing 15 blocks (creates fragmentation)")
    block_pool.free_blocks(req2_blocks)
    print(f"  Free blocks: {block_pool.get_num_free_blocks()}")
    print(f"  Usage: {block_pool.get_usage():.2%}")
    
    # Analyze the free block queue
    print("\n--- Analyzing free block distribution ---")
    free_blocks = block_pool.free_block_queue.get_all_free_blocks()
    free_block_ids = sorted([b.block_id for b in free_blocks])
    
    print(f"  Free block IDs: {free_block_ids}")
    
    # Find runs of contiguous free blocks
    runs = []
    if free_block_ids:
        current_run = [free_block_ids[0]]
        for block_id in free_block_ids[1:]:
            if block_id == current_run[-1] + 1:
                current_run.append(block_id)
            else:
                runs.append(current_run)
                current_run = [block_id]
        runs.append(current_run)
    
    print(f"\n  Contiguous free block runs: {len(runs)}")
    for i, run in enumerate(runs):
        print(f"    Run {i+1}: {len(run)} blocks (IDs {run[0]}-{run[-1]})")
    
    # Calculate fragmentation metrics
    if runs:
        largest_run = max(len(run) for run in runs)
        average_run = sum(len(run) for run in runs) / len(runs)
        
        print(f"\n  Fragmentation metrics:")
        print(f"    Number of runs: {len(runs)}")
        print(f"    Largest run: {largest_run} blocks")
        print(f"    Average run size: {average_run:.2f} blocks")
        print(f"    Total free blocks: {len(free_block_ids)}")
        
        # External fragmentation ratio (1 - largest_run / total_free)
        external_frag = 1 - (largest_run / len(free_block_ids)) if free_block_ids else 0
        print(f"    External fragmentation ratio: {external_frag:.2%}")


def analyze_block_hash_to_block_map():
    """Explore the BlockHashToBlockMap used for prefix caching."""
    print("\n" + "=" * 80)
    print("EXPLORING BlockHashToBlockMap (Prefix Cache)")
    print("=" * 80)
    
    try:
        from vllm.v1.core.block_pool import BlockHashToBlockMap
        from vllm.v1.core.kv_cache_utils import KVCacheBlock, make_block_hash_with_group_id, BlockHash
    except (ModuleNotFoundError, ImportError) as e:
        print(f"Could not import: {e}")
        raise
    
    cache = BlockHashToBlockMap()
    print(f"\nCreated empty cache, size: {len(cache)}")
    
    # Create some blocks and simulate caching
    blocks = [KVCacheBlock(block_id=i) for i in range(5)]
    
    # Create dummy block hashes (in practice, these are computed from token IDs)
    dummy_hashes = [BlockHash(f"hash_{i}".encode()) for i in range(5)]
    
    print("\nInserting blocks into cache:")
    for i, (block, hash_val) in enumerate(zip(blocks, dummy_hashes)):
        key = make_block_hash_with_group_id(hash_val, group_id=0)
        cache.insert(key, block)
        print(f"  Inserted block {block.block_id} with hash key")
    
    print(f"\nCache size after insertions: {len(cache)}")
    
    # Retrieve a block
    print("\nRetrieving a block:")
    key = make_block_hash_with_group_id(dummy_hashes[0], group_id=0)
    retrieved = cache.get_one_block(key)
    if retrieved:
        print(f"  Retrieved block {retrieved.block_id}")
    
    # Test duplicate block hashes (same hash, different block_id)
    print("\nTesting duplicate blocks (same hash, different block_id):")
    duplicate_block = KVCacheBlock(block_id=100)
    cache.insert(key, duplicate_block)
    print(f"  Inserted duplicate block {duplicate_block.block_id}")
    print(f"  Cache size: {len(cache)} (should be same as before)")
    
    # Pop a block
    print("\nPopping a block from cache:")
    popped = cache.pop(key, blocks[0].block_id)
    if popped:
        print(f"  Popped block {popped.block_id}")
    print(f"  Cache size: {len(cache)}")


def check_cuda_status():
    """Check whether PyTorch sees CUDA in the current environment."""
    print("\n" + "=" * 80)
    print("CHECKING CUDA (PyTorch)")
    print("=" * 80)

    try:
        import torch
    except (ModuleNotFoundError, ImportError) as e:
        print(f"Could not import torch: {e}")
        print("Install a CUDA-enabled PyTorch build to use GPU.")
        return False

    cuda_available = torch.cuda.is_available()
    cuda_version = torch.version.cuda
    print(f"torch.cuda.is_available(): {cuda_available}")
    print(f"torch.version.cuda: {cuda_version}")
    if cuda_available:
        try:
            print(f"GPU count: {torch.cuda.device_count()}")
            print(f"GPU 0: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            print(f"Could not query GPU details: {e}")
    return cuda_available


def main():
    """Main testing function."""
    print("\n" * 2)
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "vLLM Memory Fragmentation Analyzer" + " " * 24 + "║")
    print("║" + " " * 25 + "Testing and Exploration" + " " * 30 + "║")
    print("╚" + "=" * 78 + "╝")
    
    print("\nAttempting to import vLLM modules...")
    print("(If imports fail, this is expected if vLLM C++ extensions are not built)\n")
    
    results = []
    
    try:
        result = explore_block_structure()
        if result is not None:
            results.append("✓ Block structure exploration completed")
    except Exception as e:
        print(f"Block structure exploration failed: {e}\n")

    try:
        cuda_ok = check_cuda_status()
        if cuda_ok:
            results.append("✓ CUDA available in PyTorch")
        else:
            results.append("⚠ CUDA not available in PyTorch")
    except Exception as e:
        print(f"CUDA check failed: {e}\n")
    
    try:
        explore_free_block_queue()
        results.append("✓ Free block queue exploration completed")
    except (ModuleNotFoundError, ImportError) as e:
        print(f"Free block queue exploration skipped: {e}\n")
    except Exception as e:
        print(f"Free block queue exploration failed: {e}\n")
    
    try:
        explore_block_pool()
        results.append("✓ Block pool exploration completed")
    except (ModuleNotFoundError, ImportError) as e:
        print(f"Block pool exploration skipped: {e}\n")
    except Exception as e:
        print(f"Block pool exploration failed: {e}\n")
    
    try:
        analyze_block_hash_to_block_map()
        results.append("✓ Hash map analysis completed")
    except (ModuleNotFoundError, ImportError) as e:
        print(f"Hash map analysis skipped: {e}\n")
    except Exception as e:
        print(f"Hash map analysis failed: {e}\n")
    
    try:
        simulate_fragmentation_scenario()
        results.append("✓ Fragmentation scenario simulation completed")
    except (ModuleNotFoundError, ImportError) as e:
        print(f"Fragmentation simulation skipped: {e}\n")
    except Exception as e:
        print(f"Fragmentation simulation failed: {e}\n")
    
    print("\n" + "=" * 80)
    print("TESTING SUMMARY")
    print("=" * 80)
    
    if results:
        print(f"\n✓ Completed explorations: {len(results)}")
        for result in results:
            print(f"  {result}")
        print("\nYou can now use this knowledge to build the fragmentation analyzer.")
        print("\nNext steps:")
        print("  1. Review the ANALYZER_DESIGN.md document")
        print("  2. Implement FragmentationSnapshot class")
        print("  3. Implement FragmentationTracker with hooks")
        print("  4. Build visualization and reporting tools")
    else:
        print("\n⚠ Could not run tests due to missing vLLM C++ extensions")
        print("\nTo fix this, run one of the following:")
        print("  1. pip install -e . (from vllm directory)")
        print("  2. pip install torch vllm (if not already installed)")
        print("  3. Check that CUDA/ROCm is properly configured")
        print("\nAfter installing, run this script again.")


if __name__ == "__main__":
    main()
