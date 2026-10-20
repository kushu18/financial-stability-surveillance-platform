import time
import os
import sys

# Add project root to sys.path if not present
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import RealDataLoader
from src.simulation.montecarlo import MonteCarloStressTester

def run_benchmark():
    print("Initializing DataLoader (loading CSVs)...")
    loader = RealDataLoader()
    # Try a known date
    print("Fetching snapshots for 2020-03-24...")
    try:
        snapshots = loader.get_daily_snapshot("2020-03-24")
        if not snapshots:
            # If 2020-03-24 doesn't exist, just get the first available date
            date_keys = list(loader._grouped.groups.keys())
            snapshots = loader.get_daily_snapshot(date_keys[len(date_keys)//2])
    except Exception as e:
        print(f"Error fetching snapshot: {e}")
        return

    print(f"Loaded {len(snapshots)} bank snapshots. Preparing Monte Carlo engine...")

    mc = MonteCarloStressTester(n_runs=1000)

    # Warm-up run (100 runs) to compile JIT/cache if any
    print("Running 100-run warm-up...")
    mc._n_runs = 100
    mc._run_sync(snapshots, shock_bank="SBI", magnitude=0.75, shock_type="Benchmark")

    # Benchmark run
    mc._n_runs = 1000
    print("Running official 1000-run benchmark...")
    
    start_time = time.time()
    mc._run_sync(snapshots, shock_bank="SBI", magnitude=0.75, shock_type="Benchmark")
    end_time = time.time()

    elapsed = end_time - start_time
    print("-" * 50)
    print(f"BENCHMARK RESULT:")
    print(f"Time taken for 1,000 Monte Carlo permutations: {elapsed:.3f} seconds")
    print("-" * 50)

if __name__ == "__main__":
    run_benchmark()
