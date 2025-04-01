import math

def find_best_distribution(benchmark_data, total_jobs, total_nodes):
    """
    benchmark_data: list of dicts
        Each dict must have: {'nodes': int, 'NCORE': ..., 'NPAR': ..., 'KPAR': ..., 'loop_time': float}
    total_jobs: int
        How many total calculations to run
    total_nodes: int
        How many nodes total are available
    
    Returns a dict describing the best config and the estimated total time.
    """
    # Group data by node-count, picking the config with the smallest loop_time
    node_count_to_best = {}
    for entry in benchmark_data:
        n = entry['nodes']
        if n not in node_count_to_best or entry['loop_time'] < node_count_to_best[n]['loop_time']:
            node_count_to_best[n] = entry
    
    best_config_overall = None
    best_total_time = float('inf')
    
    # Try each node-count's best config
    for n, config in node_count_to_best.items():
        if n <= total_nodes:
            # How many jobs can run in parallel?
            parallel_jobs = total_nodes // n
            # Number of waves (batches)
            waves = math.ceil(total_jobs / parallel_jobs)
            # Total time = waves * loop_time for that config
            total_time = waves * config['loop_time']
            
            if total_time < best_total_time:
                best_total_time = total_time
                best_config_overall = config.copy()
                best_config_overall['waves'] = waves
                best_config_overall['parallel_jobs'] = parallel_jobs
                best_config_overall['total_time'] = total_time
    
    return best_config_overall

# Example usage: fill benchmark_data with your table
if __name__ == '__main__':
    # Example data from your table (partial snippet)
    benchmark_data = [
        {'nodes': 1, 'NCORE': 32, 'NPAR': 4, 'KPAR': 1, 'loop_time': 42.9},
        {'nodes': 2, 'NCORE': 32, 'NPAR': 4, 'KPAR': 2, 'loop_time': 26.4},
        {'nodes': 5, 'NCORE': 16, 'NPAR': 8, 'KPAR': 5, 'loop_time': 14.6},
        # ... fill in all rows ...
    ]
    
    total_jobs = 30
    total_nodes = 15
    
    best_config = find_best_distribution(benchmark_data, total_jobs, total_nodes)
    
    if best_config:
        print("BEST CONFIG:")
        print("Nodes:", best_config['nodes'])
        print("NCORE:", best_config['NCORE'])
        print("NPAR:", best_config['NPAR'])
        print("KPAR:", best_config['KPAR'])
        print("Parallel jobs:", best_config['parallel_jobs'])
        print("Waves:", best_config['waves'])
        print("Total time:", best_config['total_time'])
    else:
        print("No feasible config found.")
