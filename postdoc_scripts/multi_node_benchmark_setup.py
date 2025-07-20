import os
import shutil
import subprocess

# List of node counts to benchmark
NODE_CHOICES = [1, 2, 4]

# Files to copy into each node-specific directory
def get_base_files():
    return [f for f in os.listdir('.')
            if os.path.isfile(f) and f in (
                'POSCAR', 'POTCAR', 'INCAR', 'KPOINTS',
                'submit.sh', 'vasp_benchmark_setup.py', 'analyze_benchmarks.py'
            )]


def patch_file(path, marker, new_line):
    # Replace a line starting with marker with new_line
    lines = []
    with open(path, 'r') as fh:
        for l in fh:
            if l.strip().startswith(marker):
                lines.append(new_line)
            else:
                lines.append(l)
    with open(path, 'w') as fh:
        fh.writelines(lines)


def main():
    cwd = os.getcwd()
    base_files = get_base_files()

    for nodes in NODE_CHOICES:
        dir_name = f'nodes={nodes}'
        print(f"Setting up benchmarking for {nodes} node(s) in {dir_name}/")
        # Create directory
        os.makedirs(dir_name, exist_ok=True)
        # Copy base files
        for fname in base_files:
            shutil.copy(fname, os.path.join(dir_name, fname))

        # Patch vasp_benchmark_setup.py to use correct node count
        bench_script = os.path.join(dir_name, 'vasp_benchmark_setup.py')
        patch_file(
            bench_script,
            'num_nodes =',
            f'num_nodes = {nodes}\n'
        )

        # Patch submit.sh to request correct node count
        submit_sh = os.path.join(dir_name, 'submit.sh')
        patch_file(
            submit_sh,
            '#SBATCH --nodes=',
            f'#SBATCH --nodes={nodes}\n'
        )

        # Run the benchmark setup in that directory
        subprocess.run(['python', 'vasp_benchmark_setup.py'], cwd=dir_name, check=True)
        print(f"Generated configurations in {dir_name}/benchmark_folders")

    print("\nAll node-level setups generated. You can now submit jobs in each folder and run analyze_benchmarks.py similarly.")


if __name__ == '__main__':
    main()
