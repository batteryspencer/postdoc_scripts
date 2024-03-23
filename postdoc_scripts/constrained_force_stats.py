import numpy as np
import matplotlib.pyplot as plt
from glob import glob

def get_file_line_count(filename):
    with open(filename) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

# Modified function to include max_steps parameter
def read_simulation_data(folder_name, max_steps=None):
    lambda_values = []
    force_values_on_constrained_bond = []
    md_steps = 0
    
    with open(f'./{folder_name}/REPORT') as file:
        for line in file:
            if "b_m" in line:
                lambda_values.append(float(line.split()[1]))
            if "cc>" in line:
                try:
                    force_values_on_constrained_bond.append(float(line.split()[2]))
                except ValueError:
                    print('Error parsing collective variable value')
            if "MD step No." in line:
                md_steps += 1
                # Break the loop if the number of steps reaches max_steps
                if max_steps is not None and md_steps >= max_steps:
                    break
    
    return lambda_values, force_values_on_constrained_bond, md_steps

def calculate_statistics(lambda_values_per_cv):
    mean_force = np.mean(lambda_values_per_cv)
    standard_deviation = np.std(lambda_values_per_cv)
    return mean_force, standard_deviation

def cumulative_force_analysis(force_values):
    cumulative_means = []
    cumulative_stds = []
    cumulative_intervals = []

    total_number = len(force_values)
    if total_number >= 500:
        for i in range(500, total_number + 1, 500):
            cumulative_intervals.append(i)
            cumulative_means.append(np.mean(force_values[:i]))
            cumulative_stds.append(np.std(force_values[:i]))

    return cumulative_intervals, cumulative_means, cumulative_stds

def main():
    constraint_index = 0  # Specify the index of the constraint of interest
    num_constraints = get_file_line_count('ICONST')
    total_md_steps = 0  # Initialize total MD steps accumulator
    max_steps = None  # Specify the maximum number of steps to consider for analysis

    with open("INCAR", "r") as file:
        time_step = float(next((line.split('=')[1].strip() for line in file if "POTIM" in line), None))

    folders = sorted(glob('RUN_*'), key=lambda x: int(x.split('_')[1])) + ['.']
    
    print(f'Integrating over Reaction Coordinate Index: {constraint_index}, with a total of {num_constraints} constraints')
    
    all_lambda_values = []
    all_cv_values = []
    
    for folder in folders:
        lambda_values, cv_values, md_steps = read_simulation_data(folder, max_steps=max_steps)
        all_lambda_values.extend(lambda_values)
        all_cv_values.extend(cv_values)
        total_md_steps += md_steps  # Accumulate total MD steps here
    
    lambda_values_per_cv = all_lambda_values[constraint_index::num_constraints]
    mean_force, std_dev = calculate_statistics(lambda_values_per_cv)
    
    with open('force_stats_report.txt', 'w') as output_file:
        output_file.write('CV, Mean Force, Standard Deviation\n')
        output_file.write(f'{all_cv_values[0]:.2f}, {mean_force:.2f}, {std_dev:.2f}\n')
        output_file.write(f'MD steps = {total_md_steps}')  # Use the accumulated total MD steps
    
    print(f'The mean of the forces is: {mean_force:.2f} with a standard deviation of {std_dev:.2f}, sampled over {total_md_steps * time_step:.1f} fs using a {time_step:.1f} fs time step.')
    
    cumulative_intervals, cumulative_means, cumulative_stds = cumulative_force_analysis(lambda_values_per_cv)

    plt.figure()
    plt.errorbar(cumulative_intervals, cumulative_means, yerr=cumulative_stds, fmt='o-', label='Cumulative Mean Force')
    plt.xlabel('Simulation Interval (Number of Data Points)')
    plt.ylabel('Force (Arbitrary Units)')
    plt.title('Cumulative Analysis of Mean Force Over Simulation Intervals')
    plt.legend()
    plt.savefig('cumulative_force_stats.png', dpi=100)  # Updated file name for clarity

if __name__ == "__main__":
    main()

