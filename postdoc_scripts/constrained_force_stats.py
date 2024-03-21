def get_file_line_count(filename):
    with open(filename) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def read_simulation_data(folder_name):
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
    
    return lambda_values, force_values_on_constrained_bond, md_steps

def main():
    constraint_index = 0  # Specify the index of the constraint of interest
    num_constraints = get_file_line_count('ICONST')
    total_md_steps = 0  # Initialize total MD steps accumulator

    folders = sorted(glob('RUN_*'), key=lambda x: int(x.split('_')[1])) + ['.']
    
    print(f'Integrating over Reaction Coordinate Index: {constraint_index}, with a total of {num_constraints} constraints')
    
    all_lambda_values = []
    all_cv_values = []
    
    for folder in folders:
        lambda_values, cv_values, md_steps = read_simulation_data(folder)
        all_lambda_values.extend(lambda_values)
        all_cv_values.extend(cv_values)
        total_md_steps += md_steps  # Accumulate total MD steps here
if __name__ == "__main__":
    main()
