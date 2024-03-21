def get_file_line_count(filename):
    with open(filename) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def main():
    constraint_index = 0  # Specify the index of the constraint of interest
    num_constraints = get_file_line_count('ICONST')
    total_md_steps = 0  # Initialize total MD steps accumulator

if __name__ == "__main__":
    main()
