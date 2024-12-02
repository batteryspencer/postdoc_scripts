import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.interpolate import interp1d
from scipy.integrate import trapz

# Define font sizes and tick parameters as constants
LABEL_FONTSIZE = 18
TITLE_FONTSIZE = 22
TICK_LABELSIZE = 14
LEGEND_FONTSIZE = 14
TICK_LENGTH_MAJOR = 8
TICK_WIDTH_MAJOR = 1

# This function reads the force_stats_report.txt and extracts the values
def read_force_stats(file_path, target_steps=None):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        stats = {}
        
        # Parse initial values
        if len(lines) > 2:
            stats['CV'] = float(lines[1].split(':')[1].strip())
            if target_steps is None:
                stats['Mean Force'] = -1 * np.around(float(lines[2].split(':')[1].strip()), 2)
                stats['Standard Deviation'] = float(lines[3].split(':')[1].strip())
                stats['MD steps'] = int(lines[4].split(':')[1].strip())

            # Parse cumulative analysis results
            for line in lines[6:]:  # Assuming the Cumulative Analysis starts at line 7
                parts = line.split()
                if len(parts) == 3 and int(parts[0]) == target_steps:
                    stats['Mean Force'] = -1 * float(parts[1])
                    stats['Standard Deviation'] = float(parts[2])
                    stats['MD steps'] = target_steps

        return stats

def interpolate_and_find_roots(x, y, num_points=500):
    """
    Interpolates the given x, y data and finds zero crossings.

    Args:
    - x: Array of constrained bond lengths.
    - y: Array of mean forces.
    - num_points: Number of points for fine interpolation.

    Returns:
    - interp_func: The interpolation function.
    - fine_x: Fine-grained x-values for plotting the interpolated curve.
    - fine_y: Corresponding interpolated y-values for fine_x.
    - roots: Zero crossings (roots) of the interpolated curve.
    """
    # Use cubic interpolation for smoothness
    interp_func = interp1d(x, y, kind="cubic", fill_value="extrapolate")
    fine_x = np.linspace(min(x), max(x), num_points)
    fine_y = interp_func(fine_x)

    # Find zero crossings
    zero_crossings = np.where(np.diff(np.sign(fine_y)))[0]
    roots = [fine_x[zc] for zc in zero_crossings]

    return interp_func, fine_x, fine_y, roots

def compute_slope(interp_func, x_point, h=1e-6):
    """
    Computes the slope at a specific point using the central difference formula.

    Args:
    - interp_func: Interpolated function (callable).
    - x_point: The x-value where the slope is to be computed.
    - h: Small step size for finite difference (default: 1e-6).

    Returns:
    - float: The slope at x_point.
    """
    return (interp_func(x_point + h) - interp_func(x_point - h)) / (2 * h)

def compute_barriers_and_states(fine_x, fine_y, interp_func, roots):
    """
    Computes forward and reverse barriers and classifies states based on roots.

    Args:
    - fine_x: Fine-grained x-values for the interpolated curve.
    - fine_y: Corresponding y-values for fine_x.
    - interp_func: The interpolation function.
    - roots: Zero crossings of the interpolated curve.

    Returns:
    - results: A dictionary containing roots, state types, forward/reverse barriers, and free energy change.
    """
    state_types = []
    forward_barrier, reverse_barrier = 0, 0

    if len(roots) == 1:
        root = roots[0]
        root_index = np.argmin(np.abs(fine_x - root))
        slope = compute_slope(interp_func, root)

        if slope > 0:
            state_types = ["Transition State"]
            forward_barrier = abs(trapz(fine_y[:root_index + 1], fine_x[:root_index + 1]))
            reverse_barrier = abs(trapz(fine_y[root_index:], fine_x[root_index:]))
        else:
            # Check if it is initial or final state
            points_after = len(fine_x[fine_x > root])
            points_before = len(fine_x[fine_x < root])
            MIN_POINTS_TO_DETERMINE_STATE = 3
            if points_after >= MIN_POINTS_TO_DETERMINE_STATE:
                state_types = ["Initial State"]
                forward_barrier = abs(trapz(fine_y[root_index:], fine_x[root_index:]))
            elif points_before >= MIN_POINTS_TO_DETERMINE_STATE:
                state_types = ["Final State"]
                reverse_barrier = abs(trapz(fine_y[:root_index], fine_x[:root_index]))
    elif len(roots) == 2:
        root_index1 = np.argmin(np.abs(fine_x - roots[0]))
        root_index2 = np.argmin(np.abs(fine_x - roots[1]))
        slope1 = compute_slope(interp_func, roots[0])
        slope2 = compute_slope(interp_func, roots[1])

        if slope1 < 0 and slope2 > 0:
            state_types = ["Initial State", "Transition State"]
            forward_barrier = abs(trapz(fine_y[root_index1:root_index2 + 1], fine_x[root_index1:root_index2 + 1]))
        elif slope1 > 0 and slope2 < 0:
            state_types = ["Transition State", "Final State"]
            reverse_barrier = abs(trapz(fine_y[root_index1:root_index2 + 1], fine_x[root_index1:root_index2 + 1]))
    elif len(roots) == 3:
        root_index1 = np.argmin(np.abs(fine_x - roots[0]))
        root_index2 = np.argmin(np.abs(fine_x - roots[1]))
        root_index3 = np.argmin(np.abs(fine_x - roots[2]))
        state_types = ["Initial State", "Transition State", "Final State"]
        forward_barrier = abs(trapz(fine_y[root_index1:root_index2 + 1], fine_x[root_index1:root_index2 + 1]))
        reverse_barrier = abs(trapz(fine_y[root_index2:root_index3 + 1], fine_x[root_index2:root_index3 + 1]))

    free_energy_change = forward_barrier + reverse_barrier

    results = {
        "roots": roots,
        "state_types": state_types,
        "forward_barrier": forward_barrier,
        "reverse_barrier": reverse_barrier,
        "free_energy_change": free_energy_change,
    }

    return results

def calculate_barriers(x, y):
    """
    High-level function to calculate barriers and provide interpolation data.

    Args:
    - x: Array of constrained bond lengths.
    - y: Array of mean forces.

    Returns:
    - results: A dictionary containing roots, state types, forward/reverse barriers, and free energy change.
    - fine_x: Fine-grained x-values for plotting the interpolated curve.
    - fine_y: Corresponding interpolated y-values for fine_x.
    """
    interp_func, fine_x, fine_y, roots = interpolate_and_find_roots(x, y)
    results = compute_barriers_and_states(fine_x, fine_y, interp_func, roots)
    return results, fine_x, fine_y

# for target_steps in np.arange(500, 10500, 500):
for target_steps in [None]:
    # This dictionary will hold our data
    data = {'Constrained_Bond_Length (Å)' : [], 'Mean_Force (eV/Å)' : [], 'Standard_Deviation (eV/Å)' : [], 'MD_Steps': []}

    # Assuming your directories are named in the '1.06_793' format and are in the current working directory
    for folder in glob.glob("[0-9].[0-9][0-9]_*"):
        file_path = os.path.join(folder, 'force_stats_report.txt')
        if os.path.isfile(file_path):
            stats = read_force_stats(file_path, target_steps=target_steps)
            data['Constrained_Bond_Length (Å)' ].append(stats['CV'])
            data['Mean_Force (eV/Å)' ].append(stats['Mean Force'])
            data['Standard_Deviation (eV/Å)' ].append(stats['Standard Deviation'])
            data['MD_Steps'].append(stats['MD steps'])

    # Create a DataFrame from the data
    df = pd.DataFrame(data)

    # Sort the DataFrame based on the constrained bond length
    df = df.sort_values(by=['Constrained_Bond_Length (Å)' ])

    # Assuming 'df' is the DataFrame with your data sorted by 'Constrained_Bond_Length (Å)' 
    x = df['Constrained_Bond_Length (Å)' ].to_numpy()
    y = df['Mean_Force (eV/Å)' ].to_numpy()
    std_dev = df['Standard_Deviation (eV/Å)' ].to_numpy()

    # Calculate barriers and interpolation for original data
    results, fine_x, fine_y = calculate_barriers(x, y)

    # Calculate barriers for upper and lower limits
    results_upper, _, _ = calculate_barriers(x, y + std_dev)
    results_lower, _, _ = calculate_barriers(x, y - std_dev)

    # Compute standard deviations as half the difference between upper and lower estimates
    forward_barrier_std = abs(results_upper["forward_barrier"] - results_lower["forward_barrier"]) / 2
    reverse_barrier_std = abs(results_upper["reverse_barrier"] - results_lower["reverse_barrier"]) / 2

    results_string = 'Activation Barriers (Area under the curve):\n'
    if 'forward_barrier' in results:
        results_string += f"Forward Barrier: {results['forward_barrier']:.2f} ± {forward_barrier_std:.2f} eV\n"
    if 'reverse_barrier' in results:
        results_string += f"Reverse Barrier: {results['reverse_barrier']:.2f} ± {reverse_barrier_std:.2f} eV\n"
    if len(results['roots']) >= 1:
        results_string += "\nEquilibrium Bond Distances: \n"
        for i, state in enumerate(results['state_types']):
            results_string += f"{state}: {results['roots'][i]:.3f} Å\n"
    else:
        results_string += "No zero crossings found."

# Print data in a table format and save it to a text file
table_string = df.to_string(index=False)
print(table_string + '\n')
print(results_string)
with open("pmf_analysis_results.txt", "w") as text_file:
    text_file.write(table_string + '\n\n')
    text_file.write(results_string + '\n')

# Plotting
plt.figure(figsize=(10, 6))
ax = plt.gca()
plt.errorbar(df['Constrained_Bond_Length (Å)' ], df['Mean_Force (eV/Å)' ], yerr=df['Standard_Deviation (eV/Å)' ], fmt='o', color='black', ecolor='black', capsize=3.5)
# plt.plot(df['Constrained_Bond_Length (Å)' ], df['Mean_Force (eV/Å)' ] + df['Standard_Deviation (eV/Å)' ], linestyle='--', color='black', alpha=0.5)
# plt.plot(df['Constrained_Bond_Length (Å)' ], df['Mean_Force (eV/Å)' ] - df['Standard_Deviation (eV/Å)' ], linestyle='--', color='black', alpha=0.5)

# Create a polygon to fill the area under the curve
verts = [(df['Constrained_Bond_Length (Å)' ].iloc[0], 0)] + list(zip(df['Constrained_Bond_Length (Å)' ], df['Mean_Force (eV/Å)' ])) + [(df['Constrained_Bond_Length (Å)' ].iloc[-1], 0)]
poly = Polygon(verts, facecolor='0.9', edgecolor='0.1')
ax.add_patch(poly)

plt.title('Mean Force vs. Constrained Bond Length', fontsize=TITLE_FONTSIZE)
plt.xlabel('Constrained Bond Length (Å)', fontsize=LABEL_FONTSIZE)
plt.ylabel('Mean Force (eV/Å)', fontsize=LABEL_FONTSIZE)
plt.tick_params(axis='both', which='major', labelsize=TICK_LABELSIZE, length=TICK_LENGTH_MAJOR, width=TICK_WIDTH_MAJOR)
plt.savefig('mean_force_plot.png', dpi=300, bbox_inches='tight')
