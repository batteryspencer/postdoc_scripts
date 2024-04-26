import matplotlib.pyplot as plt
import numpy as np

# Data
md_steps = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 
            5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000]
activation_barriers = [0.9, 0.86, 0.92, 0.98, 0.98, 0.99, 1.0, 1.0, 0.99, 0.98,
                       0.98, 0.99, 0.98, 0.98, 0.97, 0.97, 0.97, 0.96, 0.96, 0.96]
std_dev = [0.2, 0.16, 0.15, 0.14, 0.14, 0.14, 0.14, 0.14, 0.15, 0.15,
           0.14, 0.14, 0.14, 0.14, 0.14, 0.15, 0.15, 0.15, 0.15, 0.15]

# Calculate absolute differences
differences = np.abs(np.diff(activation_barriers))
window_size = 5
smoothed_differences = np.convolve(differences, np.ones(window_size)/window_size, mode='valid')

# Determine threshold based on standard deviation
std_dev_threshold = np.std(smoothed_differences) * 0.5  # Using half of the standard deviation

# Find stabilization index
stabilization_index = np.where(smoothed_differences < std_dev_threshold)[0][0] + window_size  # Adjust for the window size

# Compute the asymptotic value from the stabilization point
asymptotic_value = np.mean(activation_barriers[stabilization_index:])

# Plotting
plt.figure(figsize=(10, 6))
plt.errorbar(md_steps, activation_barriers, yerr=std_dev, fmt='-o', color='black', ecolor='black', capsize=3.5)
plt.axhline(y=asymptotic_value, color='red', linestyle='--', label=f'Asymptote: {asymptotic_value:.2f}')
plt.xlabel('MD Steps', fontsize=12)
plt.ylabel('Activation Barrier (eV)', fontsize=12)
plt.savefig('md_force_trend.png', dpi=300, bbox_inches='tight')

print(f"\nSimulation reached stabilization at {md_steps[stabilization_index]} steps with an activation barrier of {asymptotic_value:.2f} eV.")
