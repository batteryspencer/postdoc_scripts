from ase.io import read
import numpy as np

# Load the structure from the POSCAR file
atoms = read('POSCAR')

# Identify Pt atoms and water molecules (assuming O and H are water constituents)
pt_atoms = [atom for atom in atoms if atom.symbol == 'Pt']
oxygen_atoms = [atom for atom in atoms if atom.symbol == 'O']

# Define buffer_gap and calculate z_min and z_max based on Pt atoms and the buffer
buffer_gap = 2.5  # 2-4 Å gap above and below the Pt slab is reasonable
print(f"Buffer gap: {buffer_gap} Å")

# Calculate z_min and z_max based on Pt atoms
z_min_pt = min([atom.position[2] for atom in pt_atoms])
z_max_pt = max([atom.position[2] for atom in pt_atoms])

# Define z_min and z_max for water below the Pt slab
z_min_water_below = min([atom.position[2] for atom in oxygen_atoms if atom.position[2] < z_min_pt])
z_max_water_below = max([atom.position[2] for atom in oxygen_atoms if atom.position[2] < z_min_pt])

# Define z_min and z_max for water above the Pt slab
z_min_water_above = min([atom.position[2] for atom in oxygen_atoms if atom.position[2] > z_max_pt])
z_max_water_above = max([atom.position[2] for atom in oxygen_atoms if atom.position[2] > z_max_pt])

# Count water molecules below the Pt slab
num_water_molecules_below = len([atom for atom in oxygen_atoms if z_min_water_below <= atom.position[2] <= z_max_water_below])

# Count water molecules above the Pt slab
num_water_molecules_above = len([atom for atom in oxygen_atoms if z_min_water_above <= atom.position[2] <= z_max_water_above])

# If needed, you can combine these for a total count, or keep them separate for individual density calculations
num_water_molecules_total = num_water_molecules_below + num_water_molecules_above

# Output the number of water molecules for clarity
print(f"Number of water molecules below the Pt slab: {num_water_molecules_below}")
print(f"Number of water molecules above the Pt slab: {num_water_molecules_above}")
print(f"Total number of water molecules around the Pt slab: {num_water_molecules_total}")

# Calculate the volume of the region
cell = atoms.get_cell()
area = np.linalg.norm(np.cross(cell[0], cell[1]))
totalz = cell[2][2]
effective_height = totalz - (z_max_pt - z_min_pt) - 2 * buffer_gap
effective_liquid_phase_volume = area * effective_height * 1e-24  # Convert Å^3 to cm^3
print(f"Total height of the cell: {totalz:.1f} Å")
print(f"Effective height for the liquid phase: {effective_height:.1f} Å")

# Calculate the water density
water_mass = num_water_molecules_total * 18 / 6.022e23  # mass in grams

# Include propylene in the density calculation
num_propylene_molecules = 1  # User-defined number of propylene molecules
propylene_mass = num_propylene_molecules * 42.08 / 6.022e23  # mass in grams

# Total mass of water and propylene
total_mass = water_mass + propylene_mass

# Calculate the densities
water_density = water_mass / effective_liquid_phase_volume  # g/cm^3
total_density = total_mass / effective_liquid_phase_volume  # g/cm^3

# Print the densities
print(f"Water density (without propylene): {water_density:.2f} g/cm^3")
print(f"Total liquid phase density (water + {num_propylene_molecules} propylene molecule): {total_density:.2f} g/cm^3")

# Calculate the required effective height to achieve a water density of 1 g/cm³
desired_density = 1.0  # g/cm³
required_volume = water_mass / desired_density  # cm^3
required_effective_height = required_volume / (area * 1e-24)  # Convert cm^3 back to Å^3

# Print the required effective height to achieve a water density of 1 g/cm³
print(f"Required effective height to achieve a water density of 1 g/cm³: {required_effective_height:.1f} Å")
