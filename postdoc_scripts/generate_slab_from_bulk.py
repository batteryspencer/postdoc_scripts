#!/usr/bin/env python

import numpy as np
from catkit.gen.surface import SlabGenerator
from ase.io import read, write
from ase.constraints import FixAtoms
from ase.io.trajectory import Trajectory

def check_slab_properties(slab, layers=4, top_elements=2, tolerance=0.5):
    """Check the slab for certain properties and return a report."""
    report = []
    Z_values = [atom.position[2] for atom in slab]
    
    Z_layer_counts = []
    for _ in range(layers):
        Z_clear = [z for z in Z_values if z >= np.min(Z_values) + tolerance]
        Z_values = Z_clear
        Z_layer_counts.append(len(Z_values))
    
    if Z_layer_counts[-1] == 0 and Z_layer_counts[-2] != 0:
        report.append('T')
    else:
        report.append('F')
    
    top_atom_symbols = [atom.symbol for atom in slab if atom.position[2] > (np.max([atom.position[2] for atom in slab]) - tolerance)]
    
    if len(np.unique(top_atom_symbols)) == top_elements:
        report.append('T')
    else:
        report.append('F')
    
    return report

def generate_slab(bulk_atoms, facet_miller_indices=(1, 0, 0), start_image_index=0, num_layers=4, vacuum_length=10, repeat_units=(2, 2, 1)):
    """Generate a slab based on input parameters."""
    slab_gen = SlabGenerator(bulk_atoms, facet_miller_indices, num_layers, vacuum_length, standardize_bulk=True)
    terminations = slab_gen.get_unique_terminations()
    slabs = [slab_gen.get_slab(iterm=i) for i, _ in enumerate(terminations)]
    
    slab_1x1 = slabs[start_image_index]
    slab_repeated = slabs[start_image_index] * repeat_units
    slab_report = check_slab_properties(slab_repeated, num_layers, top_elements=2, tolerance=0.5)
    
    if 'F' in slab_report and len(slabs) > 1:
        start_image_index += 1
        slab_1x1 = slabs[start_image_index]
        slab_repeated = slabs[start_image_index] * repeat_units
        slab_report = check_slab_properties(slab_repeated, num_layers, top_elements=2, tolerance=0.5)
    
    return slab_repeated

def main():
    """Main function to read bulk atoms, generate slab, and write output."""
    bulk_atoms = read('bulk.json')
    facet_miller_indices = (1, 1, 1)
    repeat_units = (3, 3, 1)
    num_layers = 8
    vacuum_length = 10
    start_image_index = 0
    
    slab = generate_slab(bulk_atoms, facet_miller_indices, start_image_index, num_layers, vacuum_length, repeat_units)
    
    slab.center(vacuum=vacuum_length, axis=2)
    
    traj = Trajectory('restart.json', 'w')
    traj.write(slab)

if __name__ == "__main__":
    main()
