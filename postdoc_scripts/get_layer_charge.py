from pathlib import Path

import numpy as np

from cathub.catmap_interface import formula_to_chemical_symbols


def get_solvation_layer_charge(configuration, adsorbate):
    chemical_symbols_dict = formula_to_chemical_symbols(adsorbate)

    element_list = []
    bader_charge_list = []
    bader_charges_filepath = Path.cwd() / configuration / bader_charges_filename
    coordinates_filepath = Path.cwd() / configuration / coordinates_filename
    with open(bader_charges_filepath, 'r') as bader_charges_file:
        for line_index, line in enumerate(bader_charges_file):
            line_elements = line.split()
            element_list.append(line_elements[3])
            bader_charge_list.append(float(line_elements[5]))

    bader_charge_array = np.asarray(bader_charge_list)
    num_atoms = len(element_list)
    coordinates = np.loadtxt(coordinates_filepath, skiprows=2, max_rows=num_atoms)[:, 1:4] * bohr
    z_coordinates = coordinates[:, 2]
    total_indices = np.arange(len(z_coordinates)).tolist()

    chemical_symbol_to_index_list = {}
    for chemical_symbol in chemical_symbols_dict:
        chemical_symbol_to_index_list[chemical_symbol] = [i for i, x in enumerate(element_list) if x == chemical_symbol]

    chemical_symbols_to_sorted_indices = {}
    anchor_first_run = 1  # first run
    for chemical_symbol, num_atoms in chemical_symbols_dict.items():
        # identify indices of the chemical symbol with lowest z-coordinate
        chemical_symbol_indices = np.asarray([i for i, x in enumerate(element_list) if x == chemical_symbol])
        chemical_symbol_z_coordinates = z_coordinates[chemical_symbol_indices]
        sort_indices = chemical_symbol_z_coordinates.argsort()
        chemical_symbols_to_sorted_indices[chemical_symbol] = chemical_symbol_indices[sort_indices]
        if anchor_first_run:
            anchor_chemical_symbol = chemical_symbol
            anchor_atom_index = chemical_symbols_to_sorted_indices[chemical_symbol][0]
            anchor_first_run = 0
        elif z_coordinates[chemical_symbols_to_sorted_indices[chemical_symbol][0]] < z_coordinates[anchor_atom_index]:
            anchor_chemical_symbol = chemical_symbol
            anchor_atom_index = chemical_symbols_to_sorted_indices[chemical_symbol][0]

    anchor_z_coordinate = z_coordinates[anchor_atom_index]
    substrate_indices = np.where(z_coordinates < anchor_z_coordinate)[0].tolist()
    non_substrate_indices = [index for index in total_indices if index not in substrate_indices]

    adsorbate_scrape = {}
    num_atoms_to_scrape = sum(chemical_symbols_dict.values())
    for chemical_symbol, num_atoms in chemical_symbols_dict.items():
        adsorbate_scrape[chemical_symbol] = num_atoms

    adsorbate_indices = [anchor_atom_index]
    adsorbate_scrape[anchor_chemical_symbol] -= 1
    num_atoms_to_scrape -= 1

    reference_atom_indices = [anchor_atom_index]
    while num_atoms_to_scrape:
        for reference_atom_index in reference_atom_indices:
            distance_to_ref = np.linalg.norm(coordinates[non_substrate_indices] - coordinates[reference_atom_index], axis=1)
            bonding_subindices_to_ref = np.where((distance_to_ref > 0) & (distance_to_ref < bond_distance_cutoff))[0]
            bonding_atom_indices_to_ref = [non_substrate_indices[index] for index in bonding_subindices_to_ref if non_substrate_indices[index] not in adsorbate_indices]
            reference_atom_indices = bonding_atom_indices_to_ref[:]
            for atom_index in bonding_atom_indices_to_ref:
                chemical_symbol = element_list[atom_index]
                if adsorbate_scrape[chemical_symbol]:
                    adsorbate_indices.append(atom_index)
                    adsorbate_scrape[chemical_symbol] -= 1
                    num_atoms_to_scrape -= 1
    
    solvation_layer_indices = [index for index in non_substrate_indices if index not in adsorbate_indices]
    solvation_layer_charges = bader_charge_array[solvation_layer_indices]
    solvation_layer_charge = solvation_layer_charges.sum()
    return solvation_layer_charge
