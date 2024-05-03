#!/usr/bin/env python

from ase import Atoms
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.build import add_adsorbate


infile = 'restart.json'
slab = read(infile)

adsorbate_molecule = ads_mol
mol_position_list = mol_pos_value
height = height_value
position = ads_pos_value

adsorbate = Atoms(adsorbate_molecule, positions=mol_position_list)
add_adsorbate(slab, adsorbate, height, position, offset=None, mol_index=0)
slab.write('restart.json')

