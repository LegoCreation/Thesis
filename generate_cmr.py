#!/usr/bin/env python3
# coding: utf-8
import numpy as np
import qml

mols = [qml.Compound(xyz="./data/splitted_data/C6H6.xyz_"+ str(i)) for i in range(1, 10001)]
mols_representations = []
for mol in mols:
    mol.generate_coulomb_matrix(size=12, sorting="unsorted")
    mols_representations.append(mol.representation)
X = np.array(mols_representations)
np.save("./data/X.npy", X)