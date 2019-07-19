import numpy as np
from microphi import AsymmetricMicrophiSolution

# Endmember names
mbrs = ['oen', 'ofs', 'mgts', 'odi', 'ofm']

# Site species information
sites = ['M1', 'M1', 'M1', 'M2', 'M2', 'M2', 'T1', 'T1']
site_species = ['Mg', 'Fe', 'Ca', 'Mg', 'Fe', 'Al', 'Si', 'Al']
alphas = [1, 1, 1, 1, 1, 1, 1, 1]

endmember_site_occupancies = np.array([[1, 0, 0, 1, 0, 0, 1, 0],
                                       [0, 1, 0, 0, 1, 0, 1, 0],
                                       [1, 0, 0, 0, 0, 1, 0.5, 0.5],
                                       [0, 0, 1, 1, 0, 0, 1, 0],
                                       [0, 1, 0, 1, 0, 0, 1, 0]])

ss = AsymmetricMicrophiSolution(mbrs, sites, site_species,
                                endmember_site_occupancies,
                                alphas)


print(ss)
