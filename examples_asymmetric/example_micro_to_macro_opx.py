import sys
import numpy as np
sys.path.append('..')
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

site_species_interactions = np.zeros((8, 8)).astype(int)
site_species_interactions[0][1] = 4
site_species_interactions[3][4] = 4
site_species_interactions[1][4] = 1

ss = AsymmetricMicrophiSolution(mbrs, sites, site_species,
                                endmember_site_occupancies,
                                alphas,
                                site_species_interactions=site_species_interactions)

print('deltaE(ofm):', (ss.endmember_energies[4]
                       - 0.5*(ss.endmember_energies[0]
                              + ss.endmember_energies[1])))


print('W(oen, ofs):', ss.endmember_interactions[0,1])
print('W(oen, ofm):', ss.endmember_interactions[0,4])
print('W(ofs, ofm):', ss.endmember_interactions[1,4])
