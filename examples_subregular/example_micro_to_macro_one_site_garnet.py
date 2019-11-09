import sys
import numpy as np
sys.path.append('..')
from microphi import SubregularMicrophiSolution

# This example presents a simple subregular one-site garnet solution
# as published in Ganguly et al., 1996

# Endmember names
mbrs = ['py', 'alm', 'gr']

# Site species information
sites = ['M1', 'M1', 'M1']
site_species = ['Mg', 'Fe', 'Ca']

endmember_site_occupancies = np.array([[1, 0, 0],
                                       [0, 1, 0],
                                       [0, 0, 1]])

binary_interactions = np.array([[0., 2117., 9834.],
                                [695., 0., 6773.],
                                [21627., 873., 0.]])*3.


ternary_interactions = np.zeros((3, 3, 3))

ss = SubregularMicrophiSolution(mbrs, sites, site_species,
                                endmember_site_occupancies,
                                site_species_interactions=[binary_interactions, ternary_interactions])
print(ss)
print('Setting compositions...')
p_mbr = np.array([0.1, 0.6, 0.3])
ss.set_composition(p_mbr)
print('\nEndmember proportions:')
print(mbrs)
print(p_mbr)
print('\nSite species proportions:')
print(sites)
print(site_species)
print(ss.site_species_proportions)
print('Excess energy: {0}'.format(ss.excess_energy))
print()
print('Check formalism: {0}'.format(ss.check_formalism()))

print('\n\nSame model with dummy site. Should produce the same answers...')
# Site species information
sites = ['M1', 'M1', 'M1', 'M2', 'M2']
site_species = ['Mg', 'Fe', 'Ca', 'Al', 'Fef']

endmember_site_occupancies = np.array([[1, 0, 0, 1, 0],
                                       [0, 1, 0, 1, 0],
                                       [0, 0, 1, 1, 0]])

binary_interactions = np.array([[0., 2117., 9834., 0., 0],
                                [695., 0., 6773., 0., 0],
                                [21627., 873., 0., 0., 0],
                                [0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0.]])*3.


ternary_interactions = np.zeros((5, 5, 5))

ss = SubregularMicrophiSolution(mbrs, sites, site_species,
                                endmember_site_occupancies,
                                site_species_interactions=[binary_interactions, ternary_interactions])
print(ss)
print('Setting compositions...')
p_mbr = np.array([0.1, 0.6, 0.3])
ss.set_composition(p_mbr)
print('\nEndmember proportions:')
print(mbrs)
print(p_mbr)
print('\nSite species proportions:')
print(sites)
print(site_species)
print(ss.site_species_proportions)
print('Excess energy: {0}'.format(ss.excess_energy))
print()


print('Check formalism: {0}'.format(ss.check_formalism()))
