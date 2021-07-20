import sys
import numpy as np
sys.path.append('..')
from microphi import SubregularMicrophiSolution

print('This example presents the conversion from a microscopic to macroscopic')
print('basis for a subregular two-site [A,B][C,D] solution')
print('as published in Myhill and Connolly (2021).')
print('')

print('The model is set up making the following model assumptions:')
print('- A-B mixing on Site M1 is nearly ideal')
print('- C-D mixing on Site M2 is moderately non-ideal and asymmetric')
print('- (wCD = 2 kJ/mol, wDC = 4 kJ/mol)')
print('- There are no interactions between species on different sites.')
print('')

print('This example is designed to demonstrate the emergence of ')
print('a ternary interaction parameter in the macroscopic expressions ')
print('of the same model.')
print('')
# Endmember names
mbrs = ['AC', 'BC', 'BD']

# Site species information
sites = ['M1', 'M1', 'M2', 'M2']
site_species = ['A', 'B', 'C', 'D']

endmember_site_occupancies = np.array([[1, 0, 1, 0],
                                       [0, 1, 1, 0],
                                       [0, 1, 0, 1]])

binary_interactions = np.array([[0., 0., 0., 0],
                                [0., 0., 0., 0],
                                [0., 0., 0., 2],
                                [0., 0., 4., 0]])


ternary_interactions = np.zeros((4, 4, 4))

ss = SubregularMicrophiSolution(mbrs, sites, site_species,
                                endmember_site_occupancies,
                                site_species_interactions=[binary_interactions, ternary_interactions])
print(ss)

print('Binary interactions in matrix form:')
print(ss.endmember_binary_interactions)

print('')
print('NB. Note the emergence of the non-zero ternary term, above.')


print('')
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
