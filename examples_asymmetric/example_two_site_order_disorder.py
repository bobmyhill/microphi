import sys
import numpy as np
sys.path.append('..')
from microphi import AsymmetricMicrophiSolution

#  The example in this file corresponds to the example illustrated in Figure 3
#  of Myhill and Connolly (submitted to GCA).

#  It consists of a two-site model, where both sites are occupied by either
#  species A or B.

sites = ['1', '1', '2', '2']
site_species = ['A', 'B', 'A', 'B']

mbrs = ['AA', 'BB', 'AB']
endmember_site_occupancies = np.array([[1., 0., 1., 0.],
                                       [0., 1., 0., 1.],
                                       [1., 0., 0., 1.]])

# Symmetric solution
alphas = np.array([1., 1., 1., 1.])

print('Solution 1: Ordered endmembers destabilised, both sites identical')
Wa = np.array([[0., 1., 0., 0.],
               [0., 0., 0., -(4.)],  # w(AABB,M1M2) = 4., so 2AB is less stable than AA+BB at 0 K.
               [0., 0., 0., 1.],
               [0., 0., 0., 0.]])

ss = AsymmetricMicrophiSolution(mbrs, sites, site_species,
                                endmember_site_occupancies,
                                alphas,
                                site_species_interactions=Wa)

print(ss)


print('Solution 2: Ordered endmembers stabilised, both sites identical')
Wa = np.array([[0., 1., 0., 0.],
               [0., 0., 0., -(-4.)],  # w(AABB,M1M2) = -4, so 2AB is more stable than AA+BB at 0 K.
               [0., 0., 0., 1.],
               [0., 0., 0., 0.]])

ss = AsymmetricMicrophiSolution(mbrs, sites, site_species,
                                endmember_site_occupancies,
                                alphas,
                                site_species_interactions=Wa)

print(ss)


print('Solution 3: Ordered endmembers stabilised, '
      'A on Site 1 assigned a higher energy than on Site 2')
E_A = 1.
n_sites = 2.
Wa = np.array([np.array([0., 1., 0., 0.]) + E_A/n_sites,
               [0., 0., 0., -(-4.)],  # w(AABB,M1M2) = -4, so 2AB is more stable than AA+BB at 0 K.
               [0., 0., 0., 1.],
               [0., 0., 0., 0.]])

ss = AsymmetricMicrophiSolution(mbrs, sites, site_species,
                                endmember_site_occupancies,
                                alphas,
                                site_species_interactions=Wa)

print(ss)
