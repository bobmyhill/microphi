
import numpy as np
from microphi import AsymmetricMicrophiSolution


mbrs = ['AA', 'BB', 'AB']
sites = ['M1', 'M1', 'M2', 'M2']
site_species = ['A', 'B', 'A', 'B']
endmember_site_occupancies = np.array([[1., 0., 1., 0.],
                              [0., 1., 0., 1.],
                              [1., 0., 0., 1.]])
alphas = np.array([1., 1., 1., 1.])
Wa = np.array([[0., 1., 0., 0.],
      [0., 0., 0., -(-4.)], # w(AABB,M1M2) = -4, so 2AB is more stable than AA+BB at 0 K.
      [0., 0., 0., 1.],
      [0., 0., 0., 0.]])

ss = AsymmetricMicrophiSolution(mbrs, sites, site_species,
                                endmember_site_occupancies,
                                alphas,
                                site_species_interactions=Wa)

print(ss)

Wa = np.array([[0., 1., 0., 0.],
               [0., 0., 0., -(-4.)], # w(AABB,M1M2) = -4, so 2AB is more stable than AA+BB at 0 K.
               [0., 0., 1., 1.],
               [0., 0., 0., 0.]])

ss = AsymmetricMicrophiSolution(mbrs, sites, site_species,
                                endmember_site_occupancies,
                                alphas,
                                site_species_interactions=Wa)

print(ss)

exit()

Wa = np.array([[0., 1., 0., 0.],
      [0., 0., 0., -(8.)], # w(AABB,M1M2) = 8., so 2AB is less stable than AA+BB at 0 K.
      [0., 0., 0., 1.],
      [0., 0., 0., 0.]])

ss = AsymmetricMicrophiSolution(mbrs, sites, site_species,
                                endmember_site_occupancies,
                                alphas,
                                site_species_interactions=Wa)

print(ss)
