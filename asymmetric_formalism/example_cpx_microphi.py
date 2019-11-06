import numpy as np
from microphi import AsymmetricMicrophiSolution


# Endmember names
mbrs = ['di', 'hed', 'cen', 'cfs']

# Site species information
sites = ['M1', 'M1', 'M1', 'M2', 'M2']
site_species = ['Mg', 'Fe', 'Ca', 'Mg', 'Fe']
alphas = [1, 1, 1.4, 1, 1]

endmember_site_occupancies = np.array([[0, 0, 1, 1, 0],
                                       [0, 0, 1, 0, 1],
                                       [1, 0, 0, 1, 0],
                                       [0, 1, 0, 0, 1]])

ss = AsymmetricMicrophiSolution(mbrs, sites, site_species,
                                endmember_site_occupancies,
                                alphas)

print(ss)
