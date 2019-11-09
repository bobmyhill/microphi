import sys
import numpy as np
sys.path.append('..')
from microphi import SubregularMicrophiSolution

# We can also use the Microphi formalism to convert from one
# endmember set to another. Here's an example in the CMS clinopyroxene system

# New endmember names
# fm has Fe on the Ca site
mbrs = ['di', 'fm', 'en', 'fs']

# Site species information
sites = ['M', 'M', 'M', 'M']
site_species = ['di', 'hed', 'en', 'fs']


notes = ['Note that the endmember free energies do not '
         'include the component from mechanical mixing.',
         'However, this is easily calculated from the transformation matrix.']
endmember_site_occupancies = np.array([[1,  0,  0,  0],
                                       [1, -1,  0,  1],
                                       [0,  0,  1,  0],
                                       [0,  0,  0,  1]])

binary_interactions = np.array([[0, 2.9e3, 29.8e3, 25.8e3],
                                [0, 0, 26.6e3, 20.9e3],
                                [0, 0, 0, 2.3e3],
                                [0, 0, 0, 0]])
binary_interactions = (binary_interactions + binary_interactions.T)

ternary_interactions = np.zeros((4, 4, 4))

ss = SubregularMicrophiSolution(mbrs, sites, site_species,
                                endmember_site_occupancies,
                                site_species_interactions = [binary_interactions,
                                                             ternary_interactions],
                                notes=notes)
print(ss)
