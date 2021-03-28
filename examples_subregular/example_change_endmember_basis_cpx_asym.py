import sys
import numpy as np
sys.path.append('..')
from microphi import SubregularMicrophiSolution

# We can also use the Microphi formalism to convert from one
# endmember set to another. Here's an example in the FMS clinopyroxene system

# This example shows that in multisite subregular models,
# ternary interactions can appear just by a change in basis.
# Therefore there is no logical justification for using an asymmetric model
# *and* setting ternary terms to zero.

# New endmember names
# mf has the composition of fm, but with the Fe and Mg sites exchanged
mbrs = ['en', 'fm', 'mf']

# Site species information
sites = ['M', 'M', 'M', 'M']
site_species = ['en', 'fs', 'fm']


notes = ['Note that the endmember free energies do not '
         'include the component from mechanical mixing.',
         'However, this is easily calculated from the transformation matrix.']
endmember_site_occupancies = np.array([[1,  0,  0],
                                       [0,  0,  1],
                                       [1,  1,  -1]]) # mf = en + fs - fm

binary_interactions = np.array([[0, 2.9e3, 29.8e3],
                                [1.e3, 0, 26.6e3],
                                [0, 2.e3, 0]])

#
ternary_interactions = np.zeros((3, 3, 3))

ss = SubregularMicrophiSolution(mbrs, sites, site_species,
                                endmember_site_occupancies,
                                endmember_proportions = [0.2, 0.3, 0.5],
                                site_species_interactions = [binary_interactions,
                                                             ternary_interactions],
                                notes=notes)
print(ss)
print(ss.check_formalism())
