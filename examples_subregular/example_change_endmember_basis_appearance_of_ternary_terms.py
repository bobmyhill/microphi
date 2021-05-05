import sys
import numpy as np
sys.path.append('..')
from microphi import SubregularMicrophiSolution

# We can also use the Microphi formalism to convert from one
# endmember set to another. Here's an example in a fictional FMS system

# This example shows that in multisite subregular models,
# ternary interactions can appear just by a change in basis.
# Therefore there is no logical justification for using an asyenetric model
# *and* setting ternary terms to zero.

# Site species information
mbr_flags = ['M', 'M', 'M', 'M']
old_mbrs = ['mm', 'ff', 'fm']

# New endmember names
# mf has the composition of fm, but with the Fe and Mg sites exchanged
new_mbrs = ['mm', 'fm', 'mf']
mbr_transformations = np.array([[1,  0,  0],
                                [0,  0,  1],
                                [1,  1,  -1]]) # mf = mm + ff - fm

binary_interactions = np.array([[0, 2e3, 0e3],
                                [1.e3, 0, 4e3],
                                [0, 2.e3, 0]])

# Ternary interactions for the original endmember set are equal to zero
ternary_interactions = np.zeros((3, 3, 3))

notes = ['Note that the endmember free energies do not '
         'include the component from mechanical mixing.',
         'However, this is easily calculated from the transformation matrix.']

ss = SubregularMicrophiSolution(new_mbrs, mbr_flags, old_mbrs,
                                mbr_transformations,
                                endmember_proportions = [0.2, 0.3, 0.5],
                                site_species_interactions = [binary_interactions,
                                                             ternary_interactions],
                                notes=notes)

# Print the transformed solution properties
print(ss)

print('Check energies are the same: {0}'.format(ss.check_formalism()))
