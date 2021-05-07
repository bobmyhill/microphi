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
mbr_flags = ['M', 'M', 'M']
old_mbrs = ['AC', 'BC', 'BD']

# New endmember names
# mf has the composition of fm, but with the Fe and Mg sites exchanged
new_mbrs = ['AC', 'BC', 'AD']
mbr_transformations = np.array([[1,  0,  0],
                                [0,  1,  0],
                                [1,  -1,  1]]) # AD = AC - BC + BD


binary_interactions = np.array([[0, 0, 2.e3],
                                [0, 0, 2.e3],
                                [4.e3, 4e3, 0]])

# Ternary interactions for the original endmember set are equal to zero
ternary_interactions = np.zeros((3, 3, 3))

notes = ['Note that the endmember free energies do not '
         'include the component from mechanical mixing.',
         'However, this is easily calculated from the transformation matrix.',
         'In this case, U(AD) = U(AC) - U(BC) + U(BD) + c, where c is the numerical value given above.']

ss = SubregularMicrophiSolution(new_mbrs, mbr_flags, old_mbrs,
                                mbr_transformations,
                                endmember_proportions = [0.2, 0.3, 0.5],
                                site_species_interactions = [binary_interactions,
                                                             ternary_interactions],
                                notes=notes)

# Print the transformed solution properties
print(ss)
print('Check energies are the same: {0}'.format(ss.check_formalism()))



# Now, let's check the inverse transformation
notes = ['The inverse transformation. ',
         'Note that because we do not include the endmember energies, ',
         'U(BD) is now non zero and equal to -U(AD)']
inv_mbr_transformations = np.array([[1,  0,  0],
                                    [0,  1,  0],
                                    [-1,  1,  1]]) # BD = -AC + BC + AD

ss_inv = SubregularMicrophiSolution(old_mbrs, mbr_flags, new_mbrs,
                                inv_mbr_transformations,
                                endmember_proportions = ss.site_species_proportions,
                                site_species_interactions = [ss.endmember_binary_interactions,
                                                             ss.endmember_ternary_interactions],
                                notes=notes)

print(ss_inv)
print('Check energies are the same: {0}'.format(ss_inv.check_formalism()))
