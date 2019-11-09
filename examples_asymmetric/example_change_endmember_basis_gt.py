import sys
import numpy as np
from sympy import nsimplify
sys.path.append('..')
from microphi import AsymmetricMicrophiSolution

# We can also use the Microphi formalism to convert from one
# endmember set to another. Here's an example in the CFMASO garnet system

# New endmember names
mbrs = ['py', 'alm', 'gr', 'kho']

# Site species information
sites = ['M', 'M', 'M', 'M']
site_species = ['py', 'alm', 'gr', 'andr']
alphas = [1, 1, 2.7, 2.7]


notes = ['Note that the endmember free energies do not '
         'include the component from mechanical mixing.',
         'However, this is easily calculated from the transformation matrix.']
endmember_site_occupancies = np.array([[1,  0,  0,  0],
                                       [0,  1,  0,  0],
                                       [0,  0,  1,  0],
                                       [1,  0, -1,  1]])

ss = AsymmetricMicrophiSolution(mbrs, sites, site_species,
                                endmember_site_occupancies,
                                alphas, notes=notes)
print(ss)

ss.set_site_species_interactions([[0, nsimplify(2.5), 31, 'wpy'],
                                  [0, 0, 5, '7/10*wpy'],
                                  [0, 0, 0, 2],
                                  [0, 0, 0, 0]])

print(ss)

ss.set_site_species_interactions([[0, '5/2', 31, nsimplify('53.2')],
                                  [0, 0, 5, nsimplify('7/10*53.2')],
                                  [0, 0, 0, 2],
                                  [0, 0, 0, 0]])

print(ss)
