import numpy as np
from microphi import AsymmetricMicrophiSolution


# Endmember names
mbrs = ['py', 'gr', 'pypygr', 'pygrgr']

# Site species information
sites = ['X1', 'X1', 'X2', 'X2', 'X3', 'X3']
site_species = ['Mg', 'Ca', 'Mg', 'Ca', 'Mg', 'Ca']
alphas = [1., 1., 1., 1., 1., 1.]

endmember_site_occupancies = np.array([[1, 0, 1, 0, 1, 0],
                                       [0, 1, 0, 1, 0, 1],
                                       [1, 0, 1, 0, 0, 1],
                                       [1, 0, 0, 1, 0, 1]])

ss = AsymmetricMicrophiSolution(mbrs, sites, site_species,
                                endmember_site_occupancies,
                                alphas)

#print(ss)

# Diagonals should be zero
x = 0

# First row and first column of each cross-site block should be zero
_ = 0

MM = 2
MC = 10 + MM

print('MC={0} , MM={1}'.format(MC, MM))

#                     X1        X2        X3
#                     1         1         1
#                     Mg   Ca   Mg   Ca   Mg   Ca
site_interactions = [[x,  MC,   _,   _,   _,   _],   # Mg X1
                     [_,   x,   _,  MM,   _,  MM],   # Ca
                     [_,   _,   x,  MC,   _,   _],   # Mg X2
                     [_,   _,   _,   x,   _,  MM],   # Ca
                     [_,   _,   _,   _,   x,  MC],   # Mg X3
                     [_,   _,   _,   _,   _,   x]]   # Ca

print('Setting site interactions...')
ss.set_site_species_interactions(site_interactions)
print(ss)
