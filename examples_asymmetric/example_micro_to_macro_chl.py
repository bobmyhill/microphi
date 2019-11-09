import sys
import numpy as np
sys.path.append('..')
from microphi import AsymmetricMicrophiSolution

# Endmember names
mbrs = ['afchl', 'ames', 'daph', 'clin', 'ochl1', 'ochl4']

# Site species information
sites = ['M1', 'M1', 'M1', 'M23', 'M23', 'M4', 'M4', 'M4', 'T2', 'T2']
site_species = ['Mg', 'Fe', 'Al', 'Mg', 'Fe', 'Mg', 'Fe', 'Al', 'Si', 'Al']
alphas = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

endmember_site_occupancies = np.array([[1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
                                       [0, 0, 1, 1, 0, 0, 0, 1, 0, 1],
                                       [0, 1, 0, 0, 1, 0, 0, 1, 0.5, 0.5],
                                       [1, 0, 0, 1, 0, 0, 0, 1, 0.5, 0.5],
                                       [1, 0, 0, 0, 1, 0, 1, 0, 1, 0],
                                       [0, 1, 0, 1, 0, 1, 0, 0, 1, 0]])

ss = AsymmetricMicrophiSolution(mbrs, sites, site_species,
                                endmember_site_occupancies,
                                alphas)
print(ss)

w = 4.
wo = 10.
p = 0.7
wc = 14.  # wcrt in paper = w(AlAlMgSi,M1T2) = w(AlAlMgSi,M4T2)
wt = 0.   # wt in paper = w(AlSi,T2)
wr = -28  # wcro in paper = w(AlAlMgMg,M1M4)

# Checks:
assert 2.*wo + wr + 2.*wc + wt == 20.
assert wo + 1./2.*wc + 1./4.*wt == 17.

# There are typos in the table on p.256 of Powell et al. (2014).
# For example, the sum for W(ames,ochl4) in the first column on that page
# can easily be shown to be equal to 17 kJ,
# rather than 13 kJ/mol as given in the table.

# Diagonals should be zero
x = 0

# First row and first column of each cross-site block should be zero
_ = 0

#                     M1             M23       M4             T2
#                     1              4         1              2
#                     Mg   Fe   Al   Mg   Fe   Mg   Fe   Al   Si   Al
site_interactions = [[x,   w,   wo,  _,   _,   _,   _,   _,   _,   _],  # Mg M1
                     [_,   x, p*wo,  _,   0,   _,   0,   0,   _,   0],  # Fe
                     [_,   _,   x,   _,   0,   _,   0, -wr,   _, -wc],  # Al
                     [_,   _,   _,   x, 4.*w,  _,   _,   0,   _,   _],  # Mg M2
                     [_,   _,   _,   _,   x,   _,   0,   0,   _,   0],  # Fe
                     [_,   _,   _,   _,   _,   x,   w,  wo,   _,   _],  # Mg M4
                     [_,   _,   _,   _,   _,   _,   x, p*wo,  _,   0],  # Fe
                     [_,   _,   _,   _,   _,   _,   _,   x,   _, -wc],  # Al
                     [_,   _,   _,   _,   _,   _,   _,   _,   x,  2.*wt],  # Si T2
                     [_,   _,   _,   _,   _,   _,   _,   _,   _,   x]]  # Al


print('Setting site interactions...')
ss.set_site_species_interactions(site_interactions)
print(ss)

print('Setting compositions...')
p_mbr = np.array([0.1, 0.2, 0.1, 0.2, 0.1, 0.3])
ss.set_composition(p_mbr)
print('\nEndmember proportions:')
print(mbrs)
print(p_mbr)
print('\nSite species proportions:')
print(sites)
print(site_species)
print(ss.site_species_proportions)
print()

print('Check formalism: {0}'.format(ss.check_formalism()))
