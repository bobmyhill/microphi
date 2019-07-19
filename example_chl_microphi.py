import numpy as np
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
wc = 14.  # AlAlMgSi, M1T2 # 2*wc+wt = 28., wt=0
wt = 0.  # AlSi, T2  #
wr = -28  # AlAlMgMg, M1M4 # -(2*wc+wt), wt=0

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
                     [_,   _,   _,   _,   _,   _,   _,   _,   x,  wt],  # Si T2
                     [_,   _,   _,   _,   _,   _,   _,   _,   _,   x]]  # Al


p_mbr = np.array([0.1, 0.2, 0.1, 0.2, 0.1, 0.3])
ss.set_composition(p_mbr)
print(ss)
ss.set_site_species_interactions(site_interactions)
print(ss)
print('Check formalism: {0}'.format(ss.check_formalism()))
