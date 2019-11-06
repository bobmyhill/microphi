import numpy as np

# the desired matrix.
# Note typos in Powell and Holland (2014) for
# afchl-ames (16 rather than 20)
# ames-daph (10 rather than 30)
# ames ochl1 (33 not 29)
# ames ochl4 (17 not 13)
W_endmember = np.array([[0, 20, 37, 17, 20, 4],
                        [0, 0,  30, 17, 33, 17],
                        [0, 0,  0,  20, 18, 33],
                        [0, 0,  0,  0,  30, 21],
                        [0, 0,  0,  0,  0,  24],
                        [0, 0,  0,  0,  0,  0]])


E = np.array([[1, 0, 0, 1, 0, 1, 0, 0, 1, 0], # afchl
              [0, 0, 1, 1, 0, 0, 0, 1, 0, 1], # ames
              [0, 1, 0, 0, 1, 0, 0, 1, 0.5, 0.5], # daph
              [1, 0, 0, 1, 0, 0, 0, 1, 0.5, 0.5], # clin
              [1, 0, 0, 0, 1, 0, 1, 0, 1, 0], # ochl1
              [0, 1, 0, 1, 0, 1, 0, 0, 1, 0]]) # ochl4


alpha = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])

w = 4.
wo = 10.
phi = 0.7
wcrt = 14. # AlAlMgSi, M1T2 # 2*wcrt+wt = 28., wt=0
wt = 0. # AlSi, T2  #
wcro = -28 # AlAlMgMg, M1M4 # -(2*wcrt+wt), wt=0


# Cross terms see a sign change on insertion into the array:
# Four components of any exchange reaction, AC, AD, BC, BD
# Exchange reaction: AC + BD = BC + AD
# Cross site term: w_ACBD = (E(BC) + E(AD)) - (E(AC) + E(BD))
# All reactions involving A and C are redundant; set to zero.
# Then w_ACBD -> - E(BD)


# Diagonals should be zero
x = 0


# First row and first column of each cross-site block should be zero
_ = 0

#              M1                         M23             M4                        T2
#              1                          4               1                         2
#              Mg       Fe       Al       Mg      Fe      Mg       Fe       Al      Si      Al
W = np.array([[x,       w,      wo,       _,      _,      _,       _,       _,      _,      _],    # Mg M1
              [_,       x,  phi*wo,       _,      0,      _,       0,       0,      _,      0],    # Fe
              [_,       _,       x,       _,      0,      _,       0,   -wcro,      _,  -wcrt],    # Al
              [_,       _,       _,       x,   4.*w,      _,       _,       0,      _,      _],    # Mg M23
              [_,       _,       _,       _,      x,      _,       0,       0,      _,      0],    # Fe
              [_,       _,       _,       _,      _,      x,       w,      wo,      _,      _],    # Mg M4
              [_,       _,       _,       _,      _,      _,       x,  phi*wo,      _,      0],    # Fe
              [_,       _,       _,       _,      _,      _,       _,       x,      _,  -wcrt],    # Al
              [_,       _,       _,       _,      _,      _,       _,       _,      x,     wt],    # Si T2
              [_,       _,       _,       _,      _,      _,       _,       _,      _,      x]])   # Al


A = E.T


n_sites = np.sum(E[0])
alpha_p = np.einsum('i, il->l', alpha, A)
B = np.einsum('i, il, l->il', alpha, A, 1./alpha_p)

# Modify W matrix to be the matrix including the effect of alphas
# (multiply elements by 2/(ai + aj))
Wmod = (2.*W/(np.einsum('i, j -> ij', alpha, np.ones(len(alpha))) +
              np.einsum('i, j -> ij', np.ones(len(alpha)), alpha)))

WC = np.einsum('il, jm, ij -> lm', B, B, Wmod)

G_mbr_p = np.einsum('l,ll->l', alpha_p, WC)

D = WC - (np.einsum('l, m -> lm', G_mbr_p/alpha_p, np.ones(len(WC))) +
          np.einsum('m, l -> lm', G_mbr_p/alpha_p, np.ones(len(WC))))/2.

# Make triangular
W_pmod = np.triu(D + D.T)

# Modify W_p to remove the alpha components
W_p = (np.einsum('i, j -> ij', alpha_p, np.ones(len(alpha_p))) +
        np.einsum('i, j -> ij', np.ones(len(alpha_p)), alpha_p))/2.*W_pmod

# Endmember proportions sum to one, not nsites
W_p *= n_sites
G_mbr_p *= n_sites

# Optionally normalise the alpha values
alpha_p /= n_sites


print(G_mbr_p)
print(alpha_p)
print(W_p)


print('CHECK VALUES')
x_mbr = np.array([0.1, 0.2, 0.1, 0.2, 0.1, 0.3])
x_sel = A.dot(x_mbr)
Wmod *= n_sites

print((alpha*x_sel).dot(Wmod).dot(alpha*x_sel)/(x_sel.dot(alpha)))

print(x_mbr.dot(G_mbr_p) +
      ((alpha_p*x_mbr).dot(2.*W_p/(np.einsum('i, j -> ij',
                                  alpha_p, np.ones(len(alpha_p))) +
                        np.einsum('i, j -> ij',
                                  np.ones(len(alpha_p)), alpha_p))).dot(alpha_p*x_mbr)) /
      alpha_p.dot(x_mbr))
