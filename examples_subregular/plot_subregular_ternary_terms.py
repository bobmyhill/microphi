import numpy as np
import matplotlib.pyplot as plt

from pylab import rcParams
plt.rcParams['font.family'] = "sans-serif"

# old_mbrs = ['AC', 'BC', 'BD']
pA = np.linspace(0., 1., 201)
pD = np.linspace(0., 1., 201)

pAm, pDm = np.meshgrid(pA, pD)


pm = np.array([pAm, 1. - pAm - pDm, pDm])

ones = np.ones(3)
id = np.eye(3)

Wb = np.array([[0, 0, 2],
               [0, 0, 2],
               [4, 4, 0]])

Wc = (np.einsum('i, jk->ijk', ones, Wb) +
      np.einsum('ij, jk->ijk', Wb, id) -
      np.einsum('ij, ik->ijk', Wb, id))/2.


fig = plt.figure(figsize=(8,4))
ax = [fig.add_subplot(1, 2, i)
      for i in range(1, 3)]


# Define a class that forces representation of float to look a certain way
# This remove trailing zero so '1.0' becomes '1'
class nf(float):
    def __repr__(self):
        str = '%.1f' % (self.__float__(),)
        if str[-1] == '0':
            return '%.0f' % self.__float__()
        else:
            return '%.1f' % self.__float__()

cs = [0,0]
lvl1 = np.linspace(1.e-5, 2., 11)
for i in [0,1]:
    print(i)
    if i == 1:
        Wc[0][1][2] += 2

    energies = np.einsum('ipq, jpq, kpq, ijk->pq', pm, pm, pm, Wc)


    cs[i] = ax[i].contour(pA, pD, energies,
                       lvl1, cmap='viridis')

    cs[i].levels = [nf(val) for val in cs[i].levels]

    ax[i].clabel(cs[i], inline=1, fmt = '%r', fontsize=10)

    ax[i].set_xlabel('p(A)')

    if i == 0:
        ax[i].set_ylabel('p(D)')


    bbox = ax[i].fill_between([0.01, 0.11], [0.01, 0.01], [0.09, 0.09], color='white', linestyle='-', zorder=101)
    bbox.set_edgecolor('k')
    ax[i].text(0.065, 0.045, 'BC', horizontalalignment='center', verticalalignment='center', color='black', zorder=102)

    bbox = ax[i].fill_between([0.01, 0.11], [0.91, 0.91], [0.99, 0.99], color='white', linestyle='-', zorder=101)
    bbox.set_edgecolor('k')
    ax[i].text(0.065, 0.945, 'BD', horizontalalignment='center', verticalalignment='center', color='black', zorder=102)


    bbox = ax[i].fill_between([0.89, 0.99], [0.01, 0.01], [0.09, 0.09], color='white', linestyle='-', zorder=101)
    bbox.set_edgecolor('k')
    ax[i].text(0.945, 0.045, 'AC', horizontalalignment='center', verticalalignment='center', color='black', zorder=102)

    bbox = ax[i].fill_between([0.89, 0.99], [0.91, 0.91], [0.99, 0.99], color='white', linestyle='-', zorder=101)
    bbox.set_edgecolor('k')
    ax[i].text(0.945, 0.945, 'AD', horizontalalignment='center', verticalalignment='center', color='black', zorder=102)


fig.savefig('ABCD_subregular_model.pdf')
plt.show()
