import numpy as np
from sympy import Matrix, Array, Rational, diag
from sympy.matrices.dense import matrix_multiply_elementwise as emult
from collections import OrderedDict
from fractions import Fraction

def symplify(f):
    return Rational(f).limit_denominator(1e9)


def simplify_matrix(arr):
    def f(i, j):
        return symplify(arr[i][j])
    return Matrix(len(arr), len(arr[0]), f)


def vector_to_array(m):
    try:
        return np.array(m[:]).astype(float)
    except TypeError:
        return m


def matrix_to_array(m):
    try:
        return np.array(m[:, :]).astype(float)
    except TypeError:
        return m


class SubregularMicrophiSolution(object):
    """
    Class for a subregular microphi solution object

    This class takes a set of endmembers, sites, site species,
    site interactions and endmember site occupancies as inputs.
    It uses these to compute the relationship
    between binary and tertiary microscopic site interactions
    and the macroscopic endmember interactions.

    Unlike the Asymmetric class, the user must currently specify
    numeric values of site interactions. This is because sympy does
    not yet consistently deal with symbolic algebra (or even assignment)
    on 3D objects.

    If the user also specifies numeric values of endmember proportions,
    these numeric values are propagated through
    the calculations. Otherwise, computations are undertaken
    and parameters returned in symbolic form.

    The class functions set_composition
    and set_site_species_interactions are used to overwrite
    the current vectors/matrices.

    Parameters
    ----------
    mbrs: list of strings
        Endmember names
    sites: list of strings
        Site names (e.g. M1), repeated if there are
        multiple species on the same site
    site_species: list of strings
        Names of species on the sites. Must correspond to the sites
    endmember_site_occupancies: 2d array
        Fractional occupancies of the site_species in each endmember
    endmember_proportions: list of floats (optional)
        List of the endmember proportions in the solution
    site_species_interactions: list containing 2d array and 3d array (optional)
        Matrices of the interactions between site species

    Attributes
    ----------
    endmember_energies: vector
        Endmember free energies contributed by the site interactions
    endmember_interactions: list containing 2d matrix and 3d matrix
        Matrices of the interactions between endmembers
    site_proportions: vector
        List of the site species proportions in the solution
    """

    def __init__(self, mbrs, sites, site_species, endmember_site_occupancies,
                 endmember_proportions=None,
                 site_species_interactions=None,
                 notes=[]):
        self.mbrs = mbrs
        self.sites = sites
        self.set_sites = list(OrderedDict.fromkeys(sites))
        self.site_species = site_species
        self.endmember_site_occupancies = endmember_site_occupancies
        self.n_mbrs = len(mbrs)
        self.n_sites = len(self.set_sites)
        self.n_site_species = len(site_species)
        self.notes = notes

        self.site_indices = [[i for i in range(self.n_site_species)
                              if self.sites[i] == set_site]
                             for set_site in self.set_sites]

        # Check that all the occupancies sum to one
        # for each site in each endmember
        for i, idx_list in enumerate(self.site_indices):
            site_matrix = np.array([self.endmember_site_occupancies[:, i]
                                    for i in idx_list])
            for j, occs in enumerate(site_matrix.T):
                if np.abs(1. - sum(occs)) > 1.e-12:
                    raise ValueError('The site occupancies do not sum to one '
                                     'for the {0} site (site {1}) in {2} '
                                     '(endmember'
                                     ' {3}).'.format(self.set_sites[i], i+1,
                                                     self.mbrs[j], j+1))

        if site_species_interactions is not None:
            self._assign_interactions(site_species_interactions)
        else:
            Wb = []
            for i in range(self.n_site_species):
                Wb.append([])
                for j in range(self.n_site_species):

                    ss_i, ss_j = (self.site_species[i], self.site_species[j])
                    s_i, s_j = (self.sites[i], self.sites[j])

                    if i >= j:  # upper triangular interactions
                        Wb[-1].append(0)
                    elif s_i == s_j:  # same site exchange
                        Wb[-1].append('w({0}{1},{2})'.format(ss_i, ss_j, s_i))
                    else:  # cross-site exchange

                        i0 = self.sites.index(s_i)
                        j0 = self.sites.index(s_j)

                        if (i0 == i or j0 == j):  # 1st row or column
                            Wb[-1].append(0)
                        else:
                            ss_i0, ss_j0 = (self.site_species[i0],
                                            self.site_species[j0])
                            string = '-w({0}{2}{1}{3},{4}{5})'.format(ss_i,
                                                                      ss_i0,
                                                                      ss_j,
                                                                      ss_j0,
                                                                      s_i,
                                                                      s_j)
                            Wb[-1].append(string)

            self.site_binary_interactions = Matrix(Wb)
            self.site_ternary_interactions = Matrix(Wt)

            raise NotImplementedError('Site species interactions (binary and ternary interactions)'
                                      'must currently be supplied as 2D/3D lists or numpy arrays.')

        if endmember_proportions is not None:
            print(endmember_proportions)
            assert len(endmember_proportions) == self.n_mbrs
            self.endmember_proportions = np.array([symplify(f) for f in
                                                   endmember_proportions])
        else:
            self.endmember_proportions = ['p({0})'.format(mbr)
                                          for mbr in self.mbrs]

        self._compute_properties()

    def _compute_properties(self):
        """
        Computes the macroscopic properties of the solution
        """
        try:
            M_pmbr = Matrix([self.endmember_proportions]).T
        except:
            M_pmbr = Matrix([p for p in self.endmember_proportions])
        self.site_species_proportions = (self.endmember_site_occupancies.T
                                         * M_pmbr)
        A = self.endmember_site_occupancies.T
        binary_matrix = self.site_binary_interactions


        # Compact 3D representation of original interactions
        W = self.site_ternary_interactions.copy()

        W += (np.einsum('ij, k -> ijk', binary_matrix, np.ones(self.n_site_species))/self.n_sites
              + np.einsum('ij, jk -> ijk', binary_matrix, np.identity(self.n_site_species))
              - np.einsum('ij, ik -> ijk', binary_matrix, np.identity(self.n_site_species)))/2.


        # Add endmember components to 3D representation
        """
        # in this function, we are not interested in endmember energies,
        # so we have commented out this expression
        W += (np.einsum('i, j, k->ijk', endmember_excesses,
                        np.ones(self.n_site_species), np.ones(self.n_site_species))
              + np.einsum('j, i, k->ijk', endmember_excesses,
                          np.ones(self.n_site_species), np.ones(self.n_site_species))
              + np.einsum('k, i, j->ijk', endmember_excesses,
                          np.ones(self.n_site_species), np.ones(self.n_site_species)))/3.
        """

        Wn = np.einsum('il, jm, kn, ijk -> lmn', A, A, A, W)

        # New endmember components
        # Wn_iii needs to be copied, otherwise just a view onto Wn
        self.endmember_energies =  np.copy(np.einsum('iii->i', Wn))

        # Removal of endmember components from 3D representation
        Wn -= (np.einsum('i, j, k->ijk',
                         self.endmember_energies, np.ones(self.n_mbrs),
                         np.ones(self.n_mbrs))
               + np.einsum('i, j, k->ijk',
                           np.ones(self.n_mbrs), self.endmember_energies,
                           np.ones(self.n_mbrs))
               + np.einsum('i, j, k->ijk',
                           np.ones(self.n_mbrs), np.ones(self.n_mbrs),
                           self.endmember_energies))/3.

        # Transformed 2D components
        # (i=j, i=k, j=k)
        self.endmember_binary_interactions = (np.einsum('jki, jk -> ij', Wn, np.identity(self.n_mbrs))
                                              + np.einsum('jik, jk -> ij', Wn, np.identity(self.n_mbrs))
                                              + np.einsum('ijk, jk -> ij', Wn,
                                                          np.identity(self.n_mbrs))).round(decimals=12)

        # Wb is the 3D matrix corresponding to the terms in the binary matrix,
        # such that the two following print statements produce the same answer
        # for a given array of endmember proportions
        #print(np.einsum('ij, i, j', new_binary_matrix, p, p*p))
        #print(np.einsum('ijk, i, j, k', Wb, p, p, p))
        Wb = (np.einsum('ijk, ij->ijk', Wn, np.identity(self.n_mbrs))
              + np.einsum('ijk, jk->ijk', Wn, np.identity(self.n_mbrs))
              + np.einsum('ijk, ik->ijk', Wn, np.identity(self.n_mbrs)))

        # Remove binary component from 3D representation
        # The extra terms are needed because the binary term in the formulation
        # of a subregular solution model given by
        # Helffrich and Wood includes ternary components (the sum_k X_k part)..
        Wn -= Wb + (np.einsum('ij, k', self.endmember_binary_interactions, np.ones(self.n_mbrs))
                    - np.einsum('ij, ik->ijk', self.endmember_binary_interactions, np.identity(self.n_mbrs))
                    - np.einsum('ij, jk->ijk', self.endmember_binary_interactions, np.identity(self.n_mbrs)))/2.

        # Find the 3D components Wijk by adding the elements at
        # the six equivalent positions in the matrix
        self.endmember_ternary_interactions = np.zeros((self.n_mbrs,
                                                        self.n_mbrs,
                                                        self.n_mbrs))
        for i in range(self.n_mbrs):
            for j in range(i+1, self.n_mbrs):
                for k in range(j+1, self.n_mbrs):
                    val = (Wn[i, j, k] + Wn[j, k, i]
                           + Wn[k, i, j] + Wn[k, j, i]
                           + Wn[j, i, k] + Wn[i, k, j]).round(decimals=12)
                    if np.abs(val) > 1.e-12:
                        self.endmember_ternary_interactions[i,j,k] = val

        # Convert sympy objects to floats if possible
        # also normalise solution to self.n_sites

        #self.endmember_energies /= self.n_sites
        #self.endmember_binary_interactions /= self.n_sites
        #self.endmember_ternary_interactions /= self.n_sites

    def set_composition(self, endmember_proportions):
        """
        Sets the endmember proportions
        """
        assert len(endmember_proportions) == self.n_mbrs
        self.endmember_proportions = np.array([symplify(f) for f in
                                               endmember_proportions])
        self._compute_properties()

    def _assign_interactions(self, site_species_interactions):
        binary_interactions, ternary_interactions = site_species_interactions
        assert len(binary_interactions) == self.n_site_species
        assert len(binary_interactions[0]) == self.n_site_species
        assert len(ternary_interactions) == self.n_site_species
        assert len(ternary_interactions[0]) == self.n_site_species
        assert len(ternary_interactions[0][0]) == self.n_site_species
        self.site_binary_interactions = np.array(binary_interactions)
        self.site_ternary_interactions = np.array(ternary_interactions)

    def set_site_species_interactions(self, site_species_interactions):
        """
        Sets the site interactions
        """
        self._assign_interactions(site_species_interactions)
        self._compute_properties()

    def __str__(self):
        """
        Print the site occupancies and the
        macroscopic properties of the solution.
        """
        vE = Matrix([['U({0})'.format(self.mbrs[i])
                      for i in range(self.n_mbrs)]]).T

        Wb = Matrix([['W({0},{1})'.format(self.mbrs[i], self.mbrs[j])
                      for j in range(self.n_mbrs)]
                     for i in range(self.n_mbrs)])

        string = 'Site occupancies\n'
        for i in range(self.n_site_species):
            string += '{0}({1}) = {2}\n'.format(self.site_species[i],
                                             self.sites[i],
                                             self.site_species_proportions[i])

        string += '\nEndmember free energies:\n'
        for i in range(len(vE)):
            string += '{0} = {1}\n'.format(vE[i], self.endmember_energies[i])


        string += '\nBinary interactions:\n'
        for i in range(self.n_mbrs):
            for j in range(self.n_mbrs):
                if (j > i):
                    string += '{0} = {1}\n'.format(Wb[i, j],
                                                self.endmember_binary_interactions[i, j])
                    string += '{0} = {1}\n'.format(Wb[j, i],
                                                self.endmember_binary_interactions[j, i])

        string2 = '\nTernary interactions:\n'
        ternary_active = False
        for i in range(self.n_mbrs):
            for j in range(i+1,self.n_mbrs):
                for k in range(j+1,self.n_mbrs):
                    if np.abs(self.endmember_ternary_interactions[i,j,k]) > 1.e-6:
                        ternary_active=True
                        string2 += 'W({0},{1},{2}) = {3} \n'.format(self.mbrs[i], self.mbrs[j], self.mbrs[k],
                                                                    self.endmember_ternary_interactions[i,j,k])
        if ternary_active:
            string += string2
        else:
            string += '\n[no non-zero ternary interactions]\n'

        for note in self.notes:
            string += note + '\n'

        return string

    def _interaction_energy(self, proportions,
                            binary_interactions, ternary_interactions):
        p = [float(x) for x in np.array(proportions).flatten()]

        return ((np.einsum('i, j, ij ->', p, p, binary_interactions)
                 + np.einsum('i, j, j, ij ->', p, p, p, binary_interactions)
                 - np.einsum('i, j, i, ij ->', p, p, p, binary_interactions))/2.
                + np.einsum('i, j, k, ijk ->', p, p, p,
                            ternary_interactions))

    @property
    def excess_energy(self):
        """
        Calculates the excess energy at a given composition
        """
        E_xs = self._interaction_energy(self.site_species_proportions,
                                        self.site_binary_interactions,
                                        self.site_ternary_interactions)
        return E_xs

    def check_formalism(self):
        """
        This function calculates the free energy at a
        given composition using the microscopic (site) and
        macroscopic (endmember) formalisms, and makes sure that they
        produce the same answer.

        This function can only be used when the input is fully numeric
        """
        E_xs1 = self.excess_energy

        pG = sum(self.endmember_proportions * self.endmember_energies)
        E_xs2 = self._interaction_energy(self.endmember_proportions,
                                         self.endmember_binary_interactions,
                                         self.endmember_ternary_interactions)

        if np.abs(pG + E_xs2 - E_xs1) < 1.e-6:
            return True
        else:
            raise Exception('Transformed solution does not produce '
                            'the same excess energy as the input.'
                            ' There must be a bug in the subregular formalism.')
