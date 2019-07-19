import numpy as np
from sympy import Matrix, Rational, diag
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


class AsymmetricMicrophiSolution(object):
    """
    Class for a microphi solution object
    (see Powell et al., 2014; doi:10.1111/jmg.12070)

    This class takes a set of endmembers, sites, site species,
    van Laar alpha parameters and endmember site occupancies. It
    uses these to compute the relationship between microscopic
    site interactions and the macroscopic endmember interactions.

    If the user specifies numeric values of endmember proportions,
    site-species specific alphas and site interactions, these numeric
    values are propagated through the calculations. Otherwise, computations
    are undertaken and parameters returned in symbolic form.

    The class functions set_alphas, set_composition
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
    alphas: list of floats (optional)
        List of the van Laar asymmetry parameters
        corresponding to each site species
    endmember_proportions: list of floats (optional)
        List of the endmember proportions in the solution
    site_species_interactions: 2d array (optional)
        Upper triangular matrix of the interactions between site species

    Attributes
    ----------
    endmember_energies: vector
        Endmember free energies contributed by the site interactions
    endmember_interactions: matrix
        Upper triangular matrix of the interactions between endmembers
    alpha_prime: vector
        Van Laar asymmetry parameters for endmember interactions
    site_proportions: vector
        List of the site species proportions in the solution
    """

    def __init__(self, mbrs, sites, site_species, endmember_site_occupancies,
                 alphas=None,
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

        if alphas is not None:
            assert len(alphas) == self.n_site_species
            self.site_species_alphas = alphas
        else:
            self.site_species_alphas = ['a({0},{1})'.format(self.sites[i],
                                               self.site_species[i])
                           for i in range(self.n_site_species)]

        if site_species_interactions is not None:
            assert len(site_species_interactions) == self.n_site_species
            assert len(site_species_interactions[0]) == self.n_site_species
            self.site_species_interactions = Matrix(site_species_interactions)
        else:
            Wa = []
            for i in range(self.n_site_species):
                Wa.append([])
                for j in range(self.n_site_species):

                    ss_i, ss_j = (self.site_species[i], self.site_species[j])
                    s_i, s_j = (self.sites[i], self.sites[j])

                    if i >= j:  # upper triangular interactions
                        Wa[-1].append(0)
                    elif s_i == s_j:  # same site exchange
                        Wa[-1].append('w({0}{1},{2})'.format(ss_i, ss_j, s_i))
                    else:  # cross-site exchange

                        i0 = self.sites.index(s_i)
                        j0 = self.sites.index(s_j)

                        if (i0 == i or j0 == j):  # 1st row or column
                            Wa[-1].append(0)
                        else:
                            ss_i0, ss_j0 = (self.site_species[i0],
                                            self.site_species[j0])
                            string = '-w({0}{2}{1}{3},{4}{5})'.format(ss_i,
                                                                      ss_i0,
                                                                      ss_j,
                                                                      ss_j0,
                                                                      s_i,
                                                                      s_j)
                            Wa[-1].append(string)

            self.site_species_interactions = Matrix(Wa)

        if endmember_proportions is not None:
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
        M_pmbr = Matrix([self.endmember_proportions]).T
        self.site_species_proportions = (self.endmember_site_occupancies.T
                                         * M_pmbr)

        A = self.endmember_site_occupancies.T
        M_alpha = Matrix([self.site_species_alphas]).T

        self.endmember_alphas = simplify_matrix(np.array(A.T*M_alpha))
        invalphap = Matrix([1./a for a in self.endmember_alphas[:]])

        B = diag(*M_alpha)*A*diag(*invalphap)
        try:
            B = simplify_matrix(np.array(B))
        except TypeError:
            pass

        f = 2. / (np.einsum('i, j -> ij', self.site_species_alphas,
                            np.ones(self.n_site_species))
                  + np.einsum('i, j -> ij', np.ones(self.n_site_species),
                              self.site_species_alphas))

        Wmod = emult(self.site_species_interactions, simplify_matrix(f))
        Q = B.T*Wmod*B

        self.endmember_energies = Matrix(np.ones(self.n_mbrs))

        self.endmember_interactions = Q[:, :]
        for i in range(self.n_mbrs):
            self.endmember_energies[i] = Q[i, i]*self.endmember_alphas[i]
            for j in range(i, self.n_mbrs):
                self.endmember_interactions[i, j] = ((Q[i, j] + Q[j, i]
                                                      - Q[i, i] - Q[j, j])
                                                     * (self.endmember_alphas[i]
                                                        + self.endmember_alphas[j])
                                                     / 2)
                self.endmember_interactions[j, i] = 0

        # Convert sympy objects to floats if possible
        # also normalise solution to self.n_sites
        normalise = symplify(self.n_sites)

        self.endmember_alphas /= normalise
        self.endmember_interactions *= normalise
        self.endmember_energies *= normalise

        self.endmember_alphas = vector_to_array(self.endmember_alphas)
        self.endmember_interactions = matrix_to_array(self.endmember_interactions)
        self.site_species_proportions = vector_to_array(self.site_species_proportions)
        self.endmember_energies = vector_to_array(self.endmember_energies)

    def set_alphas(self, alphas):
        """
        Sets the values of the site-species asymmetric parameters
        """
        assert len(alphas) == self.n_site_species
        self.site_species_alphas = alphas
        self._compute_properties()

    def set_composition(self, endmember_proportions):
        """
        Sets the endmember proportions
        """
        assert len(endmember_proportions) == self.n_mbrs
        self.endmember_proportions = np.array([symplify(f) for f in
                                               endmember_proportions])
        self._compute_properties()

    def set_site_species_interactions(self, site_species_interactions):
        """
        Sets the site interactions
        """
        assert len(site_species_interactions) == self.n_site_species
        assert len(site_species_interactions[0]) == self.n_site_species
        self.site_species_interactions = Matrix(site_species_interactions)
        self._compute_properties()

    def __str__(self):
        """
        Print the site occupancies and the
        macroscopic properties of the solution.
        """
        vE = Matrix([['U({0})'.format(self.mbrs[i])
                      for i in range(self.n_mbrs)]]).T

        Wb = Matrix([['W({0},{1})'.format(self.mbrs[i], self.mbrs[j])
                      if i < j else 0 for j in range(self.n_mbrs)]
                     for i in range(self.n_mbrs)])

        str = 'Site occupancies\n'
        for i in range(self.n_site_species):
            str += '{0}({1}) = {2}\n'.format(self.site_species[i],
                                             self.sites[i],
                                             self.site_species_proportions[i])
        str += '\nEndmember alphas\n'

        for i in range(self.n_mbrs):
            str += 'a({0}) = {1}\n'.format(self.mbrs[i],
                                           self.endmember_alphas[i])

        str += '\nEndmember interactions:\n'
        for i in range(self.n_mbrs):
            for j in range(self.n_mbrs):
                if (j > i):
                    # normalise removes alpha interactions
                    str += '{0} = {1}\n'.format(Wb[i, j],
                                                self.endmember_interactions[i,
                                                                            j])
        str += '\nEndmember free energies:\n'
        for i in range(len(vE)):
            str += '{0} = {1}\n'.format(vE[i], self.endmember_energies[i])

        for note in self.notes:
            str += note + '\n'

        return str

    def _interaction_energy(self, alphas, proportions, n, interactions, f):
        Ma = Matrix(alphas)
        if type(proportions) is not Matrix:
            Mp = Matrix([proportions]).T
        else:
            Mp = proportions
        ap = emult(Ma, Mp)
        a = 2. / (np.einsum('i, j -> ij', alphas, np.ones(n))
                  + np.einsum('i, j -> ij', np.ones(n), alphas))
        Wmod = emult(Matrix(interactions), simplify_matrix(a)) * f
        E_xs = ((ap.T * Wmod * ap) / (Mp.T * Ma))
        return np.array(E_xs)[0][0]

    @property
    def excess_energy(self):
        """
        Calculates the excess energy at a given composition
        """
        E_xs = self._interaction_energy(self.site_species_alphas,
                                        self.site_species_proportions,
                                        self.n_site_species,
                                        self.site_species_interactions,
                                        self.n_sites)
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
        E_xs2 = self._interaction_energy(self.endmember_alphas,
                                         self.endmember_proportions,
                                         self.n_mbrs,
                                         self.endmember_interactions,
                                         1.)

        assert np.abs(pG + E_xs2 - E_xs1) < 1.e-6
        return True


def microphi_from_file(filename):
    """
    Takes solution information from a file and creates an
    AsymmetricMicrophiSolution instance.
    """
    # Open the file and read the data
    with open(filename, 'r') as f:
        data = [line.split() for line in f if line[0] != '%']
        for i in range(len(data)):
            if '%' in data[i]:
                data[i] = data[i][:data[i].index('%')]

    # Assign data to appropriate variables
    mbrs = data[0]
    sites = data[1]
    alphas = np.array([float(f) for f in data[2]])
    site_species = data[3]
    endmember_site_occupancies = simplify_matrix(data[4:4+len(mbrs)])
    notes = [' '.join(d) for d in data[4+len(mbrs):]]

    # Return the microphi instance
    return AsymmetricMicrophiSolution(mbrs,
                                      sites,
                                      site_species,
                                      endmember_site_occupancies,
                                      alphas=alphas,
                                      notes=notes)
