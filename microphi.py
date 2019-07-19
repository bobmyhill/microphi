import numpy as np
from sympy import Matrix, Rational, diag
from collections import OrderedDict

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
    endmember_gibbs: vector
        Endmember gibbs free energies contributed by the site interactions
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
            self.alphas = alphas
        else:
            self.alphas = ['a({0},{1})'.format(self.sites[i],
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
            self.endmember_proportions = ['x({0})'.format(mbr)
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

        M_alpha = Matrix([self.alphas]).T

        self.alpha_prime = A.T*M_alpha
        invalphap = Matrix([1./a for a in self.alpha_prime[:]])

        B = diag(*M_alpha)*A*diag(*invalphap)
        try:
            B = simplify_matrix(np.array(B))
        except TypeError:
            pass

        for i in range(self.n_site_species):
            for j in range(i, self.n_site_species):
                self.site_species_interactions[i, j] *= 2/(M_alpha[i]
                                                           + M_alpha[j])

        Q = B.T*self.site_species_interactions*B

        self.endmember_gibbs = Matrix(np.ones(self.n_mbrs))

        self.endmember_interactions = Q[:, :]
        for i in range(self.n_mbrs):
            self.endmember_gibbs[i] = Q[i, i]*self.alpha_prime[i]
            for j in range(i, self.n_mbrs):
                self.endmember_interactions[i, j] = ((Q[i, j] + Q[j, i]
                                                      - Q[i, i] - Q[j, j])
                                                     * (self.alpha_prime[i]
                                                        + self.alpha_prime[j])
                                                     / 2)
                self.endmember_interactions[j, i] = 0

        # Convert sympy objects to floats if possible
        # also normalise solution to self.n_sites
        normalise = self.n_sites

        self.alpha_prime /= normalise
        self.endmember_interactions *= normalise
        self.endmember_gibbs *= normalise

        self.alpha_prime = vector_to_array(self.alpha_prime)
        self.endmember_interactions = matrix_to_array(self.endmember_interactions)
        self.site_species_proportions = vector_to_array(self.site_species_proportions)
        self.endmember_gibbs = vector_to_array(self.endmember_gibbs)

    def set_alphas(self, alphas):
        """
        Sets the values of the site-species asymmetric parameters
        """
        assert len(alphas) == self.n_site_species
        self.alphas = alphas
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
        vG = Matrix([['G({0})'.format(self.mbrs[i])
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
                                           self.alpha_prime[i])

        str += '\nEndmember interactions:\n'
        for i in range(self.n_mbrs):
            for j in range(self.n_mbrs):
                if (j > i):
                    # normalise removes alpha interactions
                    str += '{0} = {1}\n'.format(Wb[i, j],
                                                self.endmember_interactions[i,
                                                                            j])
        str += '\nEndmember Gibbs free energies:\n'
        for i in range(len(vG)):
            str += '{0} = {1}\n'.format(vG[i], self.endmember_gibbs[i])

        for note in self.notes:
            str += note + '\n'

        return str

    @property
    def excess_energy(self):
        """
        Calculates the excess energy at a given composition
        """
        ap = self.alphas * self.site_species_proportions
        Wmod = self.site_species_interactions*self.n_sites
        E_xs = (ap.dot(Wmod).dot(ap)
                / (self.site_species_proportions.dot(self.alphas)))
        return E_xs

    def check_formalism(self):
        """
        This function calculates the Gibbs free energy at a
        given composition using the microscopic (site) and
        macroscopic (endmember), and makes sure that they
        produce the same answer.
        """
        Ea = self.excess_energy

        app = self.alpha_prime * self.endmember_proportions
        pG = self.endmember_proportions.dot(self.endmember_gibbs)
        Eb = (pG + (app.dot(2. * self.endmember_interactions
                            / (np.einsum('i, j -> ij', self.alpha_prime,
                                         np.ones(self.n_mbrs))
                               + np.einsum('i, j -> ij', np.ones(self.n_mbrs),
                                           self.alpha_prime))).dot(app))
              / self.alpha_prime.dot(self.endmember_proportions))
        assert np.abs(Ea-Eb) < 1.e-6
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
