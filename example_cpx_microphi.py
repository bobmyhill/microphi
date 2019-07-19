import numpy as np
from microphi import AsymmetricMicrophiSolution


# Endmember names
mbrs = ['di', 'hed', 'cen', 'cfs']

# Site species information
sites = ['M1', 'M1', 'M1', 'M2', 'M2']
site_species = ['Mg', 'Fe', 'Ca', 'Mg', 'Fe']
alphas = [1, 1, 1.4, 1, 1]

endmember_site_occupancies = np.array([[0, 0, 1, 1, 0],
                                       [0, 0, 1, 0, 1],
                                       [1, 0, 0, 1, 0],
                                       [0, 1, 0, 0, 1]])

#site_species_interactions = np.zeros((5, 5))
#site_species_interactions[0][1] = 4
#site_species_interactions[3][4] = 4
#site_species_interactions[1][4] = 1

ss = AsymmetricMicrophiSolution(mbrs, sites, site_species,
                                endmember_site_occupancies,
                                alphas)  #,
#                                site_species_interactions=site_species_interactions)

print(ss)
exit()
print('deltaE(ofm):', (ss.endmember_energies[4]
                       - 0.5*(ss.endmember_energies[0]
                              + ss.endmember_energies[1])))


print('W(oen, ofs):', ss.endmember_interactions[0,1])
print('W(oen, ofm):', ss.endmember_interactions[0,4])
print('W(ofs, ofm):', ss.endmember_interactions[1,4])
