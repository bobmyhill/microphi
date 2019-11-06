import numpy as np
from microphi import microphi_from_file

print('Symbolic output only:')
ss = microphi_from_file('input_files/simple_od.dat')
print(ss)


print('After assigning endmember proportions:')
p_mbr = np.array([0.1, 0.2, 0.7])
ss.set_composition(p_mbr)
print(ss)
