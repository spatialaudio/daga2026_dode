# see https://github.com/spatialaudio/daga2026_dode

import papermill as pm
from pathlib import Path
import subprocess

# flag = 'notebooks'
flag = 'scripts'

if flag == 'notebooks':
    for nb in Path('').glob('*.ipynb'):
        print(nb)
        pm.execute_notebook(
            input_path=nb,
            output_path='executed_ipynb' / nb
        )
elif flag == 'scripts':
    # we could/should have used Python's errors and exceptions handling
    # but we lazily go for simple prints of True/False on the check files:
    # check that utilised spharpy functions work as intended:
    subprocess.call(['python', 'check_spharpy.py'])
    # check that Ynm and Pnm work as intended:
    subprocess.call(['python', 'check_Ynm_Pnm.py'])

    # Fig. 1 paper/poster:
    subprocess.call(['python', 'radial_filter_ground_truth.py'])
    # Fig. 2 paper/poster:
    subprocess.call(['python', 'radial_filter_bank.py'])
    # Fig. 3 paper:
    subprocess.call(['python', 'dodecahedron_vn.py'])
    # Fig. 4a paper, Fig. 3a poster:
    subprocess.call(['python', 'dodecahedron_sphere.py'])
    # Fig. 4b paper, Fig. 3b poster:
    subprocess.call(['python', 'dodecahedron_pressure_irs.py'])

    # not in paper / poster but might be helpful:
    subprocess.call(['python', 'dipole_velocity_from_two_caps.py'])
else:
    print('nothing to do')
