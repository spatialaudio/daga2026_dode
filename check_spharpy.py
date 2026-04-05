# see https://github.com/spatialaudio/daga2026_dode

# check that utilised spharpy & pyfar functions work as intended
# we could/should have used Python's errors and exceptions handling
# but we lazily go for simple prints of True/False

import numpy as np
import pyfar
import spharpy
from pyfar.classes.coordinates import cart2sph
from spharpy.samplings import dodecahedron, t_design

print(spharpy.__version__)  # we tested 1.0.0
print(pyfar.__version__)  # we tested 0.8.0

# check the dode, we somewhere in the code rely on certain
# directions of the dode, so make sure they are as intended
sph_sampling = dodecahedron()
e = np.array([0.607062, -0.303531, -0.303531,  0.49112347, -0.98224695,
              0.49112347, 0.98224695, -0.49112347, -0.49112347, 0.303531,
              -0.607062, 0.303531]) - sph_sampling.x
print(np.allclose(np.inner(e, e), 0))
e = np.array([0., 0.52573111, -0.52573111, 0.85065081,  0.,
              -0.85065081, 0., 0.85065081, -0.85065081,  0.52573111,
              0., -0.52573111]) - sph_sampling.y
print(np.allclose(np.inner(e, e), 0))
e = np.array([0.79465447, 0.79465447, 0.79465447, 0.18759247, 0.18759247,
              0.18759247, -0.18759247, -0.18759247, -0.18759247, -0.79465447,
              -0.79465447, -0.79465447]) - sph_sampling.z
print(np.allclose(np.inner(e, e), 0))

sph_sampling = t_design(n_max=22, criterion='const_angular_spread')
Theta = np.array([sph_sampling.colatitude, sph_sampling.azimuth]).T
print(np.allclose(Theta.shape[0], 1059))
print(np.allclose(Theta.shape[1], 2))
print(np.allclose(Theta[0, :],
                  np.array([0, 0])))
print(np.allclose(Theta[-1, :],
                  np.array([3.06734952, 4.93709158])))

# we work with polar angle = theta / azimuth angle = phi
# hence check that phi, theta, radius = cart2sph(x,y,z) does this as well
print(np.allclose(cart2sph(0, 0, +1),
                  np.array([0, 0, 1])))
print(np.allclose(cart2sph(0, 0, -1),
                  np.array([0, np.pi, 1])))
print(np.allclose(cart2sph(0, 1, 0),
                  np.array([np.pi/2, np.pi/2, 1])))
print(np.allclose(cart2sph(-2, 0, 0),
                  np.array([np.pi, np.pi/2, 2])))
print(np.allclose(cart2sph(1, 1, 1),
                  np.array([np.pi/4, np.acos(1/np.sqrt(3)), np.sqrt(3)])))
