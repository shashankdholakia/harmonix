from jax import config

config.update("jax_enable_x64", True)

import pytest
import jax
from jax import jit
import jax.numpy as jnp
from jaxoplanet.starry import Ylm, Surface
from harmonix.harmonix import Harmonix
from harmonix.solution import solution_vector, transform_to_zernike
from harmonix.utils import apply_DFTM1, compute_DFTM1
from scipy.special import j1
import numpy as np

mas2rad = jnp.pi / 180.0 / 3600.0/ 1000.0

#construct a dictionary for each spherical harmonic coefficient
#where just that coefficient is 1.0 and all others are 0.0
#this is useful to test each harmonic individually
l_max = 2
n_max = lambda l_max: l_max**2 + 2 * l_max + 1
ylm_coeffs = []
ind = 0
for l in range(l_max+1):
    for m in range(-l,l+1):
        coeffs = np.concatenate([np.array([1.0]), 
                                np.zeros(n_max(l) - 1)])
        coeffs[ind] = 1.0
        ylm_coeffs.append(((l,m),coeffs))
        ind+=1


@pytest.mark.parametrize(("lm", "coeffs")
                         ,ylm_coeffs)
def test_harmonix_vs_bruteforce_ylms(lm, coeffs):
    
    """
    Test harmonix against a brute force 2D Fourier transform of the surface map.
    Uses the intensity function in jaxoplanet.starry.surface to compute the
    pixel intensities, and then performs a 2D Fourier transform at each time step.
    """
    l, m = lm
    radius = 1.0 #radius in milliarcseconds
    u, v = np.linspace(-500,500,64), np.linspace(-500,500,64)
    wavel = 2e-6 # m
    uu, vv = np.meshgrid(u,v)
    uvgrid = np.vstack((uu.flatten(),vv.flatten())).T
    # Define the spherical harmonic map
    ylm = Ylm.from_dense(coeffs)
    star = Surface(y=ylm, inc=jnp.radians(90.), obl=0.0, period=1.0)
    res = 100
    x, y = jnp.meshgrid(jnp.linspace(-1, 1, res), jnp.linspace(-1, 1, res))
    image = star.render(res=res,theta=jnp.radians(0.))
    image = jnp.nan_to_num(image, nan=0.0)
    
    dftm = compute_DFTM1(x, y, uvgrid, wavel)
    cvis_dftm = apply_DFTM1(image, dftm)
    
    cvis = Harmonix(star).model(radius*mas2rad*2*jnp.pi*uvgrid[:,0]/wavel, radius*mas2rad*2*jnp.pi*uvgrid[:,1]/wavel, 0.0)
    v2, phase = jnp.abs(cvis)**2, jnp.angle(cvis)
    v2_dftm, phase_dftm = jnp.abs(cvis_dftm)**2, jnp.angle(cvis_dftm)
    assert jnp.allclose(v2, v2_dftm, rtol=3e-2, atol=1e-3), "Harmonix model does not match brute force DFT for visibility squared"
    
@pytest.mark.parametrize(("time"), np.linspace(0., 1., 10))
def test_harmonix_vs_bruteforce_earth_map(time):
    """
    Test harmonix against a brute force 2D Fourier transform of the Earth map.
    Uses the intensity function in jaxoplanet.starry.surface to compute the
    pixel intensities, and then performs a 2D Fourier transform at each time step.
    """
    coeffs = jnp.array([1.00,  0.22,  0.19,  0.11,  0.11,  0.07,  -0.11, 0.00,  -0.05, 
     0.12,  0.16,  -0.05, 0.06,  0.12,  0.05,  -0.10, 0.04,  -0.02, 
     0.01,  0.10,  0.08,  0.15,  0.13,  -0.11, -0.07, -0.14, 0.06, 
     -0.19, -0.02, 0.07,  -0.02, 0.07,  -0.01, -0.07, 0.04,  0.00])
    
    radius = 1.0 #radius in milliarcseconds
    u, v = np.linspace(-500,500,32), np.linspace(-500,500,32)
    wavel = 2e-6 # m
    uu, vv = np.meshgrid(u,v)
    uvgrid = np.vstack((uu.flatten(),vv.flatten())).T
    # Define the spherical harmonic map
    ylm = Ylm.from_dense(coeffs)
    star = Surface(y=ylm, inc=jnp.radians(90.), obl=0.0, period=1.0)
    res = 400
    x, y = jnp.meshgrid(jnp.linspace(-1, 1, res), jnp.linspace(-1, 1, res))
    image = star.render(res=res,theta=star.rotational_phase(time))
    image = jnp.nan_to_num(image, nan=0.0)
    
    dftm = compute_DFTM1(x, y, uvgrid, wavel)
    cvis_dftm = apply_DFTM1(image, dftm)
    
    cvis = Harmonix(star).model(radius*mas2rad*2*jnp.pi*uvgrid[:,0]/wavel, radius*mas2rad*2*jnp.pi*uvgrid[:,1]/wavel, time)
    v2, phase = jnp.abs(cvis)**2, jnp.angle(cvis)
    v2_dftm, phase_dftm = jnp.abs(cvis_dftm)**2, jnp.angle(cvis_dftm)
    assert jnp.allclose(v2, v2_dftm, rtol=3e-2, atol=1e-3), "Harmonix model does not match brute force DFT for visibility squared"