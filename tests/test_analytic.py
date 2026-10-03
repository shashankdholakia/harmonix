
from jax import config

config.update("jax_enable_x64", True)

import pytest
import jax
from jax import jit
import jax.numpy as jnp
from jaxoplanet.starry import Ylm, Surface
from harmonix.harmonix import Harmonix
from harmonix.solution import solution_vector, transform_to_zernike
from harmonix.test_utils import get_ylm_FTs, v_scipy, v_mathematica

from harmonix.utils import apply_DFTM1, compute_DFTM1
from harmonix.harmonix import rTA1
from scipy.special import j1
import numpy as np


mas2rad = jnp.pi / 180.0 / 3600.0/ 1000.0

def airy(w, lam, diam):
    '''Airy function for a circular aperture, evaluated on baselines uv (m) with diameter diam (mas) at wavelength lam (m)'''
    
    r = w/lam

    d = diam*mas2rad

    return 2 * j1(jnp.pi * r * d) / (jnp.pi * r * d)
@pytest.mark.parametrize("radius", [1.0, 2.0, 3.0])
def test_harmonix_vs_analytic(radius):
    
    """Test harmonix against an analytic solution for a uniform disk.
    Takes a radius in milliarcseconds, and computes the Fourier transform of a uniform disk."""
    
    u, v = np.linspace(-500,500,64), np.linspace(-500,500,64)
    wavel = 2e-6 # m

    uu, vv = np.meshgrid(np.linspace(-500,500,64),np.linspace(-500,500,64))
    uvgrid = np.vstack((uu.flatten(),vv.flatten())).T
    # Define the spherical harmonic map
    ylm = Ylm.from_dense(jnp.array([1.0]))
    star = Surface(y=ylm, inc=0., obl=0, period=1.0)
    # Time doesn't matter for a uniform map
    t = 0.0
    cvis = Harmonix(star, radius).model(uvgrid[:,0]/wavel, uvgrid[:,1]/wavel, t)
    
    wgrid = np.sqrt(uvgrid[:,0]**2 + uvgrid[:,1]**2)
    ft_anal = airy(np.sort(wgrid), wavel,2*radius)
    v2_anal, phase_anal = jnp.abs(ft_anal)**2, jnp.angle(ft_anal)
    v2, phase = jnp.abs(cvis)**2, jnp.angle(cvis)
    inds = np.argsort(wgrid)
    assert jnp.allclose(v2[inds], v2_anal, rtol=1e-10), "Harmonix model does not match analytic solution for visibility squared"
    assert jnp.allclose(phase[inds], phase_anal, rtol=1e-7), "Harmonix model does not match analytic solution for phase"

@pytest.mark.parametrize("u", [jnp.array([0.1, 0.1])])
def test_harmonix_vs_analytic_ld(u):
    """
    TODO: Test harmonix against an analytic solution for a uniform disk with limb darkening.
    """
    pass
    
#construct a dictionary for each spherical harmonic coefficient
#where just that coefficient is 1.0 and all others are 0.0
#this is useful to test each harmonic individually
l_max = 4
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

@pytest.mark.parametrize(("lm", "y")
                         ,ylm_coeffs)
def test_jax_vs_scipy(lm, y):
    l, m = lm
    mas2rad = np.pi / 180.0 / 3600.0/ 1000.0
    u, v = jnp.linspace(0.2,10,300),jnp.linspace(0.01,.01,300)
    scipy_test = v_scipy(l,u,v,np.pi/2,0.0,0,y)/np.sqrt(np.pi)*2/(rTA1(l)@y)
    ylm = Ylm.from_dense(y)
    star = Surface(y=ylm, inc=jnp.pi/2, obl=0.0, period=jnp.inf)
    jax_test = Harmonix(star, 1.0).model(u/mas2rad/2./np.pi, v/mas2rad/2./np.pi, 0.0)
    assert jnp.allclose(scipy_test, jax_test, rtol=1e-7), f"JAX and SciPy results do not match for l={l}, m={m}"
    
@pytest.mark.parametrize(("lm", "y")
                         ,ylm_coeffs) 
def test_jax_vs_mathematica(lm, y):
    l, m = lm
    mas2rad = np.pi / 180.0 / 3600.0/ 1000.0

    u, v = jnp.linspace(0.2,10,300),jnp.linspace(0.01,.01,300)
    mathematica_test = v_mathematica(l,u,v,np.pi/2,0.0,0,y)/jnp.sqrt(np.pi)*2/(rTA1(l)@y)
    ylm = Ylm.from_dense(y)
    star = Surface(y=ylm, inc=jnp.pi/2, obl=0.0, period=jnp.inf)
    jax_test = Harmonix(star, 1.0).model(u/mas2rad/2./np.pi, v/mas2rad/2./np.pi, 0.0)
    assert jnp.allclose(mathematica_test, jax_test, rtol=1e-5), f"JAX and Mathematica results do not match for l={l}, m={m}"


def test_basis_fts_match_direct_integration():
    """
    Test the Fourier transform of every Ylm up to l = 4 against direct
    numerical integration of jaxoplanet's intensity map over the stellar disk,
    including zero baseline. This needs no external reference solution.
    """
    from jaxoplanet.starry.core.basis import A1, poly_basis

    l_max = 4
    n = (l_max + 1) ** 2
    modes = [(l, m) for l in range(l_max + 1) for m in range(-l, l + 1)]
    hsh = [j for j, (l, m) in enumerate(modes) if (l + m) % 2 == 0]
    chsh = [j for j, (l, m) in enumerate(modes) if (l + m) % 2 == 1]

    # Gauss-Legendre in s, with r = sin(s) so that z = cos(s) is smooth at
    # the limb, and the trapezoid rule in azimuth.
    s, ws = np.polynomial.legendre.leggauss(40)
    s, ws = 0.25 * np.pi * (s + 1), 0.25 * np.pi * ws
    t = 2 * np.pi * np.arange(80) / 80
    S, T = np.meshgrid(s, t, indexing="ij")
    x, y, z = np.sin(S) * np.cos(T), np.sin(S) * np.sin(T), np.cos(S)
    weights = (ws[:, None] * (2 * np.pi / 80) * np.sin(S) * np.cos(S)).ravel()
    intensity = np.asarray(poly_basis(l_max)(x.ravel(), y.ravel(), z.ravel()))
    intensity = intensity @ np.array(A1(l_max).todense())

    rho = np.linspace(0.0, 12.0, 25)
    phi = np.linspace(-3.0, 3.0, 25)
    kernel = np.exp(-1j * rho[:, None] * (np.cos(phi)[:, None] * x.ravel()
                                          + np.sin(phi)[:, None] * y.ravel()))
    # harmonix's basis carries a factor of sqrt(pi) / 2 relative to the
    # plain Fourier integral; Harmonix.model divides it out again.
    expected = np.sqrt(np.pi) / 2 * kernel @ (weights[:, None] * intensity)

    ft_hsh, ft_chsh = solution_vector(l_max)(jnp.asarray(rho), jnp.asarray(phi))
    for j, (l, m) in enumerate(modes):
        if j in hsh:
            e = np.zeros(len(hsh))
            e[hsh.index(j)] = 1.0
            ft = ft_hsh @ transform_to_zernike(e)
        else:
            ft = ft_chsh[:, chsh.index(j)]
        assert np.allclose(ft, expected[:, j], rtol=0, atol=1e-12), f"l={l}, m={m}"


def test_zero_baseline_is_finite_with_finite_gradients():
    ylm = Ylm.from_dense(jnp.array([1.0, 0.2, -0.1, 0.3, 0.1, 0.0, 0.2, -0.2, 0.1]))
    star = Surface(y=ylm, inc=1.2, obl=0.3, period=1.0, u=[0.4, 0.2])
    u = jnp.array([0.0, 1e3, 5e7])
    v = jnp.array([0.0, -1e3, 1e7])
    cvis = Harmonix(star, 1.0).model(u, v, 0.1)
    assert jnp.allclose(cvis[0], 1.0, atol=1e-12)
    assert jnp.allclose(cvis[1], 1.0, atol=1e-5)
    grad = jax.grad(lambda r: jnp.sum(jnp.abs(Harmonix(star, r).model(u, v, 0.1)) ** 2))(1.0)
    assert jnp.isfinite(grad)
