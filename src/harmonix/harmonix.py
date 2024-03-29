import jax
from jaxoplanet.experimental.starry import Map, Ylm
from jaxoplanet.experimental.starry.rotation import dot_rotation_matrix
from jaxoplanet.experimental.starry.basis import A1
import jax.numpy as jnp
from jax import config
from jax import grad, vmap, jit

from .solution import transform_to_zernike, solution_vector
import numpy as np

from zodiax import Base
from functools import partial

config.update("jax_enable_x64", True)

lm_to_n = lambda l,m : l**2+l+m

class Harmonix(Base):
    map: Ylm
    hsh_inds: jnp.ndarray
    chsh_inds: jnp.ndarray
    data: jnp.ndarray

    def __init__(self, map):
        """Class for computation of interferometric observables from a spherical harmonic map

        Args:
            l_max (int): Maximum degree of the spherical harmonic map
            u (Array): U coordinates of the interferometer in nondimensional units
            v (_type_): V coordinates of the interferometer in nondimensional units
        TODO: 
        add limb darkening as a filter after rotation using the map*map feature
        """
        self.map = map
        self.data = map.y.todense()
        l_max = map.y.ell_max
        n_max = l_max**2 + 2 * l_max + 1
        hsh_mask = np.zeros(n_max, dtype=bool)
        
        for l in range(l_max+1):
            for m in range(-l,l+1):
                #HSH
                if (l+m)%2==0:
                    hsh_mask[lm_to_n(l,m)] = True
        self.hsh_inds, = jnp.nonzero(hsh_mask)
        self.chsh_inds, = jnp.nonzero(~hsh_mask)
        
    def rotational_phase(self, time):
        if self.map.period is None:
            return jnp.zeros_like(time)
        else:
            return 2 * jnp.pi * time / self.map.period
        
    @jit
    def model(self, u, v, t):
        rho = jnp.sqrt(u**2 + v**2)
        phi = jnp.arctan2(v,u)
        ft_hsh, ft_chsh = solution_vector(self.map.y.ell_max)(rho, phi)
        theta = self.rotational_phase(t)
        Ry = left_project(self.map.y.ell_max, self.data, theta, self.map.inc, self.map.obl)
        y_hsh = Ry[self.hsh_inds]
        y_chsh = Ry[self.chsh_inds]
        zernike_coeffs = transform_to_zernike(y_hsh)
        return (ft_hsh@zernike_coeffs + ft_chsh@y_chsh)/(rTA1(self.map.y.ell_max)@Ry)/np.sqrt(np.pi)*2
    

@partial(jit, static_argnums=0)
def left_project(deg, M, theta, inc, obl):
    # Note that here we are using the fact that R . M = (M^T . R^T)^T
    MT = jnp.transpose(M)

    # Rotate to the polar frame
    MT = dot_rotation_matrix(deg, 1.0, 0.0, 0.0, -0.5 * jnp.pi)(MT)
    MT = dot_rotation_matrix(deg, None, None, 1.0, -theta)(MT)

    # Rotate to the sky frame
    MT = dot_rotation_matrix(deg, 1.0, 0.0, 0.0, 0.5 * jnp.pi)(MT)
    MT = dot_rotation_matrix(deg, None, None, 1.0, -obl)(MT)
    MT = dot_rotation_matrix(
        deg, -jnp.cos(obl), -jnp.sin(obl), 0.0, 0.5 * jnp.pi - inc
    )(MT)

    return MT
    
    
def rT(lmax):
    rt = [0 for _ in range((lmax + 1) * (lmax + 1))]
    amp0 = jnp.pi
    lfac1 = 1.0
    lfac2 = 2.0 / 3.0
    for ell in range(0, lmax + 1, 4):
        amp = amp0
        for m in range(0, ell + 1, 4):
            mu = ell - m
            nu = ell + m
            rt[ell * ell + ell + m] = amp * lfac1
            rt[ell * ell + ell - m] = amp * lfac1
            if ell < lmax:
                rt[(ell + 1) * (ell + 1) + ell + m + 1] = amp * lfac2
                rt[(ell + 1) * (ell + 1) + ell - m + 1] = amp * lfac2
            amp *= (nu + 2.0) / (mu - 2.0)
        lfac1 /= (ell / 2 + 2) * (ell / 2 + 3)
        lfac2 /= (ell / 2 + 2.5) * (ell / 2 + 3.5)
        amp0 *= 0.0625 * (ell + 2) * (ell + 2)

    amp0 = 0.5 * jnp.pi
    lfac1 = 0.5
    lfac2 = 4.0 / 15.0
    for ell in range(2, lmax + 1, 4):
        amp = amp0
        for m in range(2, ell + 1, 4):
            mu = ell - m
            nu = ell + m
            rt[ell * ell + ell + m] = amp * lfac1
            rt[ell * ell + ell - m] = amp * lfac1
            if ell < lmax:
                rt[(ell + 1) * (ell + 1) + ell + m + 1] = amp * lfac2
                rt[(ell + 1) * (ell + 1) + ell - m + 1] = amp * lfac2
            amp *= (nu + 2.0) / (mu - 2.0)
        lfac1 /= (ell / 2 + 2) * (ell / 2 + 3)
        lfac2 /= (ell / 2 + 2.5) * (ell / 2 + 3.5)
        amp0 *= 0.0625 * ell * (ell + 4)
    return np.array(rt)

def rTA1(lmax):
    return rT(lmax) @ A1(lmax)

@partial(vmap, in_axes=(0, None, None, None))
@partial(vmap, in_axes=(1, None, None, None))
def cp_from_cvis(vis, index_cps1, index_cps2, index_cps3):
    '''
    Calculate closure phases [degrees] from complex visibilities and cp indices

    vis: complex visibilities
    index_cps1, index_cps2, index_cps3: indices for closure phases (e.g. [0,1,2] for 1st 3-baseline closure phase)

    Returns: closure phases [degrees]

    '''
    real = jnp.real(vis)
    imag = jnp.imag(vis)
    visphiall = jnp.arctan2(imag,real)
    visphiall = jnp.mod(visphiall + 10980., 360.)-180.
    visphi = jnp.reshape(visphiall,(len(vis),1))
    cp = visphi[jnp.array(index_cps1)] + visphi[jnp.array(index_cps2)] - visphi[jnp.array(index_cps3)]
    out = jnp.reshape(cp*180/np.pi,len(index_cps1))
    return out

def visibilities(harmonix_map, u, v, t):
    """Takes an array of u, v and times and returns the visibility amplitude where

    Args:
        harmonix_map (Harmonix): Harmonix map object
        u (jnp.array): N_baselines x N_samples array of u coordinates
        v (jnp.array): N_baselines x N_samples array of v coordinates
        t (float): (temporary) float of time t

    Returns:
        jnp.array: N_samples x N_baselines array of visibility amplitudes
    """
    vmap_model = vmap(vmap(harmonix_map.model, in_axes=(1, 1, None)), in_axes=(None, None, 0))
    return jnp.swapaxes(jnp.abs(vmap_model(u, v, t)),1,2)

def closure_phases(harmonix_map, u, v, t, index_cps1, index_cps2, index_cps3):
    """Takes an array of u, v and times and returns the closure phases

    Args:
        harmonix_map (Harmonix): Harmonix map object
        u (jnp.array): N_baselines x N_samples array of u coordinates
        v (jnp.array): N_baselines x N_samples array of v coordinates
        t (float): (temporary) float of time t
        index_cps1 (jnp.array): N_closure_phases array of indices
        index_cps2 (jnp.array): N_closure_phases array of indices
        index_cps3 (jnp.array): N_closure_phases array of indices

    Returns:
        jnp.array: N_samples x N_baselines array of complex visibilities
    """
    vmap_model = vmap(vmap(harmonix_map.model, in_axes=(1, 1, None)), in_axes=(None, None, 0))
    vis = jnp.swapaxes(vmap_model(u, v, t), 1,2)
    return jnp.swapaxes(cp_from_cvis(vis, index_cps1, index_cps2, index_cps3), 1,2)