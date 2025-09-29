import jax
from jaxoplanet.starry import Ylm
from jaxoplanet.starry.core.rotation import dot_rotation_matrix, left_project
from jaxoplanet.starry.core.basis import A1, U
from jaxoplanet.starry.core import solution
from jaxoplanet.starry.core.polynomials import Pijk
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
    surface: Ylm
    radius: jnp.ndarray
    hsh_inds: jnp.ndarray
    chsh_inds: jnp.ndarray
    data: jnp.ndarray
    u: jnp.ndarray

    def __init__(self, surface, radius):
        """Class for computation of interferometric observables from a spherical harmonic map

        Args:
            surface (jaxoplanet.starry.surface): The spherical harmonic map of the surface
            radius (float): The radius of the star in milliarcseconds
        """
        self.surface = surface
        self.data = surface.y.todense()[1:]
        self.u = jnp.array(surface.u)
        self.radius = radius
        l_max = surface.deg
        n_max = l_max**2 + 2 * l_max + 1
        hsh_mask = np.zeros(n_max, dtype=bool)
        
        
        # yu = jnp.array(np.linalg.inv(A1(udeg).todense())) @ pu
        # yu = Ylm.from_dense(yu.flatten(), normalize=False)
        # norm = 1 / (Pijk.from_dense(pu, degree=udeg).todense() @ solution.rT(udeg))
        # yu = yu * norm
        # self.yu = yu
        
        for l in range(l_max+1):
            for m in range(-l,l+1):
                #HSH
                if (l+m)%2==0:
                    hsh_mask[lm_to_n(l,m)] = True
        self.hsh_inds, = jnp.nonzero(hsh_mask)
        self.chsh_inds, = jnp.nonzero(~hsh_mask)
        
    def rotational_phase(self, time):
        if self.surface.period is None:
            return jnp.zeros_like(time)
        else:
            return 2 * jnp.pi * time / self.surface.period
        
    def model(self, u, v, t):
        """Computes the complex visibility of the star at a given time t

        Args:
            u (array): U coordinates of the baselines in dimentionless units (x baseline length / wavelength)
            v (array): V coordinates of the baselines in dimentionless units (y baseline length / wavelength)
            t (float): time in days relative to the reference epoch 

        Returns:
            array (complex128): complex visibility amplitudes of the star at time t
        """
        
        mas2rad = jnp.pi / 180.0 / 3600.0/ 1000.0
        u_scaled = u * self.radius * mas2rad * 2 * jnp.pi
        v_scaled = v * self.radius * mas2rad * 2 * jnp.pi
        rho = jnp.sqrt(u_scaled**2 + v_scaled**2)
        phi = jnp.arctan2(v_scaled,u_scaled)
        ft_hsh, ft_chsh = solution_vector(self.surface.deg)(rho, phi)
        theta = self.rotational_phase(t)
        #start by rotating the map (before adding the limb darkening filter)
        Ry = left_project(self.surface.ydeg, self.surface.inc, self.surface.obl, theta, 0.0, 
                          jnp.concatenate([jnp.array([1.0,]),self.data]))

        # limb darkening
        if self.surface.udeg == 0:
            pu = Pijk.from_dense(jnp.array([1]))
        else:
            u = jnp.array([1, *self.u])
            pu = Pijk.from_dense(u @ U(self.surface.udeg), degree=self.surface.udeg)
        #convert the map to the polynomial basis
        p_y = Pijk.from_dense(A1(self.surface.ydeg).todense() @ Ry, degree=self.surface.ydeg)
        #multiply the map by the limb darkening map in polynomial basis
        p_yu = p_y * pu
        p_yu = p_yu.todense().flatten()
        #don't know why I need this normalization term but I got it from jaxoplanet.starry
        norm = np.pi / (pu.tosparse() @ rT(self.surface.udeg))
        #convert back into spherical harmonic basis
        Ry_mul = self.surface.amplitude * (np.array(np.linalg.inv(A1(self.surface.deg).todense())) @ p_yu) * norm
        
        y_hsh = Ry_mul[self.hsh_inds]
        y_chsh = Ry_mul[self.chsh_inds]
        zernike_coeffs = transform_to_zernike(y_hsh)
        return (ft_hsh@zernike_coeffs + ft_chsh@y_chsh)/(rTA1(self.surface.deg)@Ry_mul)/np.sqrt(np.pi)*2
    

    
    
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