import jax
from jaxoplanet.experimental.starry.wigner import dot_rotation_matrix
import jax.numpy as jnp
from .utils import get_ylm_FTs
from scipy.special import jn as besselj
from jax import config
config.update("jax_enable_x64", True)


jax_funcs = get_ylm_FTs()

def v(l_max, u,v,inc,obl,theta, y):
    """
    Computes the visibility function for a given spherical harmonic map
    represented using the spherical harmonic coefficient vector y.
    Rotates the star to an orientation on sky given by inclination (inc)
    obliquity (obl), and theta (phase). The visibility function returns 

    Args:
        u (float): _description_
        v (float): _description_
        inc (float): _description_
        obl (float): _description_
        theta (float): _description_
        y (array): _description_
    """
    rho = u**2 + v**2
    phi = jnp.arctan2(v,u)
    nmax = l_max**2 + 2 * l_max + 1
    fT = []
    for l in range(l_max+1):
        for m in range(-l,l+1):
            fT.append(eval(jax_funcs[(l,m)]))
    fT = jnp.array(fT).T
    
    #read from bottom to top to understand in order:
    
    #now rotate by the inclination, making sure to perform it along the original y axis (hence the cos and sin of obliquity)
    x = dot_rotation_matrix(
    l_max, -jnp.cos(obl), -jnp.sin(obl), 0.0, -(0.5 * jnp.pi - inc)
    )(fT)
    #rotate to the correct obliquity
    x = dot_rotation_matrix(l_max, None, None, 1.0, obl)(x)
    #rotate back to the equator-on view
    x = dot_rotation_matrix(l_max, 1.0, 0.0, 0.0, -0.5 * jnp.pi)(x)
    #rotate by theta to the correct phase, this is done for speed?
    x = dot_rotation_matrix(l_max, None, None, 1.0, theta)(x)
    #first rotate about the x axis 90 degrees so the vector pointing at the observer is now the pole
    x = dot_rotation_matrix(l_max, 1.0, 0.0, 0.0, 0.5 * jnp.pi)(x)

    return x @ y
    
    
    
    
    