import os
from sympy.parsing.mathematica import mathematica
#from sympy.functions.special.bessel import besselj

import numpy as np
import jax.numpy as jnp

from scipy.special import spherical_jn
from scipy.special import jv as besselj

from harmonix.harmonix import left_project
from harmonix.solution import CHSH_FT, transform_to_zernike, zernike_FT, j_to_nm

from jaxoplanet.experimental.starry.rotation import dot_rotation_matrix

def get_ylm_FTs():
    results = []
    with open(os.path.join(os.path.dirname(__file__), 'SphericalHarmonicsResults_10.txt'), 'r') as file:
        for line in file:
            # Split the line into columns based on tabs
            columns = line.strip().split('\t')
            l = int(columns[0])
            m = int(columns[1])
            result = mathematica(columns[2], {'BesselJ[n, rho]':'besselj(n, rho)'})
            results.append([l,m,result])
    replacements = {'pi':'jnp.pi',
                'cos':'jnp.cos',
                'sin':'jnp.sin',
                'sqrt':'jnp.sqrt',
                'I':'1j'}
    def replace_all(text, replacements):
        for i in replacements:
            text = text.replace(str(i), str(replacements[i]))
        return text
    jax_exprs = {}
    for l, m, res in results:
        jax_exprs[(l,m)]=replace_all(str(res), replacements)
    return jax_exprs

jax_funcs = get_ylm_FTs()

def v_scipy(l_max,u,v,inc,obl,theta,y):
    rho = jnp.sqrt(u**2 + v**2)
    phi = jnp.arctan2(v,u)
    
    #start with y, do Rdot (not dotR) for each step in v except in regular order (read from top down in order)
    #end with rotated y
    
    Ry = left_project(l_max, y, theta, inc, obl)
    
    ft = np.zeros_like(rho)
    y_hsh = []
    for l in range(l_max+1):
        for m in range(-l,l+1):
            #HSH
            if (l+m)%2==0:
                y_hsh.append(Ry[l**2+l+m])
            else:
                ft += CHSH_FT(l,m)(rho,phi)*Ry[l**2+l+m]
    y_hsh = np.array(y_hsh)
    #print(theta, x)
    zs = transform_to_zernike(y_hsh)
    for j in range(len(zs)):
        n,m = j_to_nm(j)
        ft += zs[j]*zernike_FT(n,m)(rho,phi)
    return ft
        
    
def v_mathematica(l_max, u,v,inc,obl,theta, y):
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
    rho = jnp.sqrt(u**2 + v**2)
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