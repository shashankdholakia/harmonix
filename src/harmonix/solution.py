import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from scipy.special import gamma, factorial, comb, factorial2, rgamma
from scipy.special import jv, spherical_jn
from .utils import csphjy

def KroneckerDelta(i,j):
    #does numpy or scipy not have a convenience function for the Kronecker Delta???
    return int(i==j)

def A(l, m):
    """A spherical harmonic normalization constant."""
    return np.sqrt((2 - KroneckerDelta(m, 0)) * (2 * l + 1) * factorial(l - m) / (4 * np.pi * factorial(l + m)))

###############################
#HEMISPHERIC HARMONICS SOLUTION
###############################
def j_to_nm(j):
    n = int(np.ceil((-3 + np.sqrt(9 + 8 * j)) / 2))
    m = int(2 * j - n * (n + 2))
    return n,m
def nm_to_j(n,m):
    return int((n*(n+2)+m)/2)

jmax = lambda nmax: (nmax**2+3*nmax)//2

def Z(n,m):
    """A Zernike normalization constant"""
    #Using OSI/ANSI scheme as with the rest
    return np.sqrt(2*(n+1)/(1+KroneckerDelta(m,0)))

def z_to_p(j, jmax):
    """ returns the jth term of the zernike to polynomial basis"""
    n,m = j_to_nm(j)
    res = np.zeros(jmax+1)
    for s in range(0,int((n-np.abs(m))/2)+1):
        #this particular term of the polynomial has r^(n-2*s), that means we drop the r and instead add this to the array for n=n-2*s and m=m
        #not using the normalization constant, so make sure to reproduce that in the Fourier basis too
        res[nm_to_j(int(n-2*s), m)] += (-1)**s * factorial(n-s) / (factorial(s) * factorial((n+np.abs(m))/2 -s) * factorial((n-np.abs(m))/2 - s))
    return res

def hsh_to_p(n, nmax):
    """ returns the jth term of the HSH to polynomial basis"""
    l,m = j_to_nm(n)
    res = np.zeros(nmax+1)
    for k in range(0,l-np.abs(m)+1):
        for j in range(0,int(k/2)+1):
            #this particular term of the polynomial has r^(2*j+np.abs(m)), that means we drop the r and instead add this to the array for n=n-2*s and m=m
            #adding the normalization factor A(l,m) here to avoid having to add it later on: NOT REPRESENTED IN BasisTransforms.ipynb
            #version 1.15.0 of scipy returns nan instead of inf for gamma of negative integers,
            #so we instead multiply by rgamma to avoid division by nan
            res[nm_to_j(int(2*j+np.abs(m)),m)] += A(l,np.abs(m))* (-1)**(l) * 2**l * (gamma((l+np.abs(m)+k-1)/2 +1)*rgamma((-l+np.abs(m)+k-1)/2+1)/(factorial(k) * factorial(l-np.abs(m)-k))) * comb(k/2,j) * (-1)**j
    return res

def A_z_to_p(jmax):
    A = np.zeros((jmax+1,jmax+1))
    for j in range(jmax+1):
        A[j] = z_to_p(j, jmax)
    return A

def A_hsh_to_p(jmax):
    A = np.zeros((jmax+1,jmax+1))
    for j in range(jmax+1):
        A[j] = hsh_to_p(j, jmax)
    return A

def transform_to_zernike(hsh_vector):
    """Takes a vector in the hemispheric harmonic basis and transforms it into a vector in the zernike basis

    Args:
        hsh_vector (ndarray): 1d vector of coefficients in the hemispheric harmonic basis
        *must have length corresponding to jmax(n) where n is an integer!
    """
    jmax = len(hsh_vector)-1
    A = np.linalg.inv(A_z_to_p(jmax)).T@A_hsh_to_p(jmax).T
    return A@hsh_vector


def zernike_FT(n,m):
    #HAVE TO MULTIPLY BY A(l,m) ALSO
    def bessel(rho,phi):
        angular = np.where(m>=0,np.cos(m*phi), np.sin(-m*phi))
        #someone who is good at signs please help me simplify this. my family is dying
        res = (-1)**(n/2-KroneckerDelta(m,0)+1+n) * 2*np.pi*jv(n+1, rho) * angular / rho
        return res
    return bessel

def zernike_FT_jax(n,m, Jvs):
    #HAVE TO MULTIPLY BY A(l,m) ALSO
    def bessel(rho,phi):
        angular = jnp.where(m>=0,jnp.cos(m*phi), jnp.sin(-m*phi))
        #someone who is good at signs please help me simplify this. my family is dying
        res = (-1)**(n/2-KroneckerDelta(m,0)+1+n) * 2*jnp.pi*Jvs[n+1] * angular / rho
        return res
    return bessel

#############################################
#COMPLEMENTARY HEMISPHERIC HARMONICS SOLUTION
#############################################
def CHSH_FT(l,m):
    """Returns a callable function of rho, phi that returns the FT of the (l,m)th element of the CHSH basis"""
    def sph_bessel(rho,phi):
        angular = np.where(m>=0,np.cos(m*phi), np.sin(np.abs(m)*phi))
        #someone who is good at signs please help me simplify this. my family is dying
        res = A(l,np.abs(m))*2*np.pi*(1j)**(abs(m)) * (-1)**(2*l-1-KroneckerDelta(m,0)) * factorial2(l+np.abs(m), exact=True) / factorial2(l-np.abs(m)-1, exact=True) * spherical_jn(l,rho) * angular / rho
        return  res
    return sph_bessel

def CHSH_FT_jax(l,m, jvs):
    """Returns a callable function of rho, phi that returns the FT of the (l,m)th element of the CHSH basis"""
    def sph_bessel(rho,phi):
        angular = jnp.where(m>=0,jnp.cos(m*phi), jnp.sin(jnp.abs(m)*phi))
        #someone who is good at signs please help me simplify this. my family is dying
        res = A(l,np.abs(m))*2*jnp.pi*(1j)**(abs(m)) * (-1)**(2*l-1-KroneckerDelta(m,0)) * factorial2(l+np.abs(m), exact=True) / factorial2(l-np.abs(m)-1, exact=True) * jvs[l] * angular / rho
        return  res
    return sph_bessel

def solution_vector(l_max):
    """Returns a function that constructs the two part Fourier solution vector given rho, phi"""
    n_max = l_max**2 + 2 * l_max + 1
    j_max = jmax(l_max)
    @jax.jit
    @partial(jnp.vectorize, signature=f"(),()->({j_max}),({n_max-j_max})")
    def impl(rho, phi):
        ft_hsh = []
        ft_chsh = []
        bessels = jax.scipy.special.bessel_jn(rho,v=l_max+1,n_iter=100)
        spherical_bessels = csphjy(l_max,rho)
        for l in range(l_max+1):
            for m in range(-l,l+1):
                #HSH
                if (l+m)%2==0:
                    ft_hsh.append(zernike_FT_jax(l,m,bessels)(rho,phi))
                else:
                    ft_chsh.append(CHSH_FT_jax(l,m, spherical_bessels)(rho,phi))
        return jnp.stack(ft_hsh), jnp.stack(ft_chsh)
    return impl