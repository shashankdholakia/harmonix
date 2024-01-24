import numpy as np
from scipy.special import gamma, factorial, comb

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

def z_to_p(j, jmax):
    """ returns the jth term of the zernike to polynomial basis"""
    n,m = j_to_nm(j)
    res = np.zeros(jmax+1)
    for s in range(0,int((n-np.abs(m))/2)+1):
        #this particular term of the polynomial has r^(n-2*s), that means we drop the r and instead add this to the array for n=n-2*s and m=m
        res[nm_to_j(int(n-2*s), m)] += (-1)**s * factorial(n-s) / (factorial(s) * factorial((n+np.abs(m))/2 -s) * factorial((n-np.abs(m))/2 - s))
    return res

def hsh_to_p(n, nmax):
    """ returns the jth term of the HSH to polynomial basis"""
    l,m = j_to_nm(n)
    res = np.zeros(nmax+1)
    for k in range(0,l-np.abs(m)+1):
        for j in range(0,int(k/2)+1):
            #this particular term of the polynomial has r^(2*j+np.abs(m)), that means we drop the r and instead add this to the array for n=n-2*s and m=m
            res[nm_to_j(int(2*j+np.abs(m)),m)] += (-1)**(l) * 2**l * (gamma((l+np.abs(m)+k-1)/2 +1)/(factorial(k) * factorial(l-np.abs(m)-k) * gamma((-l+np.abs(m)+k-1)/2+1))) * comb(k/2,j) * (-1)**j
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
    jmax = len(hsh_vector)
    A = np.linalg.inv(A_z_to_p(jmax)).T@A_hsh_to_p(jmax).T
    return A@hsh_vector


#############################################
#COMPLEMENTARY HEMISPHERIC HARMONICS SOLUTION
#############################################