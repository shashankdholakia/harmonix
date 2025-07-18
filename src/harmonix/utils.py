import numpy as np
import jax.numpy as jnp
import jax
from jax import jit
from functools import partial

from jax.scipy.special import gamma

from jax import config
config.update("jax_enable_x64", True)  

from jax.lax import scan

from jax._src.lax import lax
from jax._src.typing import Array, ArrayLike
from jax._src.numpy.util import (
   check_arraylike, promote_dtypes_inexact, _where)
from jax._src.custom_derivatives import custom_jvp

_lax_const = lax._const


RP1 = jnp.array([
-8.99971225705559398224E8, 4.52228297998194034323E11,
-7.27494245221818276015E13, 3.68295732863852883286E15,])
RQ1 = jnp.array([
 1.0, 6.20836478118054335476E2, 2.56987256757748830383E5, 8.35146791431949253037E7, 
 2.21511595479792499675E10, 4.74914122079991414898E12, 7.84369607876235854894E14, 
 8.95222336184627338078E16, 5.32278620332680085395E18,])

PP1 = jnp.array([
 7.62125616208173112003E-4, 7.31397056940917570436E-2, 1.12719608129684925192E0, 
 5.11207951146807644818E0, 8.42404590141772420927E0, 5.21451598682361504063E0, 1.00000000000000000254E0,])
PQ1 = jnp.array([
 5.71323128072548699714E-4, 6.88455908754495404082E-2, 1.10514232634061696926E0, 
 5.07386386128601488557E0, 8.39985554327604159757E0, 5.20982848682361821619E0, 9.99999999999999997461E-1,])

QP1 = jnp.array([
 5.10862594750176621635E-2, 4.98213872951233449420E0, 7.58238284132545283818E1, 
 3.66779609360150777800E2, 7.10856304998926107277E2, 5.97489612400613639965E2, 2.11688757100572135698E2, 2.52070205858023719784E1,])
QQ1  = jnp.array([
 1.0, 7.42373277035675149943E1, 1.05644886038262816351E3, 4.98641058337653607651E3, 
 9.56231892404756170795E3, 7.99704160447350683650E3, 2.82619278517639096600E3, 3.36093607810698293419E2,])

YP1 = jnp.array([
 1.26320474790178026440E9,-6.47355876379160291031E11, 1.14509511541823727583E14,
 -8.12770255501325109621E15, 2.02439475713594898196E17,-7.78877196265950026825E17,])
YQ1 = jnp.array([
 5.94301592346128195359E2, 2.35564092943068577943E5, 7.34811944459721705660E7, 
 1.87601316108706159478E10, 3.88231277496238566008E12, 6.20557727146953693363E14, 
 6.87141087355300489866E16, 3.97270608116560655612E18,])

Z1 = 1.46819706421238932572E1
Z2 = 4.92184563216946036703E1
PIO4 = .78539816339744830962 # pi/4
THPIO4 = 2.35619449019234492885 # 3*pi/4
SQ2OPI = .79788456080286535588 # sqrt(2/pi)

def j1_small(x):
    z = x * x
    w = jnp.polyval(RP1, z) / jnp.polyval(RQ1, z)
    w = w * x * (z - Z1) * (z - Z2)
    return w

def j1_large_c(x):    
    w = 5.0 / x
    z = w * w
    p = jnp.polyval(PP1, z) / jnp.polyval(PQ1, z)
    q = jnp.polyval(QP1, z) / jnp.polyval(QQ1, z)
    xn = x - THPIO4
    p = p * jnp.cos(xn) - w * q * jnp.sin(xn)
    return p * SQ2OPI / jnp.sqrt(x)

def j1(x):
    """
    Bessel function of order one - using the implementation from CEPHES, translated to Jax.
    """
    return jnp.sign(x)*jnp.where(jnp.abs(x) < 5.0, j1_small(jnp.abs(x)),j1_large_c(jnp.abs(x)))


PP0 = jnp.array([
  7.96936729297347051624E-4,
  8.28352392107440799803E-2,
  1.23953371646414299388E0,
  5.44725003058768775090E0,
  8.74716500199817011941E0,
  5.30324038235394892183E0,
  9.99999999999999997821E-1,
])
PQ0 = jnp.array([
  9.24408810558863637013E-4,
  8.56288474354474431428E-2,
  1.25352743901058953537E0,
  5.47097740330417105182E0,
  8.76190883237069594232E0,
  5.30605288235394617618E0,
  1.00000000000000000218E0,
])

QP0 = jnp.array([
-1.13663838898469149931E-2,
-1.28252718670509318512E0,
-1.95539544257735972385E1,
-9.32060152123768231369E1,
-1.77681167980488050595E2,
-1.47077505154951170175E2,
-5.14105326766599330220E1,
-6.05014350600728481186E0,
])
QQ0 = jnp.array([1.0,
  6.43178256118178023184E1,
  8.56430025976980587198E2,
  3.88240183605401609683E3,
  7.24046774195652478189E3,
  5.93072701187316984827E3,
  2.06209331660327847417E3,
  2.42005740240291393179E2,
])

YP0 = jnp.array([
 1.55924367855235737965E4,
-1.46639295903971606143E7,
 5.43526477051876500413E9,
-9.82136065717911466409E11,
 8.75906394395366999549E13,
-3.46628303384729719441E15,
 4.42733268572569800351E16,
-1.84950800436986690637E16,
])
YQ0 = jnp.array([
 1.04128353664259848412E3,
 6.26107330137134956842E5,
 2.68919633393814121987E8,
 8.64002487103935000337E10,
 2.02979612750105546709E13,
 3.17157752842975028269E15,
 2.50596256172653059228E17,
])

DR10 = 5.78318596294678452118E0
DR20 = 3.04712623436620863991E1

RP0 = jnp.array([
-4.79443220978201773821E9,
 1.95617491946556577543E12,
-2.49248344360967716204E14,
 9.70862251047306323952E15,
])
RQ0 = jnp.array([ 1.0,
 4.99563147152651017219E2,
 1.73785401676374683123E5,
 4.84409658339962045305E7,
 1.11855537045356834862E10,
 2.11277520115489217587E12,
 3.10518229857422583814E14,
 3.18121955943204943306E16,
 1.71086294081043136091E18,
])


def j0_small(x):
    '''
    Implementation of J0 for x < 5 
    '''
    z = x * x
    # if x < 1.0e-5:
    #     return 1.0 - z/4.0

    p = (z - DR10) * (z - DR20)
    p = p * jnp.polyval(RP0,z)/jnp.polyval(RQ0, z)
    return jnp.where(x<1e-5,1-z/4.0,p)
    

def j0_large(x):
    '''
    Implementation of J0 for x >= 5
    '''

    w = 5.0/x
    q = 25.0/(x*x)
    p = jnp.polyval(PP0, q)/jnp.polyval(PQ0, q)
    q = jnp.polyval(QP0, q)/jnp.polyval(QQ0, q)
    xn = x - PIO4
    p = p * jnp.cos(xn) - w * q * jnp.sin(xn)
    return p * SQ2OPI / jnp.sqrt(x)

def j0(x):
    '''
    Implementation of J0 for all x in Jax
    '''

    return jnp.where(jnp.abs(x) < 5.0, j0_small(jnp.abs(x)),j0_large(jnp.abs(x)))

@partial(jit,static_argnums=0)
def bessel_jn(n, x):
    """Compute the Bessel function J_n(x), for n >= 0"""
    # use recurrence relations
    def body(carry, i):
        jnm1, jn = carry
        jnplus = (2*i)/x * jn - jnm1
        return (jn, jnplus), jnplus
    j0_val, j1_val = j0(x), j1(x)
    _, jn = scan(body, (j0_val, j1_val), jnp.arange(1,n))

    return jnp.concatenate((jnp.array([j0_val]), jnp.array([j1_val]), jn))

def jint(n):
    return jnp.astype(jnp.trunc(n), 'int')

def spb1(x: ArrayLike, /) -> Array:
    '''
    Calculate the spherical Bessel functions j_1(z). 
    Follows existing implementation of jnp.sinc for safety around 0, 
    using a Maclaurin series to keep continuous derivatives.

    Arguments:
        x: The argument of the function.

    Returns:
        csj: The function j_1(z).
    '''

    check_arraylike("spb1", x)
    x, = promote_dtypes_inexact(x)

    # not defined at zero
    eq_zero = lax.eq(x, _lax_const(x, 0))
    
    safe_x = _where(eq_zero, _lax_const(x, 1), x)
    return _where(eq_zero, _spb1_maclaurin(0, x),
                    lax.div(lax.sin(safe_x), safe_x**2)-lax.div(lax.cos(safe_x), safe_x))

@partial(custom_jvp, nondiff_argnums=(0,))
def _spb1_maclaurin(k, x):
  # compute the kth derivative of x -> sin(x)/x evaluated at zero (since we
  # compute the monomial term in the jvp rule)
  # TODO(mattjj): see https://github.com/google/jax/issues/10750
  if k % 2:
    return x * 0
  else:
    top = 1.j* (1.j/2)**k * jnp.sqrt(jnp.pi)
    bottom = (k-1) * gamma(2+k/2) * gamma(0.5*(k-1))
    return x * 0 + jnp.real(top / bottom)

@_spb1_maclaurin.defjvp
def _spb1_maclaurin_jvp(k, primals, tangents):
  (x,), (t,) = primals, tangents
  return _spb1_maclaurin(k, x), _spb1_maclaurin(k + 1, x) * t


def envj(n, x):
    '''
    Helper function for msta1 and msta2.

    '''
    envj = 0.5 * jnp.log10(6.28 * n) - n * jnp.log10(1.36 * x / n) # always true 

    return envj

def msta1(x, mp):
    ''' 
    Calculate the number of terms required for the spherical Bessel function.
    '''
    a0 = jnp.abs(x)
    n0 = jint(1.1 * a0) + 1
    f0 = envj(n0, a0) - mp
    n1 = n0 + 5
    f1 = envj(n1, a0) - mp

    nn = jint(n1 - (n1 - n0) / (1.0 - f0 / f1))
    f = envj(nn, a0) - mp
    n0 = n1
    f0 = f1
    n1 = nn
    f1 = f
    diff = jnp.abs(nn - n1)

    def cond_fun(inputs):
        n0, f0, n1, f1, nn, counter, diff = inputs
        return jnp.logical_and(jnp.abs(diff) > 1, counter < 20)

    def body_fun(inputs):
        n0, f0, n1, f1, nn, diff, counter = inputs
        nn = jint(n1 - (n1 - n0) / (1.0 - f0 / f1))
        diff = nn - n1
        f = envj(nn, a0) - mp
        n0 = n1
        f0 = f1
        n1 = nn
        f1 = f
        counter += 1
        return n0, f0, n1, f1, nn, diff, counter

    n0, f0, n1, f1, nn, diff, _ = jax.lax.while_loop(cond_fun, body_fun, (n0, f0, n1, f1, nn, diff, 0))

    return nn

def msta2(x, n, mp):
    ''' 
    Calculate the number of terms required for the spherical Bessel function.
    '''
    a0 = jnp.abs(x)
    hmp = 0.5 * mp
    ejn = envj(n, a0)

    obj, n0 = jax.lax.cond(ejn <= hmp, 
                       lambda _: (mp*1.0, jint(1.1 * a0) + 1), 
                       lambda _: (hmp + ejn, jint(n)), 
                       operand=None)

    f0 = envj(n0, a0) - obj
    n1 = n0 + 5
    f1 = envj(n1, a0) - obj

    nn = jint(n1 - (n1 - n0) / (1.0 - f0 / f1))

    def cond_fun(inputs):
        n0, f0, n1, f1, nn, diff, counter = inputs
        return jnp.logical_and(jnp.abs(diff) >= 1, counter < 20)

    def body_fun(inputs):
        n0, f0, n1, f1, nn, diff, counter = inputs
        nn = jint(n1 - (n1 - n0) / (1.0 - f0 / f1))
        diff = nn - n1
        f = envj(nn, a0) - obj
        n0 = n1
        f0 = f1
        n1 = nn
        f1 = f
        counter += 1
        return n0, f0, n1, f1, nn, diff, counter
    
    n0, f0, n1, f1, nn, diff, _ = jax.lax.while_loop(cond_fun, body_fun, (n0, f0, n1, f1, nn, nn-n1, 0))

    return nn + 10

@partial(custom_jvp, nondiff_argnums=(0,))
@partial(jit,static_argnums=0)
def csphjy(n, z):
    ''' 
    Spherical Bessel functions of the first and second kind, and their derivatives.
    Follows the implementation of https://github.com/emsr/maths_burkhardt/blob/master/special_functions.f90, but with the derivatives.
    Arguments:
        n: The order of the spherical Bessel function.
        z: The argument of the function.    
    Returns:
        nm: The number of terms used in the calculation.
        csj: The function j_n(z).
        cdj: The derivative of the function j_n(z).
        csy: The function y_n(z).
        cdy: The derivative of the function y_n(z).
        
    '''
    a0 = jnp.abs(z)
    nm = n
    complex = jax.dtypes.canonicalize_dtype(jnp.complex128)
    csj = jnp.zeros(n+1, dtype=complex)
    csj = csj.at[0].set(jnp.sinc(z /jnp.pi))
    csj = csj.at[1].set(spb1(z))

    if n >= 2:
        csa = csj[0]
        csb = csj[1]
        m = msta1(a0, 200)

        m, nm = jax.lax.cond(m < n, 
                     lambda _: (m, m), 
                     lambda _: (msta2(a0, n, 15), n), 
                     operand=None)

        cf0 = 0.0
        cf1 = 1.0e-100
        cf = (2.0 * m + 3.0) * cf1 / z - cf0

        def body_fun(kk, inputs):
            k = m - kk
            cf, csj, cf0, cf1 = inputs
            cf = (2.0 * k + 3.0) * cf1 / z - cf0
            def true_fun(csj):
                return csj.at[k].set(cf)
            csj = jax.lax.cond(k <= nm, true_fun, lambda csj: csj, csj)
            cf0 = cf1
            cf1 = cf
            return cf, csj, cf0, cf1

        cf, csj, cf0, cf1 = jax.lax.fori_loop(0, m+1, body_fun, (cf, csj, cf0, cf1))

        cs = jax.lax.cond(jnp.abs(csa) <= jnp.abs(csb), 
                  lambda _: csb / cf0, 
                  lambda _: csa / cf, 
                  operand=None)
        

        csj = cs * csj

    return csj

@csphjy.defjvp
def csphjy_jvp(n, primals, tangents):
    z, = primals
    z_dot, = tangents
    csj = csphjy(n,z)
    
    cdj = jnp.zeros(n+1, dtype=complex)
    cdj = cdj.at[0].set((jnp.cos(z) - jnp.sin(z) / z) / z)
    cdj = cdj.at[1:].set(csj[:-1] - (jnp.arange(1, len(csj)) + 1.0) * csj[1:] / z)
    return csj, cdj*z_dot


def maketriples_all(mask,verbose=False):
    """ returns int array of triple hole indices (0-based), 
        and float array of two uv vectors in all triangles
    """
    nholes = mask.shape[0]
    tlist = []
    for i in range(nholes):
        for j in range(nholes):
            for k in range(nholes):
                if i < j and j < k:
                    tlist.append((i, j, k))
    tarray = np.array(tlist).astype(np.int32)
    if verbose:
        print("tarray", tarray.shape, "\n", tarray)

    tname = []
    uvlist = []
    # foreach row of 3 elts...
    for triple in tarray:
        tname.append("{0:d}_{1:d}_{2:d}".format(
            triple[0], triple[1], triple[2]))
        if verbose:
            print('triple:', triple, tname[-1])
        uvlist.append((mask[triple[0]] - mask[triple[1]],
                       mask[triple[1]] - mask[triple[2]]))
    # print(len(uvlist), "uvlist", uvlist)
    if verbose:
        print(tarray.shape, np.array(uvlist).shape)
    return tarray, np.array(uvlist)

def makebaselines(mask):
    """
    ctrs_eqt (nh,2) in m
    returns np arrays of eg 21 baselinenames ('0_1',...), eg (21,2) baselinevectors (2-floats)
    in the same numbering as implaneia
    """
    nholes = mask.shape[0]
    blist = []
    for i in range(nholes):
        for j in range(nholes):
            if i < j:
                blist.append((i, j))
    barray = np.array(blist).astype(np.int32)
    # blname = []
    bllist = []
    for basepair in blist:
        # blname.append("{0:d}_{1:d}".format(basepair[0],basepair[1]))
        baseline = mask[basepair[0]] - mask[basepair[1]]
        bllist.append(baseline)
    return barray, np.array(bllist)

@jit
def compute_DFTM1(x,y,uv,wavel):
    '''
    Compute a direct Fourier transform matrix, from coordinates x and y
    (milliarcsec) to uv (metres) at a given wavelength wavel.
    '''

    # Convert to radians
    x = x * jnp.pi / 180.0 / 3600.0/ 1000.0
    y = y * jnp.pi / 180.0 / 3600.0/ 1000.0

    # get uv in nondimensional units
    uv = uv / wavel

    # Compute the matrix
    dftm = jnp.exp(-2j* jnp.pi* (jnp.outer(uv[:,0],x)+jnp.outer(uv[:,1],y)))

    return dftm

@jit
def apply_DFTM1(image,dftm):
    '''Apply a direct Fourier transform matrix to an image.'''
    image /= image.sum()
    return jnp.dot(dftm,image.ravel())