from sympy.parsing.mathematica import mathematica

import sympy as sp
from sympy.functions.special.bessel import besselj
from sympy.parsing.sym_expr import SymPyExpression
from sympy import latex

def get_ylm_FTs():
    results = []
    with open("SphericalHarmonicsResults_10.txt", 'r') as file:
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