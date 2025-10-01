# harmonix

Analytic interferometry of rotating stellar surfaces with JAX

## Overview

*harmonix* is a code to map rotating, spotted stars with observations from optical stellar interferometry. It contains new analytic solutions for the [visibility function](https://en.wikipedia.org/wiki/Interferometric_visibility) for a star with its surface map represented in the spherical harmonic basis. The code is based on the excellent [jaxoplanet](https://jax.exoplanet.codes/en/latest/) repository, which also uses the spherical harmonic map of the star to model photometric data, such as rotational light curves and transits and phase curves of exoplanets. 

## Installation

You will first need to install JAX, which is a python package similar to NumPy that comes with several other features that are useful for machine learning applications, including automatic differentiation and just-in-time compilation. 

```python
pip install "jax[cpu]"
```

then, you should be able to install harmonix with

```python
pip install harmonix
```

If you want to install the latest version of harmonix from source, which may come with certain bugfixes and features that aren't in the latest stable version on PyPi, you can install it from GitHub, for instance by doing:

```python
git clone https://github.com/shashankdholakia/harmonix.git
cd harmonix
pip install .
```

## Attribution

To cite this work, please cite the ArXiv version (until the paper on this work is accepted):

```latex
@misc{dholakia2025analyticinterferometryrotatingstellar,
      title={Analytic Interferometry of Rotating Stellar Surfaces}, 
      author={Shashank Dholakia and Benjamin J. S. Pope},
      year={2025},
      eprint={2509.25433},
      archivePrefix={arXiv},
      primaryClass={astro-ph.IM},
      url={https://arxiv.org/abs/2509.25433}, 
}
```

