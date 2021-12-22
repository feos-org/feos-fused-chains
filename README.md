# FeOs - fused-chain functional

Fused-chain Helmholtz energy functional implemented in the `feos` framework.

The development of the model is presented in a manuscript submitted to Physical Review E.

## Installation

Add this to your `Cargo.toml`

```toml
[dependencies]
feos-fused-chains = "0.1"
```

## Building python wheel

From within a Python virtual environment with `maturin` installed, type
```
maturin build --release --out dist --no-sdist
```
to build the wheel. To directly install the package into the virtual environment, use
```
maturin develop --release
```

## Example

For simple systems, an instance of the functional can be created using either of the following constructors:

```python
from feos_fused_chains import *

func = FusedChainFunctional.new_monomer(1.0)
func = FusedChainFunctional.new_dimer(1.0, 0.8, 0.5)
func = FusedChainFunctional.new_trimer(0.8, 1.0, 0.8, 0.5, 0.7)
func = FusedChainFunctional.new_homosegmented(5, 1.0, 0.6)
```
More general fluids can be specified by creating `FusedChainRecord`s first
```python
from feos_fused_chains import *
import numpy as np

dimer = FusedChainRecord(np.array([1.0, 1.0]), [(0, 1, 0.5)])
trimer = FusedChainRecord(np.array([0.6, 1.0, 0.8]), [(0, 1, 0.8), (1, 2, 0.9)])
chain = FusedChainRecord(np.array([1.0] * 5), [(i, i+1, 0.8) for i in range(4)])
func = FusedChainFunctional.from_records([dimer, trimer, chain])
```
The functional can then be used with all applicable features of the [FeOs](https://feos-org.github.io/feos/) framework.

Example: calculate the density profile of a dimer in a slit pore
```python
from feos_fused_chains import *
from feos_fused_chains.si import *

import matplotlib.pyplot as plt

dimer = FusedChainFunctional.new_dimer(1.0, 1.0, 0.6)
bulk = State(dimer, KELVIN, density=0.4/NAV/ANGSTROM**3)
profile = Pore1D(dimer, Geometry.Cartesian, 11*ANGSTROM, ExternalPotential.HardWall(1.0)).initialize(bulk).solve()

plt.plot(profile.z/ANGSTROM, (profile.density*NAV*ANGSTROM**3).T)
plt.axis([0,5,0,1.5])
```