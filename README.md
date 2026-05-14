# cudaMPCD

[![CI](https://github.com/azantop/cudaMPCD/actions/workflows/ci.yml/badge.svg)](https://github.com/azantop/cudaMPCD/actions/workflows/ci.yml)

Fast hydrodynamic solver using the method of multi-particle collision dynamics MPCD. Python frontend using pybind11 with backends implemented in C++ and CUDA for usage on CPU and GPUs. In the current state, this code can be used to simulate a Poiseuille flow, i.e. a flow between to parallel plates. This can be used for viscosity measurements. The code has implementations of 2 collision operators: standard stochastic rotation dynamics (SRD) and extended MPC with non-ideal equation-of-state.

## Architecture

```
pympcd (Python)              src/pympcd/__init__.py
    ↓
_pympcd (pybind11 module)    src/bindings/bindings.cpp
    ↓
SimulationHandle (API)       include/mpcd/api/simulation_handle.hpp
    ↓  (bridge pattern)
Backend (abstract)           src/libmpcd/common/backend.hpp
    ├── CPUBackend            src/libmpcd/cpu_backend/
    └── CUDABackend           src/libmpcd/cuda_backend/
```

Each MPCD time step consists of a translation step and a collision step. The collision step dispatches on both the algorithm and the kernel variant:

| `collision_kernel` | SRD | Extended |
|---|---|---|
| `"trivial"` | scatter/reduce, no shared memory | 6-pass multi-kernel baseline |
| `"sorting"` | trivial + counting sort prepended | trivial + counting sort prepended |
| `"optimized"` | warp-cooperative fused kernel | warp-cooperative fused kernel |

The kernel fusion keeps particle data in shared memory across the full collision step, eliminating repeated global-memory loads/stores between steps of the algorithm. Warp-level communications and reductions are hand-written using `__shfl_sync` / `__ballot_sync` rather than CUB, since the cooperative grouping logic (dynamic sub-warp partitioning based on per-cell particle counts) does not map cleanly onto CUB's fixed collective abstractions.

The extended algorithm in particular benefits strongly from kernel fusion: a fully decomposed scatter/reduce is structurally impossible here due to the stochastic collision gate and the circular dependency in momentum-conserving thermalisation in cell subgroups. The fused kernel resolves both by keeping all per-cell and particle states in shared memory within a single launch.

## Performance

Benchmarked on Ampere (1000×1000×20 cells, n=20 particles/cell):

| kernel | ms / step | speedup |
|---|---|---|
| trivial | 378 | 1.00× |
| sorting | 189 | 2.00× |
| optimized | 94 | 4.04× |

Sorting alone gives 2× from coalesced memory access. Kernel fusion adds another 2× on top by eliminating repeated global-memory passes. Both contributions are roughly equal on Ampere.

## Usage

After installation, you can import and use the module in Python:

```python
import pympcd

# Create simulation parameters
params = pympcd.Params()
params.n = 10
params.temperature = 1.0
params.volume_size = (10, 10, 100)
params.periodicity = (1, 1, 0)
params.drag = 0.001
params.delta_t = 0.02
params.experiment = "standard"
params.algorithm = "srd"
params.collision_kernel = "optimized"  # "trivial" | "sorting" | "optimized"

# Create and run simulation
sim = pympcd.Simulation(params, "cuda")
sim.step(1000)
sim.step_and_sample(100000)

# Get results
density, velocity = sim.get_mean_fields()
```
Use algorithm = "srd" for standard stochastic rotation dynamics, use algorithm = "extended" for extended MPC with non-ideal equation-of-state.
The Poiseuille flow used here, is a geometry for which the Navier-Stokes equations can be solved analytically. 
Thus, we can use this geometry for testing the code.
After the simulation, we may use the flow field data as follows:
```python
import matplotlib.pyplot as plt
import math

# Calculate velocity profile
x_velocity = velocity[:,:,:,0]; 
vel_profile = x_velocity.mean(axis=(0,1))

plt.plot(vel_profile) # will show a parabola
plt.xlabel('channel cross section')
plt.ylabel('flow speed')

# Simulation parameters:
L = params.volume_size[2] - 2 #subtract walls
g = params.drag
n = params.n
dt = params.delta_t

# Viscosity measurement:
eta = L * L * n * g / (8 * vel_profile.max())

# Viscosity theoretical for SRD algorithm:
eta_theo = (1 - math.cos(120/180*math.pi)) / (6*3*dt) * (n - 1 + math.exp(-n))

print( "SRD Viscosity; theoretical: ", eta_theo, ", measured:", eta ) 
```
The last line prints the two values of the fluid viscosity "SRD Viscosity; theoretical: 37.50, measured: 37.27"

## Installation

### Prerequisites

#### Required:
- CMake (version 3.12 or higher)
- C++ compiler with C++14 support or higher

#### Optional:
- CUDA-capable GPU and NVIDIA CUDA Toolkit
- Python 3.6+ with development headers
- pybind11
- NumPy

### Building and Installing

1. Clone the repository:
```bash
git clone https://github.com/azantop/cudaMPCD.git
cd cudaMPCD
```

2. use pip to install the package:
```bash
pip install .
```

3. For running simulations with standalone binary use Cmake:
```bash
mkdir build
cd build
cmake ..
cmake --build .
cmake --install .
```

### Docker

A Dockerfile is provided for reproducible GPU environments:
```bash
docker build -t cudampcd .
docker run --gpus all -v $(pwd):/workspace -p 8888:8888 cudampcd
# inside container:
pip install .
```

If you find this repository helpful, please consider citing our article (doi.org/10.1063/5.0037934)
```
@article{zantop2021multi,
  title={Multi-particle collision dynamics with a non-ideal equation of state. I},
  author={Zantop, Arne W and Stark, Holger},
  journal=JCP,
  volume={154},
  number={2},
  pages={024105},
  year={2021},
  publisher={AIP Publishing LLC}
}
```
