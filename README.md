# cudaMPCD
Fast hydrodynamic solver for GPUS using the method of multi-particle collision dynamics in C++/CUDA.
For configuring the simulation after compilation using the make command, modify the input_file.     
In the current state, this code can be used to simulate a Poiseuille flow, i.e. a flow between to parallel plates.

## Installation

### Prerequisites

- CMake (version 3.12 or higher)
- Ninja
- CUDA-capable GPU and NVIDIA CUDA Toolkit
- C++ compiler with C++14 support or higher
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
params.experiment = "standart"
params.algorithm = "extended"

# Create and run simulation
sim = pympcd.Simulation(params, "cuda")
sim.step(1000)
sim.step_and_sample(100000)

# Get results
density, velocity = sim.get_mean_fields()
```

## Testing
The Poiseuille flow is a geometry for which the Navier-Stokes equations can be solved analytically. 
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
L = params.volume_size[2]
g = params.drag
n = params.n
dt = params.delta_t

# Viscosity measurement:
eta = L * L * n * g / (8 * vel_profile.max())

# Viscosity theoretical:
eta_theo = (1 - math.cos(120/180*math.pi)) / (6*3*dt) * (n - 1 + math.exp(-n))

print( "theoretical: ", eta_theo, ", measured:", eta ) 
```
The last line prints the two values of the fluid viscosity "theoretical: 37.50, measured: 37.27"

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

Dependencies: CUDA, HDF5

TODO: add references (papers)
