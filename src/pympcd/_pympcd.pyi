from collections.abc import Sequence

import numpy as np

class Params:
    device_id: int
    delta_t: float
    drag: float
    equilibration_steps: int
    steps: int
    sample_rate: int
    average_samples: int
    n: int
    N: int
    temperature: float
    volume_size: tuple[float, float, float] | Sequence[float]
    periodicity: tuple[int, int, int] | Sequence[int]
    experiment: str  # "standard" | "channel"
    algorithm: str   # "srd" | "extended"
    def __init__(self) -> None: ...

class Simulation:
    def __init__(self, params: Params, backend: str = "cuda") -> None: ...
    def step(self, n_steps: int) -> None: ...
    def step_and_sample(self, n_steps: int) -> None: ...
    def get_mean_fields(self) -> tuple[np.ndarray, np.ndarray]: ...
    def get_particle_positions(self) -> np.ndarray: ...
    def get_particle_velocities(self) -> np.ndarray: ...
