import numpy as np
import pytest

import pympcd  # Your Python module


class TestCudaMPCD:
    def test_basic_simulation(self):
        """Test that basic simulation runs without crashes."""
        params = pympcd.Params()
        params.volume_size = [10, 10, 10]
        params.n = 5

        sim = pympcd.Simulation(params)

        # Should not crash
        sim.step(10)

        # Should be able to get particle data
        positions = sim.getParticlePositions()
        assert len(positions) > 0

    @pytest.mark.cuda
    def test_memory_consistency(self):
        """Test that GPU memory operations are consistent."""
        params = pympcd.Params()
        params.volume_size = [5, 5, 5]
        params.n = 2

        sim = pympcd.Simulation(params)

        # Get initial state
        pos1 = sim.getParticlePositions()

        # Run simulation
        sim.step(1)

        # Get final state
        pos2 = sim.getParticlePositions()

        # Positions should have changed
        assert not np.allclose(pos1, pos2)
