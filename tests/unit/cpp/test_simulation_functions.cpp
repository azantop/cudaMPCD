#include <gtest/gtest.h>
#include "cpu_backend/simulation_functions.hpp"

TEST(TestSimulationFunctions, translate_particles) {
    using namespace mpcd;
    Vector test_volume(10, 10, 10);
    Vector grid_shift(0, 0, 0);
    IntVector periodicity(1, 1, 1);

    cpu::VolumeContainer<MPCCell> mpc_cells(test_volume);
    std::vector<Particle> particles(10);
    std::vector<uint32_t> uniform_counter(mpc_cells.size());
    std::vector<uint32_t> uniform_list(mpc_cells.size() * 4 * 10);
    Xoshiro128Plus generator;

    for (int i = 0; i < 10; ++i) {
        for (auto& particle : particles) {
            particle.position = Vector(0, 0, 0);
            particle.velocity = generator.genUnitVector();
        }

        uniform_counter.assign(mpc_cells.size(), 0);
        cpu::translate_particles(particles, mpc_cells, generator, grid_shift, test_volume, periodicity, 1,
                                            ExperimentType::standart, uniform_counter, uniform_list);

        for (auto& particle : particles) {
            EXPECT_NEAR(particle.position.x, particle.velocity.x, 1e-6);
            EXPECT_NEAR(particle.position.y, particle.velocity.y, 1e-6);
            EXPECT_NEAR(particle.position.z, particle.velocity.z, 1e-6);
        }
    }

    for (auto& particle : particles) {
        particle.position = Vector(-5, -5, -5);
        particle.velocity = Vector(0, 0, 0);
    }

    uniform_counter.assign(mpc_cells.size(), 0);
    cpu::translate_particles(particles, mpc_cells, generator, grid_shift, test_volume, periodicity, 1,
                                        ExperimentType::standart, uniform_counter, uniform_list);

    EXPECT_EQ(uniform_counter[0], 10);
}
