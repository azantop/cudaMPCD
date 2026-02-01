#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "cuda_backend/simulation_kernels.hpp"

class SimulationKernelsTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaSetDevice(0);
    }

    void TearDown() override {
        cudaDeviceReset();
    }
};

TEST_F(SimulationKernelsTest, translateParticles) {
    using namespace mpcd;
    Vector test_volume(10, 10, 10);
    Vector grid_shift(0, 0, 0);
    IntVector periodicity(1, 1, 1);

    cuda::DeviceVolumeContainer<MPCCell> mpc_cells(test_volume);
    cuda::UnifiedVector<Particle> particles(1);
    cuda::DeviceVector<uint32_t> uniform_counter(mpc_cells.size());
    cuda::DeviceVector<uint32_t> uniform_list(particles.size());
    cuda::UnifiedVector<Xoshiro128Plus> generator(1);
    generator[0] = Xoshiro128Plus(); // init from urandom

    for (int i = 0; i < 10; ++i) {
        Vector test_velocity = generator[0].genUnitVector();
        particles[0].position = Vector(0, 0, 0);
        particles[0].velocity = test_velocity;

        particles.push();
        generator.push();

        translateParticles<<<1, 1>>>(particles, mpc_cells, generator, grid_shift, test_volume, periodicity, 1,
                                            ExperimentType::standart, uniform_counter, uniform_list);

        particles.pull();
        generator.push();

        EXPECT_EQ(particles[0].position.x, test_velocity.x);
        EXPECT_EQ(particles[0].position.y, test_velocity.y);
        EXPECT_EQ(particles[0].position.z, test_velocity.z);
    }
}
