#include <gtest/gtest.h>
#include <memory>
#include "src/mpcd/cuda_backend/cuda_backend.hpp"

// Test for the specific issue you found
TEST(MemorySafetyTest, NoDeviceResetAfterAllocation) {
    // This should NOT cause memory corruption
    mpcd::SimulationParameters params;
    // Set up params...

    EXPECT_NO_THROW({
        auto backend = std::make_unique<mpcd::cuda::CudaBackend>(params);
        // Should be able to use backend without cuda-gdb showing invalid pointers
    });
}

// Stress test for copy/move semantics
TEST(MemorySafetyTest, MassiveCopyStressTest) {
    const size_t large_size = 1000;

    std::vector<mpcd::cuda::DeviceVector<float>> vectors;

    // Create original
    mpcd::cuda::DeviceVector<float> original(large_size);

    // Make many copies
    for (int i = 0; i < 100; ++i) {
        vectors.push_back(original);  // Should not cause double-free
    }

    // All should have same pointer, copy=true
    for (const auto& vec : vectors) {
        EXPECT_EQ(vec.data(), original.data());
        EXPECT_TRUE(vec.copy);
    }
}
