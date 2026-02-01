#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "cuda_backend/gpu_arrays.hpp"

class GpuArraysTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaSetDevice(0);
    }

    void TearDown() override {
        cudaDeviceReset();
    }
};

TEST_F(GpuArraysTest, DeviceVectorBasicAllocation) {
    mpcd::cuda::DeviceVector<float> vec(1000);

    EXPECT_EQ(vec.size(), 1000);
    EXPECT_NE(vec.data(), nullptr);
    EXPECT_FALSE(vec.copy);
}

TEST_F(GpuArraysTest, DeviceVectorCopySemantics) {
    mpcd::cuda::DeviceVector<float> original(100);
    mpcd::cuda::DeviceVector<float> copy = original;

    EXPECT_EQ(original.data(), copy.data());  // Same pointer
    EXPECT_FALSE(original.copy);  // Original owns memory
    EXPECT_TRUE(copy.copy);       // Copy doesn't own memory
}

TEST_F(GpuArraysTest, UnifiedVectorToDeviceVectorCast) {
    mpcd::cuda::UnifiedVector<float> unified(50);

    // Test the problematic cast operator
    mpcd::cuda::DeviceVector<float> device = unified;

    EXPECT_EQ(device.data(), unified.device_store);
    EXPECT_TRUE(device.copy);
}
