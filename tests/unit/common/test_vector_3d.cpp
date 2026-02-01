#include <gtest/gtest.h>
#include "common/vector_3d.hpp"

class Vector3DTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(Vector3DTest, Vector3DTestInit) {
    mpcd::Vector3D<float> x(1, 2, 3);
    EXPECT_EQ(x.x, 1);
    EXPECT_EQ(x.y, 2);
    EXPECT_EQ(x.z, 3);
}

TEST_F(Vector3DTest, Vector3DTestAdd) {
    mpcd::Vector3D<float> x(1, 2, 3);
    mpcd::Vector3D<float> y(4, 5, 6);
    mpcd::Vector3D<float> z = x + y;
    EXPECT_EQ(z.x, 5);
    EXPECT_EQ(z.y, 7);
    EXPECT_EQ(x.z, 3);
}

TEST_F(Vector3DTest, Vector3DTestSub) {
    mpcd::Vector3D<float> x(1, 2, 3);
    mpcd::Vector3D<float> y(4, 5, 6);
    mpcd::Vector3D<float> z = x - y;
    EXPECT_EQ(z.x, -3);
    EXPECT_EQ(z.y, -3);
    EXPECT_EQ(x.z, 3);
}

TEST_F(Vector3DTest, Vector3DTestCross) {
    mpcd::Vector3D<float> x(1, 2, 3);
    mpcd::Vector3D<float> y(4, 5, 6);
    mpcd::Vector3D<float> z = x.crossProduct(y);
    EXPECT_EQ(z.x, -3);
    EXPECT_EQ(z.y, 6);
    EXPECT_EQ(z.z, -3);
}

TEST_F(Vector3DTest, Vector3DTestNorm) {
    mpcd::Vector3D<float> x(1, 2, 3);
    float norm = x.length();
    EXPECT_EQ(norm, std::sqrt(14.0f));
}

TEST_F(Vector3DTest, Vector3DTestDot) {
    mpcd::Vector3D<float> x(1, 2, 3);
    mpcd::Vector3D<float> y(4, 5, 6);
    float dot = x.dot(y);
    EXPECT_EQ(dot, 32);
}
