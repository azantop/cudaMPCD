#pragma once

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <fstream>

#if !defined(__CUDA_ARCH__) && !defined(__NVCC__)
    #define __host__
    #define __device__
    #define __forceinline__
#endif

// CUDA compatibility for older architectures (< 6.0)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double const val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                       __double_as_longlong(val + __longlong_as_double(assumed)));
    }
    while (assumed != old);

    return __longlong_as_double(old);
}
#endif

namespace mpcd
{
    /**
     * @brief 3D vector template class optimized for CUDA compatibility
     *
     * This struct provides efficient 3D vector operations with support for both
     * CPU and GPU execution. Direct member access ensures optimal performance
     * in CUDA kernels and tight computational loops.
     *
     * @tparam T Base type (float, double, int, etc.)
     */
    template<typename T>
    struct Vector3D{
        T x, y, z;

        // Constructors
        __forceinline__ Vector3D() = default;
        __forceinline__ Vector3D(const Vector3D&) = default;
        __forceinline__ Vector3D(Vector3D&&) = default;

        __host__ __device__ __forceinline__
        Vector3D(T _x, T _y, T _z) : x(_x), y(_y), z(_z) {}

        // Template constructor for type conversion
        template<typename U>
        __host__ __device__ __forceinline__
        Vector3D(const Vector3D<U>& v) {
            x = static_cast<T>(v.x);
            y = static_cast<T>(v.y);
            z = static_cast<T>(v.z);
        }

        // Assignment operators
        __forceinline__ Vector3D& operator=(const Vector3D&) = default;
        __forceinline__ Vector3D& operator=(Vector3D&&) = default;

        template<typename U>
        __host__ __device__ __forceinline__
        Vector3D& operator=(const Vector3D<U>& v) {
            x = static_cast<T>(v.x);
            y = static_cast<T>(v.y);
            z = static_cast<T>(v.z);
            return *this;
        }

        // Utility functions
        __host__ __device__ __forceinline__
        bool isFinite() const {
            return isfinite(x) && isfinite(y) && isfinite(z);
        }

        // Scalar assignment and comparison operators
        __host__ __device__ __forceinline__
        Vector3D& operator=(T const& value) {
            x = value; y = value; z = value; return *this;
        }

        __host__ __device__ __forceinline__
        bool operator==(const Vector3D& o) const { return (x == o.x) && (y == o.y) && (z == o.z); }

        __host__ __device__ __forceinline__
        bool operator!=(const Vector3D& o) const { return (x != o.x) || (y != o.y) || (z != o.z); }

        __host__ __device__ __forceinline__
        bool operator==(T const& value) const { return (x == value) && (y == value) && (z == value); }

        __host__ __device__ __forceinline__
        bool operator!=(T const& value) const { return (x != value) || (y != value) || (z != value); }

        // Vector arithmetic operators
        __host__ __device__ __forceinline__
        Vector3D& operator+=(const Vector3D& o) { x += o.x; y += o.y; z += o.z; return *this; }

        __host__ __device__ __forceinline__
        Vector3D& operator-=(const Vector3D& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }

        __host__ __device__ __forceinline__
        Vector3D& operator*=(const Vector3D& o) {  x *= o.x; y *= o.y; z *= o.z; return *this; }

        __host__ __device__ __forceinline__
        Vector3D& operator/=(const Vector3D& o) { x /= o.x; y /= o.y; z /= o.z; return *this; }

        // Scalar arithmetic operators
        __host__ __device__ __forceinline__
        Vector3D& operator+=(T const& value) { x += value; y += value; z += value; return *this; }

        __host__ __device__ __forceinline__
        Vector3D& operator-=(T const& value) { x -= value; y -= value; z -= value; return *this; }

        __host__ __device__ __forceinline__
        Vector3D& operator*=(T const& value) { x *= value; y *= value; z *= value; return *this; }

        __host__ __device__ __forceinline__
        Vector3D& operator/=(T const& value) {
            T inv = T(1) / value;
            x *= inv; y *= inv; z *= inv;
            return *this;
        }

        // Modulo operators (for integer types)
        __host__ __device__ __forceinline__
        Vector3D<int> operator%(const Vector3D<int>& o) const {
            return { static_cast<int>(x) % o.x, static_cast<int>(y) % o.y, static_cast<int>(z) % o.z };
        }

        __host__ __device__ __forceinline__
        Vector3D& operator%=(const Vector3D<int>& o) {
            x = x % o.x; y = y % o.y; z = z % o.z;
            return *this;
        }

        // Binary arithmetic operators
        __host__ __device__ __forceinline__
        Vector3D operator-(const Vector3D& o) const { return { x - o.x, y - o.y, z - o.z }; }

        __host__ __device__ __forceinline__
        Vector3D operator-(T const& value) const { return { x - value, y - value, z - value }; }

        __host__ __device__ __forceinline__
        Vector3D operator+(const Vector3D& o) const { return { x + o.x, y + o.y, z + o.z }; }

        __host__ __device__ __forceinline__
        Vector3D operator+(T const& value) const { return { x + value, y + value, z + value }; }

        __host__ __device__ __forceinline__
        Vector3D operator*(T const& value) const { return { x * value, y * value, z * value }; }

        __host__ __device__ __forceinline__
        Vector3D operator/(T const& value) const {
            T inv = T(1) / value;
            return { x * inv, y * inv, z * inv };
        }

        __host__ __device__ __forceinline__
        Vector3D operator-() const { return { -x, -y, -z }; }

        // Array-style access
        __host__ __device__ __forceinline__
        T& operator[](size_t const& i) { return (&x)[i]; }

        __host__ __device__ __forceinline__
        const T& operator[](size_t const& i) const { return (&x)[i]; }

        // Comparison operators returning element-wise results
        __host__ __device__ __forceinline__
        Vector3D<int> operator<(T const& value) const {
            return { x < value, y < value, z < value };
        }

        __host__ __device__ __forceinline__
        Vector3D<int> operator>(T const& value) const {
            return { x > value, y > value, z > value };
        }

        __host__ __device__ __forceinline__
        Vector3D<int> operator<(const Vector3D& p) const {
            return { x < p.x, y < p.y, z < p.z };
        }

        __host__ __device__ __forceinline__
        Vector3D<int> operator>(const Vector3D& p) const {
            return { x > p.x, y > p.y, z > p.z };
        }

        // Logical operators
        __host__ __device__ __forceinline__
        Vector3D<int> operator&&(const Vector3D& o) const {
            return { x && o.x, y && o.y, z && o.z };
        }

        __host__ __device__ __forceinline__ bool any() const { return x || y || z; }
        __host__ __device__ __forceinline__ bool all() const { return x && y && z; }

        __host__ __device__ __forceinline__
        Vector3D<int> elementWiseEqual(const Vector3D& o) const {
            return { x == o.x, y == o.y, z == o.z };
        }

        // Vector operations
        __host__ __device__ __forceinline__
        void set(T xx, T yy, T zz) {
            x = xx; y = yy; z = zz;
        }

        __host__ __device__ __forceinline__ T trace() const { return x + y + z; }
        __host__ __device__ __forceinline__ T diagonalProduct() const { return x * y * z; }
        __host__ __device__ __forceinline__ T getSquared() const { return x*x + y*y + z*z; }

        __host__ __device__ __forceinline__
        T dotProduct(const Vector3D& o) const {
            return x*o.x + y*o.y + z*o.z;
        }

        __host__ __device__ __forceinline__
        T dot(const Vector3D& o) const {
            return dotProduct(o);
        }

        __host__ __device__ __forceinline__
        Vector3D crossProduct(const Vector3D& o) const{
            return { y*o.z - z*o.y, z*o.x - x*o.z, x*o.y - y*o.x };
        }

        __host__ __device__ __forceinline__
        Vector3D cross(const Vector3D& o) const {
            return crossProduct(o);
        }

        /**
         * @brief Calculate the Euclidean length of the vector
         * @return Magnitude of the vector
         */
        __host__ __device__ __forceinline__ T length() const {
            return sqrt(getSquared());
        }

        /**
         * @brief Apply periodic boundary conditions (map to [-0.5, 0.5])
         * @return Vector with components wrapped to periodic bounds
         */
        __host__ __device__ __forceinline__
        Vector3D periodic() const{
            return { x - static_cast<T>(rintf(x)),
                     y - static_cast<T>(rintf(y)),
                     z - static_cast<T>(rintf(z)) };
        }

        __host__ __device__ __forceinline__
        Vector3D getInverse() const {
            return { T(1) / x, T(1) / y, T(1) / z };
        }

        __host__ __device__ __forceinline__
        Vector3D scaledWith(const Vector3D& o) const {
            return { x*o.x, y*o.y, z*o.z };
        }

        __host__ __device__ __forceinline__
        Vector3D xyScaledWith(const Vector3D& o) const {
            return { x*o.x, y*o.y, z };
        }

        __host__ __device__ __forceinline__
        T angleBetween(const Vector3D& o) const {
            return acos(dotProduct(o) / (o.length() * length()));
        }

        __host__ __device__ __forceinline__
        Vector3D unitVector() const {
            T invLength = T(1) / length();
            return { x * invLength, y * invLength, z * invLength };
        }

        // CUDA atomic operations
        #if defined(__CUDA_ARCH__) || defined(__NVCC__)
        __device__ __forceinline__
        void atomicAdd(const Vector3D& v) {
            ::atomicAdd(&x, v.x);
            ::atomicAdd(&y, v.y);
            ::atomicAdd(&z, v.z);
        }

        __device__ __forceinline__
        void atomicAdd(T const& a, T const& b, T const& c) {
            ::atomicAdd(&x, a);
            ::atomicAdd(&y, b);
            ::atomicAdd(&z, c);
        }
        #endif

        // Apply function to each element
        template<typename F>
        __host__ __device__ void elementWise(F&& f) {
            f(x); f(y); f(z);
        }

        // Binary I/O operations
        inline void writeBinary(std::ofstream& stream) const {
            stream.write((char*)&x, 3 * sizeof(T));
        }

        inline void readBinary(std::ifstream& stream) {
            stream.read((char*)&x, 3 * sizeof(T));
        }

        __host__ __device__ __forceinline__
        void print() const {
           printf("%f, %f, %f\n", static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
        }
    };

    // Global operators for scalar-vector operations
    template<typename T>
    __host__ __device__ __forceinline__
    Vector3D<T> operator-(T t, const Vector3D<T>& v) {
        return -(v - t);
    }

    template<typename T>
    __host__ __device__ __forceinline__
    Vector3D<T> operator+(T t, const Vector3D<T>& v) {
        return v + t;
    }

    template<typename T>
    __host__ __device__ __forceinline__
    Vector3D<T> operator*(T t, const Vector3D<T>& v){
        return v * t;
    }

    __host__ __device__ __forceinline__
    Vector3D<double> operator*(float t, const Vector3D<double>& v) {
        return v * static_cast<double>(t);
    }

    __host__ __device__ __forceinline__
    Vector3D<float> operator*(double t, const Vector3D<float>& v) {
        return v * static_cast<float>(t);
    }

    template<typename T>
    __host__ __device__ __forceinline__
    Vector3D<T> operator/(T d, const Vector3D<T>& v) {
        return v / d;
    }

    // Stream operators
    template<typename T>
    static std::ostream& operator<<(std::ostream& os, const Vector3D<T>& v) {
        os << v.x << " " << v.y << " " << v.z;
        return os;
    }

    template<typename T>
    static std::istream& operator>>(std::istream& is, Vector3D<T>& v) {
        is >> v.x >> v.y >> v.z;
        return is;
    }

    // Type aliases for common use cases
    using Float = float;
    using Vector = Vector3D<Float>;
    using FloatVector = Vector3D<float>;
    using DoubleVector = Vector3D<double>;
    using IntVector = Vector3D<int>;
    using UintVector = Vector3D<uint32_t>;

    // Component-wise mathematical functions
    __host__ __device__ __forceinline__
    Vector max(const Vector& v, const Vector& w) {
        return { fmaxf(v.x, w.x), fmaxf(v.y, w.y), fmaxf(v.z, w.z) };
    }

    __host__ __device__ __forceinline__
    Vector min(const Vector& v, const Vector& w) {
        return { fminf(v.x, w.x), fminf(v.y, w.y), fminf(v.z, w.z) };
    }

    __host__ __device__ __forceinline__
    Vector floor(const Vector& v) {
        return { floorf(v.x), floorf(v.y), floorf(v.z) };
    }

    __host__ __device__ __forceinline__
    Vector round(const Vector& v) {
        return { roundf(v.x), roundf(v.y), roundf(v.z) };
    }

    __host__ __device__ __forceinline__
    Vector abs(const Vector& v) {
        return { fabsf(v.x), fabsf(v.y), fabsf(v.z) };
    }

    // Mathematical constants
    namespace constants {
        const Vector xAxis = { 1, 0, 0 };
        const Vector yAxis = { 0, 1, 0 };
        const Vector zAxis = { 0, 0, 1 };
    }
} // namespace mpcd

#if !defined(__CUDA_ARCH__) && !defined(__NVCC__)
    #undef __host__
    #undef __device__
    #undef __forceinline__
#endif
