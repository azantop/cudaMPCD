#pragma once

#include <cstddef>
#include <fstream>
#include <vector>

#ifdef __NVCC__
    #include <cuda.h>
    #include <cuda_runtime_api.h>
    #include <cuda_profiler_api.h>

    #include "gpu_error_check.hpp"
#endif

namespace mpcd::cuda {
    template<typename T>
    struct DeviceVector;

    template<class T>
    struct UnifiedVector {
        using value_type             =       T;
        using reference              =       T&;
        using const_reference        = const T&;
        using iterator               =       T*;
        using const_iterator         = const T*;
        using reverse_iterator       =       std::reverse_iterator<iterator>;
        using const_reverse_iterator =       std::reverse_iterator<const_iterator>;
        using size_type              =       size_t;
        using difference_type        =       ptrdiff_t;

        T*         host_store;
        T*         device_store;
        size_type  count;
        bool       copy;

        // Constructors:
        //UnifiedVector();
        UnifiedVector(UnifiedVector &&) noexcept;
        UnifiedVector(const UnifiedVector &rhs);

        UnifiedVector(size_type c);

        ~UnifiedVector();

        void push();
        void pull();

        iterator begin() { return host_store; }

        const_iterator begin()  const { return begin(); }
        const_iterator cbegin() const { return begin(); }
        iterator       end()          { return host_store + count; }
        const_iterator end()    const { return end(); }
        const_iterator cend()   const { return end(); }

        reverse_iterator       rbegin()        { return reverse_iterator( end() ); }
        const_reverse_iterator rbegin()  const { return const_reverse_iterator( end() ); }
        reverse_iterator       rend()          { return reverse_iterator( begin() ); }
        const_reverse_iterator rend()    const { return const_reverse_iterator( begin() ); }

        size_type size()               { return count; }
        size_type size()         const { return count; }

        size_type max_size()     const { return count; }
        bool      empty()        const { return count == 0; }

        reference       operator[](size_type n) { return host_store[ n ]; }
        const_reference operator[](size_type n) const { return host_store[ n ]; }

        reference       front() { return host_store[ 0 ]; }
        const_reference front() const { return host_store[ 0 ]; }

        reference       back() { return host_store[ count-1 ]; }
        const_reference back()  const { return host_store[ count-1 ]; }

        T*       data() { return host_store; }
        const T* data()  const { return host_store; }

        void fill( T const value );
        void set( int i );

        operator DeviceVector<T>();

        // File io:
        void writeBinary(std::ofstream &stream);
        void readBinary(std::ifstream &stream);
    };

    template<typename T>
    struct DeviceVector
    {
        using size_type = size_t;

        #ifndef __NVCC__
            DeviceVector(size_type);
            DeviceVector(size_type, int);
            DeviceVector(T*, size_type);
            ~DeviceVector();

            size_type size()     const { return count; }

            private:
        #else
            using value_type             =       T;
            using reference              =       T&;
            using const_reference        = const T&;
            using iterator               =       T*;
            using const_iterator         = const T*;
            using reverse_iterator       =       std::reverse_iterator< iterator >;
            using const_reverse_iterator =       std::reverse_iterator< const_iterator >;
            using difference_type        =       ptrdiff_t;

            __host__ DeviceVector();
            __host__ DeviceVector(size_type);
            __host__ DeviceVector(size_type, int);
            __host__ DeviceVector(T*, size_type);
            DeviceVector( DeviceVector && ) = default;
            __host__ __device__ DeviceVector( DeviceVector const& rhs ) : store( rhs.store ), count( rhs.count ), copy( 1 ) { }
            __host__ __device__ ~DeviceVector();

            __host__ void alloc(size_type c)
            {
                if (!hasCudaDevice())
                    throw std::runtime_error("No CUDA device found");

                if (!copy && count != 0 && store)
                    cudaFree(store);

                count = c;
                cudaMalloc((void**) &store, count * sizeof(T));
                error_check((std::string("Alloc DeviceVector<") + typeid(T).name() + ">: " + std::to_string(count) + " elements").c_str());

            }

            // capacity:
            __device__ __host__ size_type size()     const { return count; }

            // element access:
            __device__ reference  operator[]( size_type n ) const { return store[n]; }

            __device__ iterator       begin()        { return store; }
            __device__ const_iterator begin()  const { return store; }
            __device__ const_iterator cbegin() const { return store; }
            __device__ iterator       end()          { return store + count; }
            __device__ const_iterator end()    const { return store + count; }
            __device__ const_iterator cend()   const { return store + count; }

            __device__ reference       front()
            {
                return store[ 0 ];
            }
            __device__ const_reference front() const { return front(); }

            __device__ reference       back()
            {
                return store[ count-1 ];
            }
            __device__ const_reference back()  const { return back(); }

            // data access:
            __device__ __host__ value_type*  data() const { return store; }

            // modify:
            __host__ void set( int i ) { error_check( cudaMemset( store, i, sizeof( T ) * count ), "gpu_vector set" ); }

            T get()
            {
                T retval;
                cudaMemcpy( &retval, store, sizeof( T ), cudaMemcpyDeviceToHost );
                cudaError_t err = cudaGetLastError();
                if ( err != 0 ) printf( "%s\n", cudaGetErrorString( err ) );
                return retval;
            }
        #endif

        // data storages:
        T*        store;
        size_type count;
        bool      copy;
    };

    template<typename T>
    DeviceVector<T> push(std::vector<T> const&);
    template<typename T>
    void push(DeviceVector<T>&, T const*);
    template<typename T>
    void push(DeviceVector<T>&, std::vector< T > const&);

    template<typename T>
    std::vector<T> pull(DeviceVector<T> const&);
    template<typename T>
    void pull(T*, DeviceVector<T> const&);
    template<typename T>
    void pull(std::vector<T>&, DeviceVector<T> const&);
} // namespace mpcd::cuda
