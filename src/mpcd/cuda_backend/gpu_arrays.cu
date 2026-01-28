#include "common/vector_3d.hpp"
#include "common/particle.hpp"
#include "common/mpc_cell.hpp"
#include "common/random.hpp"

#include "gpu_arrays.hpp"
#include "gpu_error_check.hpp"

namespace mpcd::cuda {
    // AsyncVector method definitions
    template<class T> UnifiedVector<T>::UnifiedVector() = default;
    template<class T> UnifiedVector<T>::UnifiedVector(UnifiedVector &&) = default;

    template<class T>
    UnifiedVector<T>::UnifiedVector(const UnifiedVector &rhs) : host_store(rhs.host_store), device_store(rhs.device_store), count(rhs.count) {}

    template<class T>
    UnifiedVector<T>::UnifiedVector(UnifiedVector::size_type c) : count(c){
        cudaMallocHost((void**) &host_store, count * sizeof(T));
        error_check((std::string("Host alloc UnifiedVector<") + typeid(T).name() + ">: " + std::to_string(count) + " elements").c_str());

        cudaMalloc((void**) &device_store, count * sizeof(T));
        error_check((std::string("Device alloc UnifiedVector<") + typeid(T).name() + ">: " + std::to_string(count) + " elements").c_str());
    }

    template<class T>
    UnifiedVector<T>::~UnifiedVector() {
        if (count != 0) {
            cudaFree(device_store);
            error_check((std::string("Device free UnifiedVector<") + typeid(T).name() + ">").c_str());

            cudaFreeHost( host_store );
            error_check((std::string("Host free UnifiedVector<") + typeid(T).name() + ">").c_str());

        }
    }

    template<class T>
    void UnifiedVector<T>::push() {
        cudaMemcpy( device_store, host_store, count * sizeof( T ), cudaMemcpyHostToDevice );
        error_check((std::string("UnifiedVector<") + typeid(T).name() + ">::push").c_str());

    }

    template<class T>
    void UnifiedVector<T>::pull() {
        cudaMemcpy( host_store, device_store, count * sizeof( T ), cudaMemcpyDeviceToHost );
        error_check((std::string("UnifiedVector<") + typeid(T).name() + ">::pull").c_str());
    }

    template<class T>
    void UnifiedVector<T>::fill( T const value )
    {   for( size_t i = 0; i < count; ++i ) host_store[i] = value;   }

    template<class T>
    void UnifiedVector<T>::set( int i )
    {
        cudaMemset(device_store, i, sizeof(T) * count);
        error_check((std::string("UnifiedVector<") + typeid(T).name() + ">::set").c_str());
    }

    namespace hidden {
        template<typename T>
        struct isVector3D_ : std::false_type {};

        template<typename T>
        struct isVector3D_<math::Vector3D<T>> : std::true_type {};

        template<typename T>
        constexpr bool isVector3D = isVector3D_<T>::value;
    }

    template<class T>
    void UnifiedVector<T>::writeBinary(std::ofstream &stream) {
        if constexpr (std::is_same_v<T, Particle> || hidden::isVector3D<T>) {
            // Complex types with their own readBinary
            for (size_t i = 0; i < count; ++i) {
                host_store[i].writeBinary( stream );
            }
        } else {
            // POD types - direct read
            stream.write(reinterpret_cast<char*>(host_store), count * sizeof(T));
        }
    }

    template<class T>
    void UnifiedVector<T>::readBinary(std::ifstream &stream) {
        if constexpr (std::is_same_v<T, Particle> || hidden::isVector3D<T>) {
            // Complex types with their own readBinary
            for (size_t i = 0; i < count; ++i) {
                host_store[i].readBinary(stream);
            }
        } else {
            // POD types - direct read
            stream.read(reinterpret_cast<char*>(host_store), count * sizeof(T));
        }
    }

    template<typename T>
    UnifiedVector<T>::operator DeviceVector<T>() {
        return DeviceVector<T>(device_store, count); // sets copy = true
    }

    // Force instantiation
    template class UnifiedVector<unsigned>;
    template class UnifiedVector<unsigned long>;
    template class UnifiedVector<float>;
    template class UnifiedVector<double>;
    template class UnifiedVector<Particle>;
    template class UnifiedVector<math::Vector3D<float>>;
    template class UnifiedVector<MPCCell>;
    template class UnifiedVector<FluidState>;

    template<typename T>
    DeviceVector<T>::DeviceVector() : count(0), copy(0) {}

    template<typename T>
    DeviceVector<T>::DeviceVector(DeviceVector::size_type c) : count(c) {
        alloc(c);
    }

    template<typename T>
    DeviceVector<T>::DeviceVector(DeviceVector::size_type c, int init) : count(c) {
        alloc(c);
        set(init);
    }

    template<typename T>
    DeviceVector<T>::DeviceVector(T* data, DeviceVector::size_type c) : store(data), count(c), copy(1) {}

    template<typename T>
    __host__ __device__ DeviceVector<T>::~DeviceVector() {
        #ifndef __CUDA_ARCH__
            if ( !copy and count != 0 ) {
                cudaFree(store);
                error_check((std::string("delete/free GpuVector type: ") + typeid(T).name()).c_str());
            }
        #endif
    }

    template<typename T>
    DeviceVector<T> push(std::vector<T> const& source) {
        DeviceVector<T> destination(source.size());
        cudaMemcpy(destination.data(), source.data(), sizeof(T) * source.size(), cudaMemcpyHostToDevice);
        return destination;
    }

    template<typename T>
    void push(DeviceVector<T>& destination, std::vector< T > const& source) {
        if(destination.size() != source.size())
            throw std::runtime_error("Destination vector size mismatch");

        cudaMemcpy(destination.data(), source.data(), sizeof(T) * source.size(), cudaMemcpyHostToDevice);
    }

    template<typename T>
    void push(DeviceVector<T>& destination, T const* source) {
        cudaMemcpy(destination.data(), source, sizeof(T) * destination.size(), cudaMemcpyHostToDevice);
    }

    template<typename T>
    std::vector<T> pull(DeviceVector<T> const& source){
        std::vector<T> destination( source.size() );
        cudaMemcpy(destination.data(), source.data(), sizeof(T) * source.size(), cudaMemcpyDeviceToHost);
        return destination;
    }

    template<typename T>
    void pull(std::vector<T>& destination, DeviceVector<T> const& source){
        if(destination.size() != source.size())
            throw std::runtime_error("Destination vector size mismatch");

        cudaMemcpy(destination.data(), source.data(), sizeof(T) * source.size(), cudaMemcpyDeviceToHost);
    }

    template<typename T>
    void pull(T* destination, DeviceVector<T> const& source){
        cudaMemcpy(destination, source.data(), sizeof(T) * source.size(), cudaMemcpyDeviceToHost);
    }

    template class DeviceVector<unsigned int>;
    template class DeviceVector<unsigned long>;
    template class DeviceVector<float>;
    template class DeviceVector<double>;
    template class DeviceVector<Particle>;
    template class DeviceVector<math::Vector3D<float>>;
    template class DeviceVector<MPCCell>;
    template class DeviceVector<FluidState>;
    template class DeviceVector<Xoshiro128Plus>;

    template DeviceVector<float> push(std::vector<float> const&);
    template DeviceVector<double> push(std::vector<double> const&);
    template DeviceVector<Particle> push(std::vector<Particle> const&);
    template DeviceVector<math::Vector3D<float>> push(std::vector<math::Vector3D<float>> const&);
    template void push(DeviceVector<float>&, std::vector<float> const&);
    template void push(DeviceVector<double>&, std::vector<double> const&);
    template void push(DeviceVector<Particle>&, std::vector<Particle> const&);
    template void push(DeviceVector<math::Vector3D<float>>& , std::vector<math::Vector3D<float>> const&);
    template void push(DeviceVector<float>&, float const*);
    template void push(DeviceVector<double>&, double const*);
    template void push(DeviceVector<Particle>&, Particle const*);
    template void push(DeviceVector<math::Vector3D<float>>& , math::Vector3D<float> const*);

    template std::vector<float> pull(DeviceVector<float> const&);
    template std::vector<double> pull(DeviceVector<double> const&);
    template std::vector<Particle> pull(DeviceVector<Particle> const&);
    template std::vector<math::Vector3D<float>> pull(DeviceVector<math::Vector3D<float>> const&);
    template void pull(std::vector<float>&, DeviceVector<float> const&);
    template void pull(std::vector<double>&, DeviceVector<double> const&);
    template void pull(std::vector<Particle>&, DeviceVector<Particle> const&);
    template void pull(std::vector<math::Vector3D<float>>&, DeviceVector<math::Vector3D<float>> const&);
    template void pull(float*, DeviceVector<float> const&);
    template void pull(double*, DeviceVector<double> const&);
    template void pull(Particle*, DeviceVector<Particle> const&);
    template void pull(math::Vector3D<float>*, DeviceVector<math::Vector3D<float>> const&);
} // namespace mpcd::cuda
