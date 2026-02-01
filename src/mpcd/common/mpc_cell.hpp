#pragma once

#include "common/particle.hpp"
#include "common/vector_3d.hpp"

#if defined(__CUDACC__) || defined(__NVCC__) // hide device functions from gcc
    #include "cuda_backend/gpu_utilities.hpp"
#endif

namespace mpcd {
    /**
    *  @brief Object for data and functions related to the MPC algorithm. Additionally, the
    *         class holds pointers to obstacles.
    */
    struct MPCCell
    {
        using Vector = mpcd::Vector;
        using Float  = mpcd::Float;

        unsigned density;
        Vector   mean_velocity,
                 centre_of_mass;

    #if defined(__CUDACC__) || defined(__NVCC__) // hide device functions from gcc

        __device__ void atomic_add(MPCCell const& rhs) {
            atomicAdd(&density, rhs.density);

            mean_velocity .atomicAdd(rhs.mean_velocity);
            centre_of_mass.atomicAdd(rhs.centre_of_mass);
        }

        /**
        *  @brief increment counter and return last value.
        */
        __device__ unsigned get_particle_index() {
            return atomicAdd( &density, 1 );
        }

        __device__ unsigned add(Particle const& particle) {
            mean_velocity .atomicAdd(particle.velocity);
            centre_of_mass.atomicAdd(particle.position);
            return atomicAdd(&density, 1);
        }

        __device__ void unlocked_add(Particle const& particle) {
            density += 1;
            mean_velocity  += particle.velocity;
            centre_of_mass += particle.position;
        }

        __device__ void unlocked_subtract_velocity(Particle const& particle) {
            mean_velocity  -= particle.velocity;
        }

        __device__ void average() {
            centre_of_mass  *= Float(1.0) / density;
            mean_velocity   *= Float(1.0) / density;
        }

        __device__ Vector const get_correction(Vector const& position) const {
            return mean_velocity;
        }


        __device__ void add_reduce_only(Vector const& velocity) {
            atomicAdd( &density, 1 );
            mean_velocity.atomicAdd( velocity );
        }

        __device__ void average_reduce_only() { if(density > 0) { mean_velocity = mean_velocity / density; }}

        __device__ void clear() {
            density        = {};
            mean_velocity  = {};
            centre_of_mass = {};
        }

        __device__ void group_reduce(unsigned group_size) {
            if (group_size > 1) {
                density          = cuda::gpu_utilities::group_sum( density,         -1u, group_size );
                mean_velocity.x  = cuda::gpu_utilities::group_sum( mean_velocity.x,  -1u, group_size );
                mean_velocity.y  = cuda::gpu_utilities::group_sum( mean_velocity.y,  -1u, group_size );
                mean_velocity.z  = cuda::gpu_utilities::group_sum( mean_velocity.z,  -1u, group_size );
                centre_of_mass.x = cuda::gpu_utilities::group_sum( centre_of_mass.x, -1u, group_size );
                centre_of_mass.y = cuda::gpu_utilities::group_sum( centre_of_mass.y, -1u, group_size );
                centre_of_mass.z = cuda::gpu_utilities::group_sum( centre_of_mass.z, -1u, group_size );
            }
        }

    #endif // hide from gcc
    };

    template<typename T>
    static std::ostream& operator<< (std::ostream& os, MPCCell const& c)
    {
        os << c.density << ' ' << c.centre_of_mass << ' ' << c.mean_velocity;
        return os;
    }

    /**
    *  @brief Minimal fluid state struct. The 3rd moment is not neccesary because there is a thermostat.
    */
    struct FluidState {
        mpcd::Float  density;       // 1st moment
        mpcd::Vector mean_velocity; // 2nd moment
    };
} // namespace mpcd::cuda
