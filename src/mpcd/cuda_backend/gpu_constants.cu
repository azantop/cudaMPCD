
#include "parameter_set.hpp"
#include "device_volume_container"
#include "particle.hpp"
#include "mpc_cell.hpp"
#include "gpu_random.hpp"

namespace gpu_const
{
    __constant__ parameter_set                             parameters;
    __constant__ DeviceVolumeContainer< MPCCell > mpc_cells;
    __constant__ async_vector< particle_type >             particles;
    __constant__ gpu_vector< Xoshiro128Plus >              generator;
    __constant__ gpu_vector< uint32_t >                    uniform_list,
                                                           uniform_counter;
}
