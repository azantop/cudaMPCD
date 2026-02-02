#include <vector>
#include <mpcd/api/simulation_parameters.hpp>

#include "common/vector_3d.hpp"
#include "common/particle.hpp"
#include "common/random.hpp"
#include "common/mpc_cell.hpp"

#include "cpu_backend/volume_container.hpp"

namespace mpcd::cpu {
    void distribute_particles(std::vector<Particle>&, VolumeContainer<MPCCell>& mpc_cells, SimulationParameters const&, Xoshiro128Plus&);

    // MPC streaming step kernel
    void translate_particles(std::vector<Particle>& particles, VolumeContainer<MPCCell>& mpc_cells,
                                        Xoshiro128Plus& random, mpcd::Vector grid_shift,
                                        mpcd::Vector volume_size, mpcd::IntVector periodicity, mpcd::Float delta_t,
                                        ExperimentType experiment,
                                        std::vector<uint32_t>& uniform_counter, std::vector<uint32_t>& uniform_list);

    void srd_collision(std::vector<Particle>& particles, VolumeContainer<MPCCell>& mpc_cells,
                                        Xoshiro128Plus& random, mpcd::Vector grid_shift,
                                        mpcd::Vector volume_size, mpcd::IntVector periodicity, mpcd::Float delta_t,
                                        mpcd::Float drag, mpcd::Float thermal_velocity, uint32_t n_density,
                                        std::vector<uint32_t>& uniform_counter, std::vector<uint32_t>& uniform_list);
}
