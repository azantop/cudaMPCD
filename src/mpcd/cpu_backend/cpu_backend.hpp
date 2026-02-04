#pragma once

#include <vector>
#include <mpcd/api/simulation_parameters.hpp>

#include "backend/backend.hpp"
#include "common/vector_3d.hpp"
#include "common/particle.hpp"
#include "common/mpc_cell.hpp"
#include "common/random.hpp"

#include "cpu_backend/volume_container.hpp"

namespace mpcd::cpu {

    class CPUBackend : public Backend
    {
        using Vector = mpcd::Vector;
        using Float  = mpcd::Float;

        std::vector<Particle>          particles;   // SRD fluid particles

        Vector                         grid_shift;  // SRD grid shift

        VolumeContainer<MPCCell>       mpc_cells;   // SRD cell storage
        std::vector<FluidState>        cell_states, // for averaging over the fluid state
                                       kahan_c;

        // The indices for fluid particles are stored in a lookup table for the collision step.
        // This optimizes the data througput, because particles can be stored in shared memory
        // and only need to be loaded once:
        std::vector<uint32_t>          uniform_list,    // the index lookup
                                       uniform_counter; // next free table entry, used with atomicAdd.

        Xoshiro128Plus                 random;     // random number generatofor the cpu


        // To furthe optimize memory loading, the particle array is sorted according to the SRD cell-index.
        // This enables array striding, ie. coalesced memory loading:
        size_t step_counter, resort_rate;

        enum SamplingState {
            SAMPLING_IN_PROGRESS,
            SAMPLING_COMPLETED
        };
        SamplingState sampling_state;
        size_t        sample_counter;

        void translationStep();  // SRD streaming step
        void collisionStep();    // SRD collision step

        public:

        // routines:
        CPUBackend(SimulationParameters const&);  // initialization

        // data io:
        void writeSample();
        void writeBackupFile();

        // backend overrides:
        void step(int n_steps) override;
        void stepAndAccumulateSample(int n_steps) override;
        void getSampleMean(std::vector<float>& mean_density, std::vector<float>& mean_velocity) override;
        size_t getNParticles() override;
        void getParticlePositions(std::vector<float>& positions) override;
        void getParticleVelocities(std::vector<float>& velocities) override;
        void setParticlePositions(std::vector<float> const& positions) override;
        void setParticleVelocities(std::vector<float> const& velocities) override;
        void setParticles(std::vector<float> const& positions, std::vector<float> const& velocities) override;
    };
} // namespace mpcd::cuda
