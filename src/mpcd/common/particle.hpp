#pragma once
#include "common/vector_3d.hpp"

namespace mpcd {
    struct alignas(16) Particle {
        using Vector = math::Vector;

        uint16_t flags,
                 cidx;
        Vector   position,
                 velocity;
        uint32_t cell_idx;

        void writeBinary(std::ofstream &stream) const {
            stream.write((char*) &position, sizeof(Vector) * 2);
        }

        void readBinary(std::ifstream &stream) {
            stream.read((char*) &position, sizeof(Vector) * 2);
        }
    };
} // namespace mpcd::cuda
