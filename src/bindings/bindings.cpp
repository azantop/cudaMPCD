#include <initializer_list>
#include <cstddef>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <mpcd/api/simulation_parameters.hpp>
#include <mpcd/api/simulation_handle.hpp>

namespace py = pybind11;

using mpcd::api::SimulationHandle;
using mpcd::SimulationParameters;
using mpcd::ExperimentType;
using mpcd::MPCDAlgorithm;

/*
 * Helper function to convert a vector to a NumPy array.
 * Assumes column-major order.
 */
 template<typename T>
 py::array_t<T> col_major_vector_to_numpy(std::vector<T>&& vec, std::vector<ssize_t> shape) {
     auto* heap_vec = new std::vector<T>(std::move(vec));
     py::capsule owner(heap_vec, [](void* p) { delete static_cast<std::vector<T>*>(p); });

     std::vector<size_t> strides(shape.size());
     strides.back() = sizeof(T);
     for (int i = shape.size() - 2; i >= 0; --i) {
         strides[i] = strides[i + 1] * shape[i + 1];
     }

     return py::array_t<T>(shape, strides, heap_vec->data(), owner);
 }

 /*
  * Helper function to convert a vector to a NumPy array.
  * Assumes column-major order.
  */
  template<typename T>
  py::array_t<T> vector_to_numpy(std::vector<T>&& vec, std::vector<ssize_t> shape, std::vector<ssize_t> strides) {
      auto* heap_vec = new std::vector<T>(std::move(vec));
      py::capsule owner(heap_vec, [](void* p) { delete static_cast<std::vector<T>*>(p); });
      return py::array_t<T>(shape, strides, heap_vec->data(), owner);
  }

PYBIND11_MODULE(_pympcd, m) {
    m.doc() = "Python bindings for Multi Particle Collision Dynamics (MPCD) Simulation.";

    py::class_<SimulationHandle>(m, "Simulation", "Handle class for managing simulation state and interaction with backend")
    .def(py::init<SimulationParameters const&, std::string>())
        .def("step", &SimulationHandle::step, "Perform n_steps simulation step.")
        .def("step_and_sample", &SimulationHandle::stepAndAccumulateSample, "Perform n_steps simulation step and accumulate statistics.")
        .def("get_mean_fields", [](SimulationHandle &self) {
            std::vector<float> density, velocity;
            self.getSampleMean(density, velocity);
            auto const& parameters = self.getParameters();

            // Construct shape and strides of 3D voxel arrays:
            ssize_t x = parameters.volume_size[0], y = parameters.volume_size[1], z = parameters.volume_size[2],
                    sf = static_cast<ssize_t>(sizeof(float));
            std::vector<ssize_t> shape = {x, y, z}, strides = {sf, y * sf, x * y * sf};

            auto density_np = vector_to_numpy(std::move(density), shape, strides);

            std::for_each(strides.begin(), strides.end(), [](ssize_t &stride) { stride *= 3; });
            strides.push_back(sizeof(float));
            shape.push_back(3);

            auto velocity_np = vector_to_numpy(std::move(velocity), shape, strides);

            return py::make_tuple(density_np, velocity_np);
        }, "Return the mean density and velocity fields (array of voxels) of the fluid after accumulating statistics.")
        .def("get_particle_positions", [](const SimulationHandle &self) {
            std::vector<float> positions;
            self.getParticlePositions(positions);  // return by const ref or copy
            // Reshape to 2D array of vectors with column-major order:
            ssize_t num_particles = positions.size() / 3, dims = 3;
            return col_major_vector_to_numpy(std::move(positions), {num_particles, dims});
        }, "Returns the positions of the particles as array of vectors. Data is copied from internal vectors into NumPy arrays.")
        .def("get_particle_velocities", [](const SimulationHandle &self) {
            std::vector<float> velocities;
            self.getParticleVelocities(velocities);  // return by const ref or copy
            // Reshape to 2D array of vectors with column-major order:
            ssize_t num_particles = velocities.size() / 3, dims = 3;
            return col_major_vector_to_numpy(std::move(velocities), {num_particles, dims});
        }, "Returns the velocities of the particles as array of vectors. Data is copied from internal vectors into NumPy arrays.");

    py::class_<SimulationParameters>(m, "Params", "Class representing simulation parameters")
        .def(py::init<>())
        .def_readwrite("device_id", &SimulationParameters::device_id)
        .def_readwrite("delta_t", &SimulationParameters::delta_t)
        .def_readwrite("N", &SimulationParameters::N)
        .def_readwrite("drag", &SimulationParameters::drag)
        .def_readwrite("equilibration_steps", &SimulationParameters::equilibration_steps)
        .def_readwrite("steps", &SimulationParameters::steps)
        .def_readwrite("sample_rate", &SimulationParameters::sample_every)
        .def_readwrite("average_samples", &SimulationParameters::average_samples)
        .def_property("temperature",
                    [](const SimulationParameters &p) { return p.temperature; },
                    [](SimulationParameters &p, double temperature) {
                        p.temperature = temperature;
                        p.thermal_velocity = std::sqrt(temperature);
                    }
                )
        .def_property("volume_size",
                    [](const SimulationParameters &p) { return py::make_tuple(p.volume_size[0], p.volume_size[1], p.volume_size[2]); },
                    [](SimulationParameters &p, py::tuple t) {
                        if (t.size() != 3)
                            throw std::runtime_error("Expected 3 elements");

                        for (int i = 0; i < 3; ++i) {
                            p.volume_size[i] = t[i].cast<float>();

                            if (p.periodicity[i] == 0)
                                p.volume_size[i] += 2; // add wall layer
                        }
                        p.N = p.n * (p.volume_size[0] - 2 * (1 - p.periodicity[0])) *
                                    (p.volume_size[1] - 2 * (1 - p.periodicity[1])) *
                                    (p.volume_size[2] - 2 * (1 - p.periodicity[2]));
                    }
                )
        .def_property("n",
                    [](const SimulationParameters &p) { return p.n; },
                    [](SimulationParameters &p, int n) {
                        p.n = n;
                        p.N = p.n * p.volume_size[0] * p.volume_size[1] * p.volume_size[2];
                    }
                )
        .def_property("periodicity",
                    [](const SimulationParameters &p) { return py::make_tuple(p.periodicity[0], p.periodicity[1], p.periodicity[2]); },
                    [](SimulationParameters &p, py::tuple t) {
                        if (t.size() != 3)
                            throw std::runtime_error("Expected 3 elements");

                        for (int i = 0; i < 3; ++i) {
                            p.periodicity[i] = t[i].cast<int>();

                            if (p.periodicity[i] == 0 and p.volume_size[i] != 0)
                                p.volume_size[i] += 2; // add wall layer if initialized
                        }
                        p.N = p.n * (p.volume_size[0] - 2 * (1 - p.periodicity[0])) *
                                    (p.volume_size[1] - 2 * (1 - p.periodicity[1])) *
                                    (p.volume_size[2] - 2 * (1 - p.periodicity[2]));                    }
                )
        .def_property("experiment",
                    [](const SimulationParameters &p) { return p.experiment == ExperimentType::standart ? "standart" : "channel"; },
                    [](SimulationParameters &p, const std::string &experiment) {
                        p.experiment = experiment == "standart" ? ExperimentType::standart : ExperimentType::channel;
                    }
                )
        .def_property("algorithm",
                    [](const SimulationParameters &p) { return p.algorithm == MPCDAlgorithm::srd ? "srd" : "extended"; },
                    [](SimulationParameters &p, const std::string &algorithm) {
                        p.algorithm = algorithm == "srd" ? MPCDAlgorithm::srd : MPCDAlgorithm::extended;
                    }
                );


    m.attr("__version__") = "1.0.0";
}
