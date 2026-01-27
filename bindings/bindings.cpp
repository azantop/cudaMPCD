#include <initializer_list>
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
template<typename T, typename U>
py::array_t<T> vector_to_numpy(std::vector<T>&& vec, std::initializer_list<U>&& shape ) {
    auto* heap_vec = new std::vector<T>(std::move(vec));
    py::capsule owner(heap_vec, [](void* p) { delete static_cast<std::vector<T>*>(p); });
    return py::array_t<T>(shape, heap_vec->data(), owner);
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

            // Reshape to 3D voxel arrays with column-major order:
            size_t x = parameters.volume_size[0], y = parameters.volume_size[1], z = parameters.volume_size[2];
            return py::make_tuple(vector_to_numpy(std::move(density), {x, y, z}),
                                  vector_to_numpy(std::move(velocity), {x, y, z, 3ul}));
        }, "Return the mean density and velocity fields (array of voxels) of the fluid after accumulating statistics.")
        .def("get_particle_positions", [](const SimulationHandle &self) {
            std::vector<float> positions;
            self.getParticlePositions(positions);  // return by const ref or copy
            // Reshape to 2D array of vectors with column-major order:
            return vector_to_numpy(std::move(positions), {positions.size(), 3ul});
        }, "Returns the positions of the particles as array of vectors. Data is copied from internal vectors into NumPy arrays.")
        .def("get_particle_velocities", [](const SimulationHandle &self) {
            std::vector<float> velocities;
            self.getParticleVelocities(velocities);  // return by const ref or copy
            // Reshape to 2D array of vectors with column-major order:
            return vector_to_numpy(std::move(velocities), {velocities.size(), 3ul});
        }, "Returns the velocities of the particles as array of vectors. Data is copied from internal vectors into NumPy arrays.");

    py::class_<SimulationParameters>(m, "Params", "Class representing simulation parameters")
        .def(py::init<>())
        .def_readwrite("device_id", &SimulationParameters::device_id)
        .def_readwrite("delta_t", &SimulationParameters::delta_t)
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
                        p.volume_size[0] = t[0].cast<int>();
                        p.volume_size[1] = t[1].cast<int>();
                        p.volume_size[2] = t[2].cast<int>();
                        p.N = p.n * p.volume_size[0] * p.volume_size[1] * p.volume_size[2];
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
                        p.periodicity[0] = t[0].cast<bool>();
                        p.periodicity[1] = t[1].cast<bool>();
                        p.periodicity[2] = t[2].cast<bool>();
                    }
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
