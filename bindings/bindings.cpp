#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <mpcd/common/simulation_parameters.hpp>
#include <mpcd/api/simulation_handle.hpp>

namespace py = pybind11;

using mpcd::api::SimulationHandle;
using mpcd::SimulationParameters;
using mpcd::ExperimentType;
using mpcd::MPCDAlgorithm;

template<typename T>
py::array_t<T> vector_to_numpy(std::vector<T>&& vec) {
    auto* heap_vec = new std::vector<T>(std::move(vec));
    py::capsule owner(heap_vec, [](void* p) { delete static_cast<std::vector<T>*>(p); });
    return py::array_t<T>(heap_vec->size(), heap_vec->data(), owner);
}

PYBIND11_MODULE(simulator_bindings, m) {
    m.doc() = "Python bindings for CUDA AsyncVector with numpy integration";

    py::class_<SimulationHandle>(m, "SimulationHandle")
    .def(py::init<SimulationParameters const&, std::string>())
        .def("step", &SimulationHandle::step)
        .def("stepAndAccumulateSample", &SimulationHandle::stepAndAccumulateSample)
        .def("getSampleMean", [](SimulationHandle &self) {
            std::vector<float> density, velocity;
            self.getSampleMean(density, velocity);
            return py::make_tuple(vector_to_numpy(std::move(density)), vector_to_numpy(std::move(velocity)));
        })
        .def("getParticlePositions", [](const SimulationHandle &self) {
            std::vector<float> positions;
            self.getParticlePositions(positions);  // return by const ref or copy
            return vector_to_numpy(std::move(positions));
        })
        .def("getParticleVelocities", [](const SimulationHandle &self) {
            std::vector<float> velocities;
            self.getParticleVelocities(velocities);  // return by const ref or copy
            return vector_to_numpy(std::move(velocities));
        });

    py::class_<SimulationParameters>(m, "SimulationParameters")
        .def(py::init<>())
        .def_readwrite("device_id", &SimulationParameters::device_id)
        .def_readwrite("delta_t", &SimulationParameters::delta_t)
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
