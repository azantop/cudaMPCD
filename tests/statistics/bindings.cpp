#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "common/random.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_test_statistics, m) {
    m.doc() = "Random number generator testing utilities";

    py::class_<mpcd::Xoshiro128Plus>(m, "Xoshiro128Plus")
        .def(py::init<>())
        .def("seed", &mpcd::Xoshiro128Plus::seed)
        .def("uniform", &mpcd::Xoshiro128Plus::genUniformFloat)
        .def("gaussian", &mpcd::Xoshiro128Plus::gaussianf);

    m.def("generate_uniform_samples", [](size_t n, uint64_t seed_a, uint64_t seed_b) {
        mpcd::Xoshiro128Plus rng;
        rng.seed(seed_a, seed_b);

        std::vector<float> samples;
        samples.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            samples.push_back(rng.genUniformFloat());
        }
        return samples;
    });

    m.def("generate_gaussian_samples", [](size_t n, uint64_t seed_a, uint64_t seed_b) {
        mpcd::Xoshiro128Plus rng;
        rng.seed(seed_a, seed_b);

        std::vector<float> samples;
        samples.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            samples.push_back(rng.gaussianf());
        }
        return samples;
    });
}
