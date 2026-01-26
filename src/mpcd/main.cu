#include <string>
#include <iostream>

#include <mpcd/common/simulation_parameters.hpp>

#include "common/probing.hpp"
#include "common/status_io.hpp"
#include "cuda_backend/cuda_backend.hpp"


int main(int argc, char **argv) {
    using namespace mpcd;

    // --- initialize:
    if (argc != 2) {
        std::cout << "Please start the program with the name of an input file as parameter. The program is now exiting." << std::endl;
        return -1;
    }

    SimulationParameters parameters(std::string(argv[1]), false);
    cuda::CudaBackend simulation(parameters);

    // --- simulation:
    if (not parameters.read_backup)
    {
        status::report("equilibration");
        for (size_t equilibration_step = 0; equilibration_step < parameters.equilibration_steps; ++equilibration_step)
        {
            simulation.step(1);

            if (!(equilibration_step % 100))
                status::update(equilibration_step, parameters.equilibration_steps);
        }
        status::report_done();
    }

    status::report("sampling");
    size_t start = (parameters.read_backup && (parameters.load_directory == "./")) ? status::read_time_file(parameters.load_directory) + 1 : 0;
    for (size_t step = start; step < parameters.steps; ++step)
    {
        simulation.step(1);

        if (do_sampling(step, parameters))
            simulation.writeSample();

        if (!(step % 100))
            status::update(step, parameters.steps);

        if (status::time_out || (parameters.write_backup
            &&!(step % (parameters.volume_size[0] * parameters.volume_size[1] * parameters.volume_size[2] * parameters.n > 10e7 ? 100'000 : 500'000))
            && step != 0))
        {
            status::write_time_file(step);
            simulation.writeBackupFile();

            if ( status::time_out )
                std::exit( 0x0 );
        }
    }
    status::report_done();
    status::write_time_file(parameters.steps);

    // --- final data output:
    if (parameters.steps and (start < parameters.steps))
        simulation.writeSample();

    if (parameters.write_backup)
        simulation.writeBackupFile();

    return 0x0;
}
