//#include <sys/syslimits.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>

#include <cassert>
#include <string>
#include <fstream>
#include <iostream>

#include <cuda_runtime_api.h>

#include <mpcd/common/simulation_parameters.hpp>

namespace mpcd {
    SimulationParameters::SimulationParameters(std::string const& file_name, bool read_only)
    {
        std::string   parameter_name,
                    branch;

        //std::unordered_map< std::string, std::any > parameter_name_map;

        std::ifstream input_file( file_name );

        if( !input_file )
        {
            std::cout << "no such input_file: " << file_name << std::endl;
            exit(0);
        }

        while ( parameter_name != "general" )
            input_file >> parameter_name;

        std::getline( input_file, parameter_name );

        input_file >> parameter_name >> output_directory;

        input_file >> parameter_name >> branch;
        if ( branch == "standart" )
            experiment = ExperimentType::standart;

        input_file >> parameter_name >> volume_size[0] >> volume_size[1] >> volume_size[2];
        input_file >> parameter_name >> periodicity[0] >> periodicity[1] >> periodicity[2];

        for (int i = 0; i < 3; i++)
            volume_size[i] -= ( periodicity[i] - 1 ) * 2; // add wall layer

        // skip 3 lines:
        std::getline( input_file, parameter_name ); std::getline( input_file, parameter_name ); std::getline( input_file, parameter_name );

        // fluid parameters:

        input_file >> parameter_name >> branch;
        if ( branch == "srd" )
            algorithm = MPCDAlgorithm::srd;
        else if ( branch == "extended" )
            algorithm = MPCDAlgorithm::extended;

        std::cout << "using " << branch << " MPCD algorithm" << std::endl;

        input_file >> parameter_name >> n;
        input_file >> parameter_name >> temperature;
        input_file >> parameter_name >> delta_t;
        input_file >> parameter_name >> drag;

        std::getline( input_file, parameter_name ); std::getline( input_file, parameter_name ); std::getline( input_file, parameter_name );

        // simulation parameters:

        input_file >> parameter_name >> equilibration_steps;
        input_file >> parameter_name >> steps;
        input_file >> parameter_name >> sample_every;
        input_file >> parameter_name >> average_samples;
        input_file >> parameter_name >> branch; write_backup             = ( branch == "yes" );
        input_file >> parameter_name >> branch; read_backup              = ( branch == "yes" );

        if ( branch != "yes" and branch != "no" ) // load status from directory
        {
            read_backup = true;
            load_directory = branch;
        }
        else
            load_directory = "./";

        input_file >> parameter_name >> device_id;

        // ----------- implementation hints etc:

        if ( experiment == ExperimentType::channel )
        {
            std::cout << "implementation of ghost particles not complete for this geometry. program end..." << std::endl;
            std::exit( 0 );
        }

        // -----------

        if ( not read_only ) // create simulation output folder given in the input file
        {
            std::string code_id = "";

            std::ifstream commit_info;
            commit_info.open( "commit_info" );

            auto ptr = get_current_dir_name();
            auto dp  = opendir( ( ptr + std::string("/") + output_directory ).c_str() );
            free( ptr );

            if ( dp )
            {
                std::cout << "using directory " << output_directory << " ..." << std::endl;

                closedir( dp );
                if ( false )
                {
                    std::cerr << "overwrite existing data? y/n: " << std::flush;
                    std::string s;
                    std::cin >> s;
                    if ( s[0] == 'n' )
                        exit( 1 );
                }

                if ( 0 != chdir( ( get_current_dir_name() + std::string("/") +  output_directory ).c_str() ) )
                {
                    std::cout << "problem changinng into the requested directory... exiting..." << std::endl;
                    std::exit( 0 );
                }
            }
            else
            {
                std::cout << "creating new directory " << output_directory << " ..." << std::endl;

                int pos;
                for (;;)
                {
                    pos = output_directory.find( "/" );
                    if( pos == -1 )
                    break;

                    std::string folder = output_directory.substr( 0, pos );
                    if ( 0 != mkdir( ( get_current_dir_name() + std::string("/") + folder ).c_str(), 0777 ) ) { }
                    if ( 0 != chdir( ( get_current_dir_name() + std::string("/") + folder ).c_str() ) )
                    {
                        std::cout << "problem cd-ing into the requested directory... exiting..." << std::endl;
                        std::exit( 0 );
                    }
                    output_directory = output_directory.substr( pos+1 );
                }

                if ( 0 != mkdir( ( get_current_dir_name() + std::string("/") + output_directory ).c_str(), 0777 ) ) { }
                if ( 0 != chdir( ( get_current_dir_name() + std::string("/") + output_directory ).c_str() ) )
                {
                    std::cout << "problem cd-ing into the requested directory... exiting..." << std::endl;
                    std::exit( 0 );
                }
            }

            input_file.clear();
            input_file.seekg( 0 );

            // add info file to the new directory

            std::ofstream info_file( "simulation.info" );

            /*
            if ( not commit_info )
            {
                std::cout << "commit_info file missing! terminating here!" << std::endl;
                exit( 0 );
            }

            while ( !commit_info.eof() )
            {
                char c;
                commit_info.get( c );
                info_file << c;
            }*/

            info_file << "simulation parameters where: \n";

            while ( !input_file.eof() )
            {
                char c;
                input_file.get( c );
                info_file << c;
            }
        }

        N = n * volume_size[0] * volume_size[1] * volume_size[2];
        thermal_velocity = std::sqrt( temperature );
    }
} // namespace mpcd
