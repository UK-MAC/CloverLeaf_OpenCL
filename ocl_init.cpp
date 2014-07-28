#if defined(MPI_HDR)
#include "mpi.h"
#endif
#include "ocl_common.hpp"
#include "ocl_strings.hpp"

#include <sstream>
#include <iostream>

CloverChunk chunk;

extern "C" void initialise_ocl_
(int* in_x_min, int* in_x_max,
 int* in_y_min, int* in_y_max,
 int* in_z_min, int* in_z_max,
 int* profiler_on)
{
    chunk = CloverChunk(in_x_min, in_x_max,
                        in_y_min, in_y_max,
                        in_z_min, in_z_max,
                        profiler_on);
}

// default ctor
CloverChunk::CloverChunk
(void)
{
    ;
}

extern "C" double omp_get_wtime();

CloverChunk::CloverChunk
(int* in_x_min, int* in_x_max,
 int* in_y_min, int* in_y_max,
 int* in_z_min, int* in_z_max,
 int* in_profiler_on)
:x_min(*in_x_min),
 x_max(*in_x_max),
 y_min(*in_y_min),
 y_max(*in_y_max),
 z_min(*in_z_min),
 z_max(*in_z_max),
 profiler_on(*in_profiler_on)
{
#ifdef OCL_VERBOSE
    DBGOUT = stdout;
#else
    if (NULL == (DBGOUT = fopen("/dev/null", "w")))
    {
        DIE("Unable to open /dev/null to discard output\n");
    }
#endif

#if defined(MPI_HDR)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    rank = 0;
#endif

    double t0 = omp_get_wtime();

    if (!rank)
    {
        fprintf(stdout, "Initialising OpenCL\n");
    }

    initOcl();
    initProgram();
    initSizes();
    initReduction();
    initBuffers();
    initArgs();

#if defined(MPI_HDR)
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    if (!rank)
    {
        fprintf(stdout, "Finished initialisation in %f seconds\n", omp_get_wtime()-t0);
    }
}

void CloverChunk::initOcl
(void)
{
    std::vector<cl::Platform> platforms;
    try
    {
        cl::Platform::get(&platforms);
    }
    catch (cl::Error e)
    {
        DIE("Error in fetching platforms (%s), error %d\n", e.what(), e.err());
    }

    if (platforms.size() < 1)
    {
        DIE("No platforms found\n");
    }

    // Read in from file - easier than passing in from fortran
    FILE* input = fopen("clover.in", "r");
    if (NULL == input)
    {
        // should never happen
        DIE("Input file not found\n");
    }

    int desired_vendor = platformRead(input);


    if (desired_vendor == NO_PLAT)
    {
        DIE("No platform specified in tea.in\n");
    }
    else if (desired_vendor == LIST_PLAT)
    {
        // special case to print out platforms instead
        fprintf(stdout, "Listing platforms\n\n");

        for (size_t pp = 0; pp < platforms.size(); pp++)
        {
            std::string profile, version, name, vendor;
            platforms.at(pp).getInfo(CL_PLATFORM_PROFILE, &profile);
            platforms.at(pp).getInfo(CL_PLATFORM_VERSION, &version);
            platforms.at(pp).getInfo(CL_PLATFORM_NAME, &name);
            platforms.at(pp).getInfo(CL_PLATFORM_VENDOR, &vendor);

            fprintf(stdout, "Platform %zu: %s - %s (profile = %s, version = %s)\n",
                pp, vendor.c_str(), name.c_str(), profile.c_str(), version.c_str());

            std::vector<cl::Device> devices;
            platforms.at(pp).getDevices(CL_DEVICE_TYPE_ALL, &devices);

            for (size_t ii = 0; ii < devices.size(); ii++)
            {
                std::string devname;
                cl_device_type dtype;
                devices.at(ii).getInfo(CL_DEVICE_NAME, &devname);
                devices.at(ii).getInfo(CL_DEVICE_TYPE, &dtype);
                // trim whitespace
                devname.erase(devname.find_last_not_of(" \n\r\t")+1);
                devname.erase(devname.begin(), devname.begin()+devname.find_first_not_of(" \n\r\t"));

                std::string dtype_str = strType(dtype);
                fprintf(stdout, " Device %zu: %s (%s)\n", ii, devname.c_str(), dtype_str.c_str());
            }
        }

        exit(0);
    }
    else if (desired_vendor == ANY_PLAT)
    {
#if defined(MPI_HDR)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
        fprintf(stdout, "No platform specified - using platform %d in rank %d\n",
            rank, rank);
        platform = platforms.at(rank);
    }
    else
    {
        // go through all platforms
        for (size_t ii = 0;;)
        {
            std::string plat_name;
            platforms.at(ii).getInfo(CL_PLATFORM_VENDOR, &plat_name);
            fprintf(DBGOUT, "Checking platform %s\n", plat_name.c_str());

            // if the platform name given matches one in the LUT
            if (platformMatch(plat_name) == desired_vendor)
            {
                fprintf(DBGOUT, "correct vendor platform found\n");
                platform = platforms.at(ii);
                break;
            }

            // if there are no platforms left to match
            if (platforms.size() == ++ii)
            {
                DIE("correct vendor platform NOT found\n");
            }
        }
    }

    int preferred_device = preferredDevice(input);
    preferred_device = (preferred_device < 0) ? 0 : preferred_device;
    fprintf(DBGOUT, "Preferred device is %d\n", preferred_device);
    bool usefirst = paramEnabled(input, "opencl_usefirst");
    desired_type = typeRead(input);
    fclose(input);

    // try to create a context with the desired type
    cl_context_properties properties[3] = {CL_CONTEXT_PLATFORM,
        reinterpret_cast<cl_context_properties>(platform()), 0};

    try
    {
        context = cl::Context(desired_type, properties);
    }
    catch (cl::Error e)
    {
        if (e.err() == CL_DEVICE_NOT_AVAILABLE)
        {
            DIE("Devices found but are not available (CL_DEVICE_NOT_AVAILABLE)\n");
        }
        // if there's no device of the desired type in this context
        else if (e.err() == CL_DEVICE_NOT_FOUND)
        {
            fprintf(stderr, "No devices of specified type found:\n");
            std::vector<cl::Device> devices;
            platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

            for (size_t ii = 0; ii < devices.size(); ii++)
            {
                std::string devname;
                cl_device_type dtype;
                devices.at(ii).getInfo(CL_DEVICE_NAME, &devname);
                devices.at(ii).getInfo(CL_DEVICE_TYPE, &dtype);

                std::string dtype_str = strType(dtype);
                fprintf(stderr, "%s (%s)\n", devname.c_str(), dtype_str.c_str());
            }

            DIE("Unable to get devices of desired type");
        }
        else
        {
            DIE("Error %d (%s) in creating context\n", e.err(), e.what());
        }
    }

#if defined(MPI_HDR)
    // gets devices one at a time to prevent conflicts (on emerald)
    int ranks, cur_rank = 0;

    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    do
    {
        if (rank == cur_rank)
        {
#endif
            // index of device to use
            int actual_device = 0;

            // get devices - just choose the first one
            std::vector<cl::Device> devices;
            context.getInfo(CL_CONTEXT_DEVICES, &devices);

            if (usefirst)
            {
                // always use specified device and ignore rank
                actual_device = preferred_device;
            }
            else
            {
                actual_device = preferred_device + (rank % devices.size());
            }

            std::string devname;

            if (preferred_device < 0)
            {
                // if none specified or invalid choice, choose 0
                fprintf(stdout, "No device specified, choosing device 0\n");
                actual_device = 0;
                device = devices.at(actual_device);
            }
            else if (actual_device >= devices.size())
            {
                DIE("Device %d was selected in rank %d but there are only %zu available\n",
                    actual_device, rank, devices.size());
            }
            else
            {
                device = devices.at(actual_device);
            }

            device.getInfo(CL_DEVICE_NAME, &devname);

            fprintf(stdout, "OpenCL using device %d (%s) in rank %d\n",
                actual_device, devname.c_str(), rank);

            // choose reduction based on device type
            switch (desired_type)
            {
            case CL_DEVICE_TYPE_GPU : 
                device_type_prepro = "-DCL_DEVICE_TYPE_GPU ";
                break;
            case CL_DEVICE_TYPE_CPU : 
                device_type_prepro = "-DCL_DEVICE_TYPE_CPU ";
                break;
            case CL_DEVICE_TYPE_ACCELERATOR : 
                device_type_prepro = "-DCL_DEVICE_TYPE_ACCELERATOR ";
                break;
            default :
                device_type_prepro = "-DCL_DEVICE_TYPE_GPU ";
                break;
            }
#if defined(MPI_HDR)
        }
        MPI_Barrier(MPI_COMM_WORLD);
    } while ((cur_rank++) < ranks);

    MPI_Barrier(MPI_COMM_WORLD);
#endif

    // initialise command queue
    if (profiler_on)
    {
        // turn on profiling
        queue = cl::CommandQueue(context, device,
                                 CL_QUEUE_PROFILING_ENABLE, NULL);
    }
    else
    {
        queue = cl::CommandQueue(context, device);
    }
}


