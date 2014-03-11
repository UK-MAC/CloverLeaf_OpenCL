#if defined(MPI_HDR)
#include "mpi.h"
#endif
#include "ocl_common.hpp"
#include "ocl_strings.hpp"

#include <sstream>

CloverChunk chunk;

extern "C" void initialise_ocl_
(int* in_x_min, int* in_x_max,
 int* in_y_min, int* in_y_max,
 int* profiler_on)
{
    chunk = CloverChunk(in_x_min, in_x_max,
                        in_y_min, in_y_max,
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
 int* in_profiler_on)
:x_min(*in_x_min),
 x_max(*in_x_max),
 y_min(*in_y_min),
 y_max(*in_y_max),
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

    if (!rank)
    {
        fprintf(stdout, "Initialising OpenCL\n");
    }

#if defined(MPI_HDR)
    double t0 = MPI_Wtime();
#else
    double t0 = omp_get_wtime();
#endif

    initOcl();

    initProgram();
    initSizes();
    initReduction();
    initBuffers();
    initArgs();

#if defined(MPI_HDR)
    MPI_Barrier(MPI_COMM_WORLD);
    if (!rank)
    {
        fprintf(stdout, "Finished initialisation in %lf seconds\n", MPI_Wtime()-t0);
    }
#else
    if (!rank)
    {
        fprintf(stdout, "Finished initialisation in %lf seconds\n", omp_get_wtime()-t0);
    }
#endif
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
    int preferred_device = preferredDevice(input);
    fprintf(DBGOUT, "Preferred device is %d\n", preferred_device);
    desired_type = typeRead(input);

    fclose(input);

    size_t ii = 0;

    // go through all platforms
    while(1)
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

    // try to create a context with the desired type
    cl_context_properties properties[3] = {CL_CONTEXT_PLATFORM,
        reinterpret_cast<cl_context_properties>(platform()), 0};

    try
    {
        context = cl::Context(desired_type, properties);
    }
    catch (cl::Error e)
    {
        // if there's no device of the desired type in this context
        if (e.err() == CL_DEVICE_NOT_FOUND
        || e.err() == CL_DEVICE_NOT_AVAILABLE)
        {
            fprintf(stderr, "No devices of specified type found:\n");
            std::vector<cl::Device> devices;
            platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

            for (int ii = 0; ii < devices.size(); ii++)
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
            DIE("Error in creating context %d\n", e.err());
        }
    }

#if defined(MPI_HDR)
    // gets devices one at a time to prevent conflicts (on emerald)
    int ranks, rank, cur_rank = 0;

    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    do
    {
        if (rank == cur_rank)
        {
#endif
            // get devices - just choose the first one
            std::vector<cl::Device> devices;
            context.getInfo(CL_CONTEXT_DEVICES, &devices);

            std::string devname;

            if (preferred_device < 0)
            {
                // if none specified or invalid choice, choose 0
                fprintf(stdout,
                    "No device specified, choosing device 0\n");
                device = devices.at(0);
            }
            else if (preferred_device+rank > devices.size())
            {
                // if preferred does not exist, choose 0 and warn
                fprintf(stderr,
                    "WARNING - device %d does not exist as there are only %zu available - choosing 0\n",
                    preferred_device, devices.size());
                device = devices.at(0);
            }
            else
            {
                device = devices.at(preferred_device+rank);
            }

            device.getInfo(CL_DEVICE_NAME, &devname);

#if defined(MPI_HDR)
            fprintf(stdout, "OpenCL using %s in rank %d\n", devname.c_str(), rank);
#else
            fprintf(stdout, "OpenCL using %s\n", devname.c_str());
#endif
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
                device_type_prepro = "-DNODEVICETYPE ";
                break;
            }
#if defined(MPI_HDR)
        }
        MPI_Barrier(MPI_COMM_WORLD);
    } while ((cur_rank++) < ranks);
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

