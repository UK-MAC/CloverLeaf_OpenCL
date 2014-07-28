#if defined(MPI_HDR)
#include "mpi.h"
#endif
#include "ocl_common.hpp"

#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <cstdarg>

std::string errToString(cl_int err)
{
    switch (err) {
        case CL_SUCCESS:                            return "Success!";
        case CL_DEVICE_NOT_FOUND:                   return "Device not found.";
        case CL_DEVICE_NOT_AVAILABLE:               return "Device not available";
        case CL_COMPILER_NOT_AVAILABLE:             return "Compiler not available";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "Memory object allocation failure";
        case CL_OUT_OF_RESOURCES:                   return "Out of resources";
        case CL_OUT_OF_HOST_MEMORY:                 return "Out of host memory";
        case CL_PROFILING_INFO_NOT_AVAILABLE:       return "Profiling information not available";
        case CL_MEM_COPY_OVERLAP:                   return "Memory copy overlap";
        case CL_IMAGE_FORMAT_MISMATCH:              return "Image format mismatch";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "Image format not supported";
        case CL_BUILD_PROGRAM_FAILURE:              return "Program build failure";
        case CL_MAP_FAILURE:                        return "Map failure";
        case CL_INVALID_VALUE:                      return "Invalid value";
        case CL_INVALID_DEVICE_TYPE:                return "Invalid device type";
        case CL_INVALID_PLATFORM:                   return "Invalid platform";
        //case CL_INVALID_PROPERTY:                   return "Invalid property";
        case CL_INVALID_DEVICE:                     return "Invalid device";
        case CL_INVALID_CONTEXT:                    return "Invalid context";
        case CL_INVALID_QUEUE_PROPERTIES:           return "Invalid queue properties";
        case CL_INVALID_COMMAND_QUEUE:              return "Invalid command queue";
        case CL_INVALID_HOST_PTR:                   return "Invalid host pointer";
        case CL_INVALID_MEM_OBJECT:                 return "Invalid memory object";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "Invalid image format descriptor";
        case CL_INVALID_IMAGE_SIZE:                 return "Invalid image size";
        case CL_INVALID_SAMPLER:                    return "Invalid sampler";
        case CL_INVALID_BINARY:                     return "Invalid binary";
        case CL_INVALID_BUILD_OPTIONS:              return "Invalid build options";
        case CL_INVALID_PROGRAM:                    return "Invalid program";
        case CL_INVALID_PROGRAM_EXECUTABLE:         return "Invalid program executable";
        case CL_INVALID_KERNEL_NAME:                return "Invalid kernel name";
        case CL_INVALID_KERNEL_DEFINITION:          return "Invalid kernel definition";
        case CL_INVALID_KERNEL:                     return "Invalid kernel";
        case CL_INVALID_ARG_INDEX:                  return "Invalid argument index";
        case CL_INVALID_ARG_VALUE:                  return "Invalid argument value";
        case CL_INVALID_ARG_SIZE:                   return "Invalid argument size";
        case CL_INVALID_KERNEL_ARGS:                return "Invalid kernel arguments";
        case CL_INVALID_WORK_DIMENSION:             return "Invalid work dimension";
        case CL_INVALID_WORK_GROUP_SIZE:            return "Invalid work group size";
        case CL_INVALID_WORK_ITEM_SIZE:             return "Invalid work item size";
        case CL_INVALID_GLOBAL_OFFSET:              return "Invalid global offset";
        case CL_INVALID_EVENT_WAIT_LIST:            return "Invalid event wait list";
        case CL_INVALID_EVENT:                      return "Invalid event";
        case CL_INVALID_OPERATION:                  return "Invalid operation";
        case CL_INVALID_GL_OBJECT:                  return "Invalid OpenGL object";
        case CL_INVALID_BUFFER_SIZE:                return "Invalid buffer size";
        case CL_INVALID_MIP_LEVEL:                  return "Invalid mip-map level";
        default: return "Unknown";
    }
}

void CloverChunk::enqueueKernel
(cl::Kernel const& kernel,
 int line, const char* file,
 const cl::NDRange offset_range,
 const cl::NDRange global_range,
 const cl::NDRange local_range,
 const std::vector< cl::Event > * const events,
 cl::Event * const event)
{
    try
    {
        if (profiler_on)
        {
            // time it
            cl::Event *prof_event;
            cl_ulong start, end;

            // used if no event was passed
            cl::Event no_event_passed = cl::Event();

            if (event != NULL)
            {
                prof_event = event;
            }
            else
            {
                prof_event = &no_event_passed;
            }

            std::string func_name;
            kernel.getInfo(CL_KERNEL_FUNCTION_NAME, &func_name);

            #if 0
            fprintf(stdout, "Enqueueing kernel: %s\n", func_name.c_str());
            fprintf(stdout, "%zu global dimensions\n", global_range.dimensions());
            fprintf(stdout, "%zu local dimensions\n", local_range.dimensions());
            fprintf(stdout, "%zu offset dimensions\n", offset_range.dimensions());
            fprintf(stdout, "Global size: [%zu %zu %zu]\n", global_range[0],global_range[1], global_range[2]);
            fprintf(stdout, "Local size:  [%zu %zu %zu]\n", local_range[0], local_range[1], local_range[2]);
            fprintf(stdout, "Offset size: [%zu %zu %zu]\n", offset_range[0],offset_range[1], offset_range[2]);
            fprintf(stdout, "\n");
            fflush(stdout);
            #endif

            queue.enqueueNDRangeKernel(kernel,
                                       offset_range,
                                       global_range,
                                       local_range,
                                       events,
                                       prof_event);
            prof_event->wait();

            prof_event->getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
            prof_event->getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
            double taken = static_cast<double>(end-start)*1.0e-6;

            if (kernel_times.end() != kernel_times.find(func_name))
            {
                kernel_times.at(func_name) += taken;
            }
            else
            {
                kernel_times[func_name] = taken;
            }
        }
        else
        {
            // just launch kernel
            queue.enqueueNDRangeKernel(kernel,
                                       offset_range,
                                       global_range,
                                       local_range,
                                       events,
                                       event);
        }
    }
    catch (cl::Error e)
    {
        std::string func_name;
        kernel.getInfo(CL_KERNEL_FUNCTION_NAME, &func_name);

        // invalid work group size
        if (e.err() == -54)
        {
            std::stringstream errstr;
            errstr << "Error in enqueueing kernel " << func_name;
            errstr << " at line " << line << " in " << file << std::endl;
            errstr << errToString(e.err()).c_str() << std::endl;

            errstr << "Launched with ";
            errstr << global_range.dimensions() << " global dimensions, ";
            errstr << local_range.dimensions() << " local dimensions." << std::endl;

            for (unsigned int ii = 0; ii < global_range.dimensions(); ii++)
            {
                errstr << "Launch dimension " << ii << ": ";
                errstr << "global " << global_range[ii] << ", ";
                errstr << "local " << local_range[ii] << " ";
                errstr << "(offset " << offset_range[ii] << ") - ";
                errstr << "(" << global_range[ii] << "%" << local_range[ii] << ") ";
                errstr << "= " << global_range[ii] % local_range[ii] << std::endl;
            }

            DIE(errstr.str().c_str());
        }
        else
        {
            DIE("Error in enqueueing kernel '%s' at line %d in %s\n"
                "Error in %s, code %d (%s) - exiting\n",
                 func_name.c_str(), line, file,
                 e.what(), e.err(), errToString(e.err()).c_str());
        }
    }
}

// called when something goes wrong
void CloverChunk::cloverDie
(int line, const char* filename, const char* format, ...)
{
    fprintf(stderr, "@@@@@\n");
    fprintf(stderr, "\x1b[31m");
    fprintf(stderr, "Fatal error at line %d in %s:", line, filename);
    fprintf(stderr, "\x1b[0m");
    fprintf(stderr, "\n");

    va_list arglist;
    va_start(arglist, format);
    vfprintf(stderr, format, arglist);
    va_end(arglist);

    // TODO add logging or something

    fprintf(stderr, "\nExiting\n");

#if defined(MPI_HDR)
    MPI_Abort(MPI_COMM_WORLD, 1);
#else
    exit(1);
#endif
}

// print out timing info when done
CloverChunk::~CloverChunk
(void)
{
    if (profiler_on)
    {
        fprintf(stdout, "@@@@@ PROFILING @@@@@\n");

        for (std::map<std::string, double>::iterator ii = kernel_times.begin();
            ii != kernel_times.end(); ii++)
        {
            fprintf(stdout, "%30s : %.3f\n", (*ii).first.c_str(), (*ii).second);
        }
    }
}

std::vector<double> CloverChunk::dumpArray
(const std::string& arr_name, int x_extra, int y_extra, int z_extra)
{
    // number of bytes to allocate for 3d array
    #define BUFSZ3D(x_extra, y_extra)   \
        ( ((x_max) + 4 + x_extra)       \
        * ((y_max) + 4 + y_extra)       \
        * ((z_max) + 4 + z_extra)       \
        * sizeof(double) )

    std::vector<double> host_buffer(BUFSZ3D(x_extra, y_extra)/sizeof(double));

    queue.finish();

    try
    {
        queue.enqueueReadBuffer(arr_names.at(arr_name),
            CL_TRUE, 0, BUFSZ3D(x_extra, y_extra), &host_buffer[0]);
    }
    catch (cl::Error e)
    {
        DIE("Error '%s (%d)' reading array %s back from device",
            e.what(), e.err(), arr_name.c_str());
    }
    catch (std::out_of_range e)
    {
        DIE("Error - %s was not in the arr_names map\n", arr_name.c_str());
    }

    queue.finish();

    return host_buffer;
}


