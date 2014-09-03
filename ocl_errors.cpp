#if defined(MPI_HDR)
#include "mpi.h"
#endif
#include "ocl_common.hpp"

#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <cstdarg>
#include <numeric>

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
            static cl::Event no_event_passed = cl::Event();

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
            fprintf(stdout, "Global size: [%zu %zu]\n", global_range[0], global_range[1]);
            fprintf(stdout, "Local size:  [%zu %zu]\n", local_range[0], local_range[1]);
            fprintf(stdout, "Offset size: [%zu %zu]\n", offset_range[0], offset_range[1]);
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
                kernel_calls.at(func_name) += 1;
                kernel_times.at(func_name) += taken;
            }
            else
            {
                kernel_calls[func_name] = 1;
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
                // only print this if there is actually an offset
                if (offset_range.dimensions()) errstr << "(offset " << offset_range[ii] << ") - ";
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

        if (kernel_times.size() > 0)
        {
            // how many arrays each kernel accesses
            std::map<std::string, double> kernel_params;

            /*
             *  could do this as some kind of static thing but some paramters
             *  needs changing depending on the run time parameters. This is
             *  horrendously ugly but it is only done when profiling is on (slow
             *  anyway)
             *
             *  Overall some of these kernels are slightly udnerstimated due to
             *  things like field_summary writing back the reduction values, but
             *  these take up a minor part of the run time
             *
             *  This also assumes that they only work on the inner cells, which
             *  is not always the case but would require yet another gignatic
             *  tabel specifying how many cells each kernel actually accessed.
             *  Underestimating it and assuming perfect caching is about as
             *  sensible as possible
             */
            kernel_params["PdV_not_predict"] = 16;
            kernel_params["PdV_predict"] = 13;
            kernel_params["accelerate"] = 13;
            kernel_params["advec_cell_ener_flux_x"] = 6;
            kernel_params["advec_cell_ener_flux_y"] = 6;
            kernel_params["advec_cell_ener_flux_z"] = 6;
            kernel_params["advec_cell_pre_vol_x"] = 4.5;
            kernel_params["advec_cell_pre_vol_y"] = 4.5;
            kernel_params["advec_cell_pre_vol_z"] = 4.5;
            kernel_params["advec_cell_x"] = 6;
            kernel_params["advec_cell_y"] = 6;
            kernel_params["advec_cell_z"] = 6;
            kernel_params["advec_mom_flux_x"] = 4;
            kernel_params["advec_mom_flux_y"] = 4;
            kernel_params["advec_mom_flux_z"] = 4;
            kernel_params["advec_mom_node_flux_post_x_1"] = 2;
            kernel_params["advec_mom_node_flux_post_x_2"] = 3;
            kernel_params["advec_mom_node_flux_post_y_1"] = 2;
            kernel_params["advec_mom_node_flux_post_y_2"] = 3;
            kernel_params["advec_mom_node_flux_post_z_1"] = 2;
            kernel_params["advec_mom_node_flux_post_z_2"] = 3;
            kernel_params["advec_mom_node_pre_x"] = 3;
            kernel_params["advec_mom_node_pre_y"] = 3;
            kernel_params["advec_mom_node_pre_z"] = 3;
            kernel_params["advec_mom_vol"] = 6;
            kernel_params["advec_mom_xvel"] = 4;
            kernel_params["advec_mom_yvel"] = 4;
            kernel_params["advec_mom_zvel"] = 4;
            kernel_params["calc_dt"] = 10;
            kernel_params["field_summary"] = 7;
            kernel_params["flux_calc_x"] = 4;
            kernel_params["flux_calc_y"] = 4;
            kernel_params["flux_calc_z"] = 4;
            kernel_params["generate_chunk"] = 5;
            kernel_params["generate_chunk_init"] = 5;
            kernel_params["ideal_gas"] = 4;
            kernel_params["initialise_chunk_first"] = 1;
            kernel_params["initialise_chunk_second"] = 4;
            kernel_params["reset_field"] = 10;
            kernel_params["revert"] = 4;
            kernel_params["viscosity"] = 6;
            // 6 <= (avg depth) * 2 for writing and reading 2 slices of array
            //   <= ((1+2)/2) * 2
            kernel_params["update_halo_left"] = 3.0/(x_max);
            kernel_params["update_halo_right"] = 3.0/(x_max);
            kernel_params["update_halo_bottom"] = 3.0/(y_max);
            kernel_params["update_halo_top"] = 3.0/(y_max);
            // slighty underestimated, but roughly correct
            kernel_params["reduction"] = 1.0/(LOCAL_X*LOCAL_Y);

            double total_transferred = 0.0;
            double total_kernel_time = 0.0;

            for (std::map<std::string, double>::iterator ii = kernel_times.begin();
                ii != kernel_times.end(); ii++)
            {
                total_kernel_time += ii->second;
            }

            // FIXME fix this
            fprintf(stdout, "THESE TIMINGS HAVE NOT BEEN FIXED AND ARE NOT ACCURATE\n");

            fprintf(stdout, "%30s   %7s %5s %9s\n", "Kernel name", "runtime", "calls", "bandwidth");

            std::map<std::string, double>::iterator ii = kernel_times.begin();
            std::map<std::string, int>::iterator jj = kernel_calls.begin();

            for (; ii != kernel_times.end(); ii++, jj++)
            {
                double kernel_transferred = (x_max*y_max*sizeof(double)
                    *jj->second*kernel_params.at(jj->first))*1e-9;
                double kernel_bw = kernel_transferred/(ii->second/1000.0);

                fprintf(stdout, "%30s %9.3f %5d %7.5f\n",
                    ii->first.c_str(), ii->second, jj->second, kernel_bw);

                total_transferred += kernel_transferred;
            }

            fprintf(stdout, "Total kernel time %f ms\n", total_kernel_time);
            fprintf(stdout, "Total kernel memory transferred %f GB\n", total_transferred);
            fprintf(stdout, "Average kernel bandwidth %f GB/s\n", total_transferred/(total_kernel_time/1000.0));
        }
    }
}

std::vector<double> CloverChunk::dumpArray
(const std::string& arr_name, int x_extra, int y_extra)
{
    // number of bytes to allocate for 2d array
    #define BUFSZ2D(x_extra, y_extra)   \
        ( ((x_max) + 4 + x_extra)       \
        * ((y_max) + 4 + y_extra)       \
        * sizeof(double) )

    std::vector<double> host_buffer(BUFSZ2D(x_extra, y_extra)/sizeof(double));

    queue.finish();

    try
    {
        queue.enqueueReadBuffer(arr_names.at(arr_name),
            CL_TRUE, 0, BUFSZ2D(x_extra, y_extra), &host_buffer[0]);
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

