#include "ocl_common.hpp"

#include <cstdio>
#include <sstream>
#include <cmath>

void CloverChunk::initReduction
(void)
{
    /*
     *  create a reduction kernel, one for each layer, with the right parameters
     */
    fprintf(DBGOUT, "\n---- Reduction ----\n");

    // each work group reduces to 1 value inside each kernel
    const size_t total_to_reduce = ceil(float(reduced_cells)/(LOCAL_X*LOCAL_Y));

    fprintf(DBGOUT, "Total cells to reduce = %zu\n", reduced_cells);
    size_t reduction_global_size = total_to_reduce;
    fprintf(DBGOUT, "Reduction within work group reduces to = %zu\n", reduction_global_size);

    // each thread can load 2 values to reduce at once
    //reduction_global_size /= 2;
    //fprintf(DBGOUT, "Loading two values per thread reduces to = %d\n", reduction_global_size);

    int ii = 0;

    while (++ii)
    {
        /*
         *  TODO
         *
         *  one set of reduction kernels for big rows, one for small rows
         */

        // different kernels for different types and operations
        cl::Kernel sum_double, min_double, max_double;
        cl::Kernel max_int;

        // make options again
        std::stringstream options("");

#ifdef __arm__
        options << "-D CLOVER_NO_BUILTINS ";
#endif

        // which stage this reduction kernel is at - starts at 1
        options << "-D RED_STAGE=" << ii << " ";
        // original total number of elements to reduce
        options << "-D ORIG_ELEMS_TO_REDUCE=" << total_to_reduce << " ";

        // device type in the form "-D..."
        options << device_type_prepro;
        options << "-w ";

        // the actual number of elements that needs to be reduced in this stage
        const size_t stage_elems_to_reduce = reduction_global_size;
        options << "-D ELEMS_TO_REDUCE=" << stage_elems_to_reduce << " ";

        fprintf(DBGOUT, "\n\nStage %d:\n", ii);
        fprintf(DBGOUT, "%zu elements remaining to reduce\n", stage_elems_to_reduce);

        /*
         *  To get the local size to use at this stage, figure out the largest
         *  power of 2 that is under the global size
         *
         *  NB at the moment, enforcing power of 2 local size anyway
         *  NB also, 128 was preferred work group size on phi
         */
        size_t reduction_local_size = LOCAL_X*LOCAL_Y;

        // if there are more elements to reduce than the standard local size
        if (reduction_global_size > reduction_local_size)
        {
            /*
             *  If the standard reduction size is smaller than the number of
             *  actual elements remaining, then do a reduction as normal,
             *  writing back multiple values into the reduction buffer
             */

            /*
             *  Calculate the total number of threads to launch at this stage by
             *  making it divisible by the local size so that a binary reduction can
             *  be done, then dividing it by 2 to account for each thread possibly
             *  being able to load 2 values at once
             *
             *  Keep track of original value for use in load threshold calculation
             */
            while (reduction_global_size % reduction_local_size)
            {
                reduction_global_size++;
            }
            //reduction_global_size /= 2;
        }
        else
        {
            /*
             *  If we are down to a number of elements that is less than we can
             *  fit into one workgroup, then just launch one workgroup which
             *  finishes the reduction
             */
            while (reduction_local_size >= reduction_global_size*2)
            {
                reduction_local_size /= 2;
            }

            /*
             *  launch one work group to finish
             */
            reduction_global_size = reduction_local_size;
        }

        fprintf(DBGOUT, "Padded total number of threads to launch is %zu\n", reduction_global_size);
        fprintf(DBGOUT, "Local size for reduction is %zu\n", reduction_local_size);

        options << "-D GLOBAL_SZ=" << reduction_global_size << " ";
        options << "-D LOCAL_SZ=" << reduction_local_size << " ";

        // FIXME not working properly - just load one per thread for now
        #if 0
        // threshold for a thread loading 2 values
        size_t red_load_threshold = stage_elems_to_reduce/2;
        options << "-DRED_LOAD_THRESHOLD=" << red_load_threshold << " ";
        fprintf(DBGOUT, "Load threshold is %zu\n", red_load_threshold);
        #endif
        options << "-D RED_LOAD_THRESHOLD=" << 0 << " ";
        options << "-I. ";

        fprintf(DBGOUT, "\n");

        // name of reduction kernel, data type, what the reduction does
        #define MAKE_REDUCE_KNL(name, data_type, init_val)          \
        {                                                           \
            std::string red_options = options.str()                 \
                + "-D red_"+#name+" "                               \
                + "-D reduce_t="#data_type+" "                      \
                + "-D INIT_RED_VAL="+#init_val+" ";                 \
            fprintf(DBGOUT, "Making reduction kernel '%s_%s' ",     \
                    #name, #data_type);                             \
            fprintf(DBGOUT, "with options string:\n%s\n",           \
                    red_options.c_str());                           \
            try                                                     \
            {                                                       \
                compileKernel(red_options,                          \
                    "./kernel_files/reduction_cl.cl",               \
                    "reduction",                                    \
                    name##_##data_type);                            \
            }                                                       \
            catch (KernelCompileError err)                          \
            {                                                       \
                DIE("Errors in compiling reduction %s_%s:\n%s\n",   \
                    #name, #data_type, err.what());                 \
            }                                                       \
            fprintf(DBGOUT, "Kernel '%s_%s' successfully built\n",  \
                    #name, #data_type);                             \
            reduce_kernel_info_t info;                              \
            info.kernel = name##_##data_type;                       \
            info.global_size = cl::NDRange(reduction_global_size);  \
            info.local_size = cl::NDRange(reduction_local_size);    \
            name##_red_kernels_##data_type.push_back(info);         \
            fprintf(DBGOUT, "\n");                                  \
        }

        MAKE_REDUCE_KNL(sum, double, 0.0);
        MAKE_REDUCE_KNL(max, double, 0.0);
        MAKE_REDUCE_KNL(min, double, DBL_MAX);
        MAKE_REDUCE_KNL(max, int, 0);

        fprintf(DBGOUT, "%zu/", reduction_global_size);
        reduction_global_size /= reduction_local_size;
        fprintf(DBGOUT, "%zu = %zu remaining\n", reduction_local_size, reduction_global_size);

        if (reduction_global_size <= 1)
        {
            break;
        }
    }

    fprintf(DBGOUT, "---- Reduction ----\n\n");
}

