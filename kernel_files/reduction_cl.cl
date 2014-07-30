#include "./kernel_files/macros_cl.cl"

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

/*
 *  Need to be defined:
 *  reduce_t = array type to reduce (double or int)
 *  REDUCE = operation to do
 *  INIT_RED_VAL = initial value before loading - eg, 0 for a sum
 *  LOCAL_SZ = local work group size, 1D
 */

#if defined(red_sum)
inline reduce_t REDUCE
(reduce_t a, reduce_t b)
{
    return SUM(a,b);
}
#elif defined (red_min)
inline reduce_t REDUCE
(reduce_t a, reduce_t b)
{
    return MIN(a,b);
}
#elif defined (red_max)
inline reduce_t REDUCE
(reduce_t a, reduce_t b)
{
    return MAX(a,b);
}
#else
    #error No definition for reduction
#endif

__kernel void reduction
(__global       reduce_t * const __restrict input)
{
    const int lid = get_local_id(0);
    const int gid = get_global_id(0);

    __local reduce_t scratch[LOCAL_SZ];

    // initialises to some initial value - different for different reductions
    scratch[lid] = INIT_RED_VAL;

    /*
     *  Read and write to two opposite halves of the reduction buffer so that
     *  there are no data races with reduction. eg, first stages reads from
     *  first half of buffer and writes results into second half, then second
     *  stage of reduction reads from second half and writes back into first,
     *  etc.
     */
    size_t dest_offset;
    size_t src_offset;

    if (!(RED_STAGE % 2))
    {
        src_offset = ORIG_ELEMS_TO_REDUCE;
        dest_offset = 0;
    }
    else
    {
        src_offset = 0;
        dest_offset = ORIG_ELEMS_TO_REDUCE;
    }

    /*
     *  If the number of elements to reduce is not a power of 2 then 2 values
     *  can be loaded for an initial reduction for some threads but not for
     *  others, defined by the threshold corresponding to the difference between
     *  the next power of 2 up from the number to reduce
     *
     *  if there are 900 values to reduce with a 256 local size, then launch 2
     *  groups of total thread count 512, and load 2 values to reduce on load in
     *  the first 250 of these threads
     */
    if (0&&gid < RED_LOAD_THRESHOLD)
    // FIXME this isn't working properly for now - just load one per thread
    {
        // TODO when this is fixed then do it in a vector for xeon phi things?
        // load + reduce at the same time
        scratch[lid] = REDUCE(input[src_offset + gid],
            input[src_offset + gid + GLOBAL_SZ]);
    }
    else if (gid < ELEMS_TO_REDUCE)
    {
        // just load
        scratch[lid] = input[src_offset + gid];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

#if defined(CL_DEVICE_TYPE_GPU)

    for (int offset = LOCAL_SZ / 2; offset > 0; offset /= 2)
    {
        if (lid < offset)
        {
            scratch[lid] = REDUCE(scratch[lid], scratch[lid + offset]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

#elif defined(CL_DEVICE_TYPE_CPU)

    if (0 == lid)
    {
        for (int offset = 1; offset < LOCAL_SZ; offset++)
        {
            scratch[0] = REDUCE(scratch[0], scratch[offset]);
        }
    }

#elif defined(CL_DEVICE_TYPE_ACCELERATOR)

    // TODO special reductions for xeon phi in some fashion
    if (0 == lid)
    {
        for (int offset = 1; offset < LOCAL_SZ; offset++)
        {
            scratch[0] = REDUCE(scratch[0], scratch[offset]);
        }
    }

#else

    #error No device type specified for reduction

#endif

    if (0 == lid)
    {
#if (LOCAL_SZ == GLOBAL_SZ)
        // last stage - write back into 0 - no chance of data race
        input[0] = scratch[0];
#else
        input[dest_offset + get_group_id(0)] = scratch[0];
#endif
    }
}

