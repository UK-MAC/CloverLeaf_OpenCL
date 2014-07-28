#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define __kernel_indexes                            \
    const size_t column = get_global_id(0);            \
    const size_t row = get_global_id(1);    \
    const size_t slice = get_global_id(2);            \
    const size_t loc_column = get_local_id(0);            \
    const size_t loc_row = get_local_id(1);            \
    const size_t loc_slice = get_local_id(2);            \
    const size_t lid = loc_slice*LOCAL_X*LOCAL_Y + loc_row*LOCAL_X + loc_column;    \
    const size_t gid = slice*get_global_size(1)*get_global_size(0) + row*get_global_size(0) + column;

#define THARR2D(x_offset, y_offset, big_row)        \
    (                                               \
      column                      /* horizontal  */ \
    + row*(x_max + 4)             /* vertical    */ \
    + (x_offset)                  /* horz offset */ \
    + (y_offset)*(x_max + 4)      /* vert offset */ \
    + (big_row)*(row + (y_offset))/* big row   */   \
    )

#define THARR3D(x_offset, y_offset, z_offset, big_row, big_col)   \
    ((slice+z_offset)*(x_max+4+big_row)*(y_max+4+big_col)       \
    + THARR2D(x_offset, y_offset, big_row))

#ifdef CLOVER_NO_BUILTINS
    #define MAX(a,b) (a<b?a:b)
    #define MIN(a,b) (a>b?a:b)
    #define SUM(a,b) (a+b)
    #define SIGN(a,b) (((b) <  (0) && (a > (0))||((b) > (0) && ((a)<(0)))) ? (-a) : (a))
    #define SQRT(a) sqrt(convert_float(a))
#else
    #define MAX(a,b) max(a,b)
    #define MIN(a,b) min(a,b)
    #define SUM(a,b) ((a)+(b))
    #define SIGN(a,b) copysign(a,b)
    #define SQRT(a) sqrt(a)
#endif

// TODO probably can optimise reductions somehow
#if defined(CL_DEVICE_TYPE_GPU)

    // binary tree reduction
    #define REDUCTION(in, out, operation)                           \
        barrier(CLK_LOCAL_MEM_FENCE);                               \
        for (int offset = BLOCK_SZ / 2; offset > 0; offset /= 2)    \
        {                                                           \
            if (lid < offset)                                       \
            {                                                       \
                in[lid] = operation(in[lid],                        \
                                    in[lid + offset]);              \
            }                                                       \
            barrier(CLK_LOCAL_MEM_FENCE);                           \
        }                                                           \
        if(!lid)                                                    \
        {                                                           \
            out[get_group_id(2)*get_num_groups(1)*get_num_groups(0) + \
                get_group_id(1)*get_num_groups(0) + \
                get_group_id(0)] = in[0]; \
        }

#elif defined(CL_DEVICE_TYPE_CPU)

    // loop in first thread
    #define REDUCTION(in, out, operation)                       \
        barrier(CLK_LOCAL_MEM_FENCE);                           \
        if (0 == lid)                                           \
        {                                                       \
            for (int offset = 1; offset < BLOCK_SZ; offset++)   \
            {                                                   \
                in[0] = operation(in[0], in[offset]);           \
            }                                                   \
            out[get_group_id(2)*get_num_groups(1)*get_num_groups(0) + \
                get_group_id(1)*get_num_groups(0) + \
                get_group_id(0)] = in[0]; \
        }

#elif defined(CL_DEVICE_TYPE_ACCELERATOR)

    // loop in first thread
    #define REDUCTION(in, out, operation)                       \
        barrier(CLK_LOCAL_MEM_FENCE);                           \
        if (0 == lid)                                           \
        {                                                       \
            for (int offset = 1; offset < BLOCK_SZ; offset++)   \
            {                                                   \
                in[0] = operation(in[0], in[offset]);           \
            }                                                   \
            out[get_group_id(2)*get_num_groups(1)*get_num_groups(0) + \
                get_group_id(1)*get_num_groups(0) + \
                get_group_id(0)] = in[0]; \
        }

#else

    #error No device type specified - don't know which reduction to use

#endif


