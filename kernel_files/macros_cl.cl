#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define __kernel_indexes                            \
    const size_t column = get_global_id(0);			\
    const size_t row = get_global_id(1);				\
    const size_t loc_column = get_local_id(0);			\
    const size_t loc_row = get_local_id(1);			\
    const size_t lid = loc_row*LOCAL_X + loc_column;	\
    const size_t gid = row*get_global_size(0) + column;

#define THARR2D(x_offset, y_offset, big_row)        \
    (                                               \
      column                      /* horizontal  */ \
    + row*(x_max + 4)             /* vertical    */ \
    + (x_offset)                  /* horz offset */ \
    + (y_offset)*(x_max + 4)      /* vert offset */ \
    + (big_row)*(row + (y_offset))/* big row   */   \
    )

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
            out[get_group_id(1)*get_num_groups(0) + get_group_id(0)] = in[0]; \
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
            out[get_group_id(1)*get_num_groups(0) + get_group_id(0)] = in[0]; \
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
            out[get_group_id(1)*get_num_groups(0) + get_group_id(0)] = in[0]; \
        }

#if 0

    /*
     *  TODO
     *  
     *  8/16 wide vector units
     *  4 cores per thing
     *  57-61 cpus
     */

    #if 0
    #define REDUCTION(in, out, operation)                    \
        barrier(CLK_LOCAL_MEM_FENCE);                               \
        for (size_t offset = BLOCK_SZ / 2; offset > 0; offset /= 2) \
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
            out[get_group_id(1)*get_num_groups(0) + get_group_id(0)] = in[0]; \
        }
    #else
    #define REDUCTION(in, out, operation)                    \
    { \
        barrier(CLK_LOCAL_MEM_FENCE);                               \
        const size_t vecsz = 512/(sizeof(in[0])*8);  \
        const size_t redsz = vecsz*2; \
        size_t remain = BLOCK_SZ ; \
        do \
        { \
            if (!(lid % redsz) && lid < remain)                \
            {                                                           \
                /*for (size_t offset = 1; offset < redsz; offset++)    \
                {                                                       \
                    in[lid] = operation(in[lid], in[lid + offset]);               \
                }*/                                                       \
                /*in[0 + lid] = operation(in[0 + lid], in[0 + lid+vecsz]); \
                in[1 + lid] = operation(in[1 + lid], in[1 + lid+vecsz]); \
                in[2 + lid] = operation(in[2 + lid], in[2 + lid+vecsz]); \
                in[3 + lid] = operation(in[3 + lid], in[3 + lid+vecsz]); \
                in[4 + lid] = operation(in[4 + lid], in[4 + lid+vecsz]); \
                in[5 + lid] = operation(in[5 + lid], in[5 + lid+vecsz]); \
                in[6 + lid] = operation(in[6 + lid], in[6 + lid+vecsz]); \
                in[7 + lid] = operation(in[7 + lid], in[7 + lid+vecsz]);*/ \
                for (size_t offset = 0; offset < vecsz; offset++)    \
                {                                                       \
                    in[offset + lid] = operation(in[offset + lid], in[offset + lid+vecsz]); \
                }                                                       \
                barrier(CLK_LOCAL_MEM_FENCE);                               \
                in[lid/redsz] = in[lid]; \
            } \
            else \
            { \
                barrier(CLK_LOCAL_MEM_FENCE);                               \
                break;\
            } \
        } while ((remain = remain/redsz) >= redsz); \
        out[get_group_id(1)*get_num_groups(0) + get_group_id(0)] = in[0]; \
        /*if (0 == lid)                                               \
        {                                                           \
            for (size_t offset = 1; offset < BLOCK_SZ/vecsz; offset++)    \
            {                                                       \
                in[0] = operation(in[0], in[offset]);               \
            }                                                       \
            out[get_group_id(1)*get_num_groups(0) + get_group_id(0)] = in[0]; \
        }*/ \
        /*for (size_t offset = BLOCK_SZ / (2*vecsz); offset > 0; offset /= 2) \
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
            out[get_group_id(1)*get_num_groups(0) + get_group_id(0)] = in[0]; \
        }*/ \
    }
    #endif
#endif

#else

    #error No device type specified - don't know which reduction to use

#endif

