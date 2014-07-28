#include "ocl_common.hpp"
extern CloverChunk chunk;

// define a generic interface for fortran
#define C_PACK_INTERFACE(operation, dir)                            \
extern "C" void operation##_##dir##_buffers_ocl_                    \
(int *xmin, int *xmax, int *ymin, int *ymax,                        \
 int *chunk_1, int *chunk_2, int *external_face,                    \
 int *x_inc, int *y_inc, int *depth, int *which_field,              \
 double *field_ptr, double *buffer_1, double *buffer_2)             \
{                                                                   \
    chunk.operation##_##dir(*chunk_1, *chunk_2, *external_face,     \
                            *x_inc, *y_inc, *depth,                 \
                            *which_field, buffer_1, buffer_2);  \
}

C_PACK_INTERFACE(pack, left_right)
C_PACK_INTERFACE(unpack, left_right)
C_PACK_INTERFACE(pack, top_bottom)
C_PACK_INTERFACE(unpack, top_bottom)

/*
 *  Takes the host buffer to be packed for sending over mpi and a callback to
 *  the relevant opencl function to either read or write from the device to pack
 *  or unpack it for sending over mpi.
 *
 *  'dest' is the origin of the rectangle to be copied (either row or column) -
 *  needed because the packing and unpacking goes into different places in the
 *  grid.
 */
void CloverChunk::packRect
(double* host_buffer, buffer_func_t buffer_func,
 int x_inc, int y_inc, int edge, int dest,
 int which_field, int depth)
{
    cl::Buffer *device_buf;

    cl::size_t<3> b_origin;
    cl::size_t<3> h_origin;
    cl::size_t<3> region;

    size_t b_row_pitch = sizeof(double)*(x_max + 4 + x_inc);
    size_t b_slice_pitch = 0;
    size_t h_row_pitch = 0;
    size_t h_slice_pitch = 0;

    h_origin[0] = 0;
    h_origin[1] = 0;
    h_origin[2] = 0;

    #define CASE_BUF(which_array)   \
    case FIELD_##which_array:       \
    {                               \
        device_buf = &which_array;  \
        break;                      \
    }

    switch (which_field)
    {
    CASE_BUF(density0); break;
    CASE_BUF(density1); break;
    CASE_BUF(energy0); break;
    CASE_BUF(energy1); break;
    CASE_BUF(pressure); break;
    CASE_BUF(viscosity); break;
    CASE_BUF(soundspeed); break;
    CASE_BUF(xvel0); break;
    CASE_BUF(xvel1); break;
    CASE_BUF(yvel0); break;
    CASE_BUF(yvel1); break;
    CASE_BUF(vol_flux_x); break;
    CASE_BUF(vol_flux_y); break;
    CASE_BUF(mass_flux_x); break;
    CASE_BUF(mass_flux_y); break;
    default:
        device_buf = NULL;
        DIE("Invalid face %d passed to left/right pack buffer\n", which_field);
    }

    switch (edge)
    {
    // depth*y_max+... region - 1 or 2 columns
    case CHUNK_LEFT:
        b_origin[0] = dest;
        b_origin[1] = (y_min+1) - depth;
        b_origin[2] = 0;
        region[0] = depth;
        region[1] = y_max + y_inc + (2*depth);
        region[2] = 1;
        break;
    case CHUNK_RIGHT:
        b_origin[0] = dest;
        b_origin[1] = (y_min+1) - depth;
        b_origin[2] = 0;
        region[0] = depth;
        region[1] = y_max + y_inc + (2*depth);
        region[2] = 1;
        break;

    // depth*x_max+... region - 1 or 2 rows
    case CHUNK_BOTTOM:
        b_origin[0] = (x_min+1) - depth;
        b_origin[1] = dest;
        b_origin[2] = 0;
        region[0] = x_max + x_inc + (2*depth);
        region[1] = depth;
        region[2] = 1;
        break;
    case CHUNK_TOP:
        b_origin[0] = (x_min+1) - depth;
        b_origin[1] = dest;
        b_origin[2] = 0;
        region[0] = x_max + x_inc + (2*depth);
        region[1] = depth;
        region[2] = 1;
        break;
    default:
        DIE("Invalid face identifier %d passed to mpi buffer packing\n", edge);
    }

    // [0] is in bytes
    b_origin[0] *= sizeof(double);
    region[0] *= sizeof(double);

    // not actually using events for this
    static cl::Event dummy;

    try
    {
        (queue.*buffer_func)(*device_buf,
                             CL_FALSE,
                             b_origin,
                             h_origin,
                             region,
                             b_row_pitch,
                             b_slice_pitch,
                             h_row_pitch,
                             h_slice_pitch,
                             host_buffer,
                             NULL,
                             dummy);
    }
    catch (cl::Error e)
    {
        DIE("Error in copying rect (%s), error %d\n", e.what(), e.err());
    }
}

/*
 *  Call the buffer packing/unpacking function with the relevant arguments for
 *  the operation being done and the side its being done on
 */
#define CHECK_PACK(op, side1, side2, dest1, dest2)                          \
    if (external_face != chunk_1)                                           \
    {                                                                       \
        packRect(buffer_1,                                                  \
                 (buffer_func_t)&cl::CommandQueue::enqueue##op##BufferRect, \
                 x_inc, y_inc, side1, dest1,                                \
                 which_field, depth);                                       \
    }                                                                       \
    if (external_face != chunk_2)                                           \
    {                                                                       \
        packRect(buffer_2,                                                  \
                 (buffer_func_t)&cl::CommandQueue::enqueue##op##BufferRect, \
                 x_inc, y_inc, side2, dest2,                                \
                 which_field, depth);                                       \
    }                                                                       \
    if (external_face != chunk_1 || external_face != chunk_2)               \
    {                                                                       \
        queue.finish();                                                     \
    }

void CloverChunk::pack_left_right
(PACK_ARGS)
{
    CHECK_PACK(Read,
               CHUNK_LEFT, CHUNK_RIGHT,
               (x_min+1) + x_inc - 1 + 1,
               (x_max+1) + 1 - depth)
}

void CloverChunk::unpack_left_right
(PACK_ARGS)
{
    CHECK_PACK(Write,
               CHUNK_LEFT, CHUNK_RIGHT,
               (x_min+1) - depth,
               (x_max+1) + x_inc + 1)
}

void CloverChunk::pack_top_bottom
(PACK_ARGS)
{
    CHECK_PACK(Read,
               CHUNK_TOP, CHUNK_BOTTOM,
               (y_min+1) + y_inc - 1 + depth,
               (y_max+1) + 1 - depth)
}

void CloverChunk::unpack_top_bottom
(PACK_ARGS)
{
    CHECK_PACK(Write,
               CHUNK_TOP, CHUNK_BOTTOM,
               (y_min+1) - depth,
               (y_max+1) + y_inc + 1)
}


