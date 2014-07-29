#include "ocl_common.hpp"

#include <numeric>

extern "C" void ocl_pack_buffers_
(int fields[NUM_FIELDS], int offsets[NUM_FIELDS], int * depth,
 int * face, double * buffer)
{
    if (std::accumulate(fields, fields + (NUM_FIELDS-1), 0) > 0)
    {
        // only call if there's actually something to pack
        chunk.packUnpackAllBuffers(fields, offsets, *depth, *face, 1, buffer);
    }
}

extern "C" void ocl_unpack_buffers_
(int fields[NUM_FIELDS], int offsets[NUM_FIELDS], int * depth,
 int * face, double * buffer)
{
    if (std::accumulate(fields, fields + (NUM_FIELDS-1), 0) > 0)
    {
        // only call if there's actually something to unpack
        chunk.packUnpackAllBuffers(fields, offsets, *depth, *face, 0, buffer);
    }
}

void CloverChunk::packUnpackAllBuffers
(int fields[NUM_FIELDS], int offsets[NUM_FIELDS], int depth,
 int face, int pack, double * buffer)
{
    buffer_func_t pack_func;
    int dest;
    int x_inc, y_inc, z_inc;

    /*
     *  First of all, figure out which opencl function to use based on whether
     *  we're packing or unpacking
     */
    if (pack)
    {
        pack_func = (buffer_func_t)&cl::CommandQueue::enqueueReadBufferRect;
    }
    else
    {
        pack_func = (buffer_func_t)&cl::CommandQueue::enqueueWriteBufferRect;
    }

    // make sure jobs are finished
    queue.finish();

    for (int ii = 0; ii < NUM_FIELDS; ii++)
    {
        int which_field = ii+1;

        if (fields[ii])
        {
            if (offsets[ii] < 0)
            {
                DIE("Tried to pack/unpack field %d but invalid offset %d given\n",
                    ii, offsets[ii]);
            }

            x_inc = y_inc = z_inc = 0;

            // set x/y/z inc for array
            switch (which_field)
            {
            case FIELD_xvel0:
            case FIELD_yvel0:
            case FIELD_zvel0:
            case FIELD_xvel1:
            case FIELD_yvel1:
            case FIELD_zvel1:
                x_inc = y_inc = z_inc = 1; break;
            case FIELD_mass_flux_x:
            case FIELD_vol_flux_x:
                x_inc = 1; break;
            case FIELD_mass_flux_y:
            case FIELD_vol_flux_y:
                y_inc = 1; break;
            case FIELD_mass_flux_z:
            case FIELD_vol_flux_z:
                z_inc = 1; break;
            }

            // set the destination
            if (pack)
            {
                switch (face)
                {
                case CHUNK_LEFT:
                    dest = (x_min+1) + x_inc - 1 + depth; break;
                case CHUNK_RIGHT:
                    dest = (x_max+1) + 1 - depth; break;
                case CHUNK_BOTTOM:
                    dest = (y_min+1) + y_inc - 1 + depth; break;
                case CHUNK_TOP:
                    dest = (y_max+1) + 1 - depth; break;
                case CHUNK_BACK:
                    dest = (z_min+1) + z_inc - 1 + depth; break;
                case CHUNK_FRONT:
                    dest = (z_max+1) + 1 - depth; break;
                default:
                    DIE("Invalid face identified %d passed to pack\n", face);
                }
            }
            else
            {
                switch (face)
                {
                case CHUNK_LEFT:
                    dest = (x_min+1) - depth; break;
                case CHUNK_RIGHT:
                    dest = (x_max+1) + x_inc + depth; break;
                case CHUNK_BOTTOM:
                    dest = (y_min+1) - depth; break;
                case CHUNK_TOP:
                    dest = (y_max+1) + y_inc + depth; break;
                case CHUNK_BACK:
                    dest = (z_min+1) - depth; break;
                case CHUNK_FRONT:
                    dest = (z_max+1) + z_inc + depth; break;
                default:
                    DIE("Invalid face identified %d passed to unpack\n", face);
                }
            }

            packRect(buffer + offsets[ii], pack_func,
                x_inc, y_inc, z_inc,
                face, dest, which_field, depth);
        }
    }

    // make sure mem copies are finished
    queue.finish();
}

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
 int x_inc, int y_inc, int z_inc,
 int face, int dest,
 int which_field, int depth)
{
    cl::Buffer *device_buf;

    cl::size_t<3> b_origin;
    cl::size_t<3> h_origin;
    cl::size_t<3> region;

    #define CASE_BUF(which_array)   \
    case FIELD_##which_array:       \
    {                               \
        device_buf = &which_array;  \
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
    CASE_BUF(yvel0); break;
    CASE_BUF(zvel0); break;
    CASE_BUF(xvel1); break;
    CASE_BUF(yvel1); break;
    CASE_BUF(zvel1); break;
    CASE_BUF(vol_flux_x); break;
    CASE_BUF(vol_flux_y); break;
    CASE_BUF(vol_flux_z); break;
    CASE_BUF(mass_flux_x); break;
    CASE_BUF(mass_flux_y); break;
    CASE_BUF(mass_flux_z); break;
    default:
        device_buf = NULL;
        DIE("Invalid face %d passed to left/right pack buffer\n", which_field);
    }

    size_t b_row_pitch = sizeof(double)*(x_max + 4 + x_inc);
    size_t b_slice_pitch = sizeof(double)*(y_max + 4 + y_inc);
    size_t h_row_pitch = 0;
    size_t h_slice_pitch = 0;

    h_origin[0] = 0;
    h_origin[1] = 0;
    h_origin[2] = 0;

    switch (face)
    {
    // depth*y_max+... region - 1 or 2 columns
    case CHUNK_LEFT:
    case CHUNK_RIGHT:
        b_origin[0] = dest;
        b_origin[1] = (y_min+1) - depth;
        b_origin[2] = (z_min+1) - depth;
        region[0] = depth;
        region[1] = y_max + y_inc + (2*depth);
        region[2] = z_max + z_inc + (2*depth);
        break;

    // depth*x_max+... region - 1 or 2 rows
    case CHUNK_BOTTOM:
    case CHUNK_TOP:
        b_origin[0] = (x_min+1) - depth;
        b_origin[1] = dest;
        b_origin[2] = (z_min+1) - depth;
        region[0] = x_max + x_inc + (2*depth);
        region[1] = depth;
        region[2] = z_max + z_inc + (2*depth);
        break;

    // depth*z_max+... region - 1 or 2 slices
    case CHUNK_BACK:
    case CHUNK_FRONT:
        b_origin[0] = (x_min+1) - depth;
        b_origin[1] = (y_min+1) - depth;
        b_origin[2] = dest;
        region[0] = x_max + x_inc + (2*depth);
        region[1] = y_max + y_inc + (2*depth);
        region[2] = depth;
        break;
    default:
        DIE("Invalid face identifier %d passed to mpi buffer packing\n", face);
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

