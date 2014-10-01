#include "ocl_common.hpp"
#include <numeric>

extern "C" void ocl_pack_buffers_
(int fields[NUM_FIELDS], int offsets[NUM_FIELDS], int * depth,
 int * face, double * buffer)
{
    chunk.packUnpackAllBuffers(fields, offsets, *depth, *face, 1, buffer);
}

extern "C" void ocl_unpack_buffers_
(int fields[NUM_FIELDS], int offsets[NUM_FIELDS], int * depth,
 int * face, double * buffer)
{
    chunk.packUnpackAllBuffers(fields, offsets, *depth, *face, 0, buffer);
}

void CloverChunk::packUnpackAllBuffers
(int fields[NUM_FIELDS], int offsets[NUM_FIELDS],
 const int depth, const int face, const int pack,
 double * buffer)
{
    const int n_exchanged = std::accumulate(fields, fields + NUM_FIELDS, 0);

    if (n_exchanged < 1)
    {
        return;
    }

    // which buffer is being used for this operation
    cl::Buffer * side_buffer = NULL;

    switch (face)
    {
    case CHUNK_LEFT:
        side_buffer = &left_buffer;
        break;
    case CHUNK_RIGHT:
        side_buffer = &right_buffer;
        break;
    case CHUNK_BOTTOM:
        side_buffer = &bottom_buffer;
        break;
    case CHUNK_TOP:
        side_buffer = &top_buffer;
        break;
    default:
        DIE("Invalid face identifier %d passed to mpi buffer packing\n", face);
    }

    cl::Kernel * pack_kernel = NULL;

    // set which kernel to call
    if (pack)
    {
        switch (face)
        {
        case CHUNK_LEFT:
            pack_kernel = &pack_left_buffer_device;
            break;
        case CHUNK_RIGHT:
            pack_kernel = &pack_right_buffer_device;
            break;
        case CHUNK_BOTTOM:
            pack_kernel = &pack_bottom_buffer_device;
            break;
        case CHUNK_TOP:
            pack_kernel = &pack_top_buffer_device;
            break;
        default:
            DIE("Invalid face identifier %d passed to pack\n", face);
        }
    }
    else
    {
        switch (face)
        {
        case CHUNK_LEFT:
            pack_kernel = &unpack_left_buffer_device;
            break;
        case CHUNK_RIGHT:
            pack_kernel = &unpack_right_buffer_device;
            break;
        case CHUNK_BOTTOM:
            pack_kernel = &unpack_bottom_buffer_device;
            break;
        case CHUNK_TOP:
            pack_kernel = &unpack_top_buffer_device;
            break;
        default:
            DIE("Invalid face identifier %d passed to unpack\n", face);
        }
    }

    pack_kernel->setArg(3, *side_buffer);
    pack_kernel->setArg(4, depth);

    // size of this buffer
    size_t side_size = 0;
    // reuse the halo update kernels sizes to launch packing kernels
    cl::NDRange pack_global, pack_local;

    switch (face)
    {
    case CHUNK_LEFT:
    case CHUNK_RIGHT:
        side_size = lr_mpi_buf_sz;
        pack_global = update_lr_global_size[depth-1];
        pack_local = update_lr_local_size[depth-1];
        break;
    case CHUNK_BOTTOM:
    case CHUNK_TOP:
        side_size = bt_mpi_buf_sz;
        pack_global = update_ud_global_size[depth-1];
        pack_local = update_ud_local_size[depth-1];
        break;
    default:
        DIE("Invalid face identifier %d passed to mpi buffer packing\n", face);
    }

    if (!pack)
    {
        queue.enqueueWriteBuffer(*side_buffer, CL_TRUE, 0,
            n_exchanged*depth*side_size,
            buffer);
    }

    for (int ii = 0; ii < NUM_FIELDS; ii++)
    {
        int which_field = ii+1;

        if (fields[ii])
        {
            if (offsets[ii] < 0 || offsets[ii] > NUM_FIELDS*side_size)
            {
                DIE("Tried to pack/unpack field %d but invalid offset %d given\n",
                    ii, offsets[ii]);
            }

            int x_inc = 0, y_inc = 0;

            // set x/y/z inc for array
            switch (which_field)
            {
            case FIELD_xvel0:
            case FIELD_yvel0:
            case FIELD_xvel1:
            case FIELD_yvel1:
                x_inc = y_inc = 1;
                break;
            case FIELD_mass_flux_x:
            case FIELD_vol_flux_x:
                x_inc = 1;
                break;
            case FIELD_mass_flux_y:
            case FIELD_vol_flux_y:
                y_inc = 1;
                break;
            case FIELD_density0:
            case FIELD_density1:
            case FIELD_energy0:
            case FIELD_energy1:
            case FIELD_pressure:
            case FIELD_viscosity:
            case FIELD_soundspeed:
                break;
            default:
                DIE("Invalid field number %d in choosing _inc values\n", which_field);
            }

            #define CASE_BUF(which_array)   \
            case FIELD_##which_array:       \
            {                               \
                device_array = &which_array;\
            }

            cl::Buffer * device_array = NULL;

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
                DIE("Invalid face %d passed to left/right pack buffer\n", which_field);
            }

            #undef CASE_BUF

            // set args + launch kernel
            pack_kernel->setArg(0, x_inc);
            pack_kernel->setArg(1, y_inc);
            pack_kernel->setArg(2, *device_array);
            pack_kernel->setArg(5, offsets[ii]);

            enqueueKernel(*pack_kernel, __LINE__, __FILE__,
                          cl::NullRange,
                          pack_global,
                          pack_local);
        }
    }

    if (pack)
    {
        queue.finish();
        queue.enqueueReadBuffer(*side_buffer, CL_TRUE, 0,
            n_exchanged*depth*side_size,
            buffer);
    }
}

