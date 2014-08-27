#include "ocl_common.hpp"

extern "C" void ocl_allocate_mpi_buffers_
(int * lr_align_elems, int * bt_align_elems, int * fb_align_elems)
{
    chunk.allocateMPIBuffers(lr_align_elems, bt_align_elems, fb_align_elems);
}

void CloverChunk::initBuffers
(void)
{
    const std::vector<double> zeros(total_cells, 0.0);

    #define BUF_ALLOC(name, buf_sz)                 \
        try                                         \
        {                                           \
            name = cl::Buffer(context,              \
                              CL_MEM_READ_WRITE,    \
                              (buf_sz));            \
            queue.enqueueWriteBuffer(name,          \
                                     CL_TRUE,       \
                                     0,             \
                                     (buf_sz),      \
                                     &zeros[0]);    \
        }                                           \
        catch (cl::Error e)                         \
        {                                           \
            DIE("Error in creating %s buffer %d\n", \
                    #name, e.err());                \
        }

    #define BUF1DX_ALLOC(name, x_e)     \
        BUF_ALLOC(name, (x_max+4+x_e) * sizeof(double))

    #define BUF1DY_ALLOC(name, y_e)     \
        BUF_ALLOC(name, (y_max+4+y_e) * sizeof(double))

    #define BUF1DZ_ALLOC(name, z_e)     \
        BUF_ALLOC(name, (z_max+4+z_e) * sizeof(double))

    #define BUF2D_ALLOC(name, x_e, y_e) \
        BUF_ALLOC(name, (x_max+4+x_e) * (y_max+4+y_e) * sizeof(double))
    #define BUF3D_ALLOC(name, x_e, y_e,z_e) \
        BUF_ALLOC(name, (x_max+4+x_e) * (y_max+4+y_e) *(z_max+4+z_e) * sizeof(double))

    BUF3D_ALLOC(density0, 0, 0,0);
    BUF3D_ALLOC(density1, 0, 0,0);
    BUF3D_ALLOC(energy0, 0, 0,0);
    BUF3D_ALLOC(energy1, 0, 0,0);

    BUF3D_ALLOC(pressure, 0, 0,0);
    BUF3D_ALLOC(soundspeed, 0, 0,0);
    BUF3D_ALLOC(viscosity, 0, 0,0);
    BUF3D_ALLOC(volume, 0, 0,0);

    BUF3D_ALLOC(xvel0, 1, 1,1);
    BUF3D_ALLOC(xvel1, 1, 1,1);
    BUF3D_ALLOC(yvel0, 1, 1,1);
    BUF3D_ALLOC(yvel1, 1, 1,1);
    BUF3D_ALLOC(zvel0, 1, 1,1);
    BUF3D_ALLOC(zvel1, 1, 1,1);

    BUF3D_ALLOC(xarea, 1, 0,0);
    BUF3D_ALLOC(vol_flux_x, 1, 0,0);
    BUF3D_ALLOC(mass_flux_x, 1, 0,0);

    BUF3D_ALLOC(yarea, 0, 1,0);
    BUF3D_ALLOC(vol_flux_y, 0, 1,0);
    BUF3D_ALLOC(mass_flux_y, 0, 1,0);

    BUF3D_ALLOC(zarea, 0, 0,1);
    BUF3D_ALLOC(vol_flux_z, 0, 0,1);
    BUF3D_ALLOC(mass_flux_z, 0, 0,1);

    BUF1DX_ALLOC(cellx, 0);
    BUF1DX_ALLOC(celldx, 0);
    BUF1DX_ALLOC(vertexx, 1);
    BUF1DX_ALLOC(vertexdx, 1);

    BUF1DY_ALLOC(celly, 0);
    BUF1DY_ALLOC(celldy, 0);
    BUF1DY_ALLOC(vertexy, 1);
    BUF1DY_ALLOC(vertexdy, 1);

    BUF1DZ_ALLOC(cellz, 0);
    BUF1DZ_ALLOC(celldz, 0);
    BUF1DZ_ALLOC(vertexz, 1);
    BUF1DZ_ALLOC(vertexdz, 1);

    // work arrays used in various kernels (post_vol, pre_vol, mom_flux, etc)
    BUF3D_ALLOC(work_array_1, 1, 1,1);
    BUF3D_ALLOC(work_array_2, 1, 1,1);
    BUF3D_ALLOC(work_array_3, 1, 1,1);
    BUF3D_ALLOC(work_array_4, 1, 1,1);
    BUF3D_ALLOC(work_array_5, 1, 1,1);
    BUF3D_ALLOC(work_array_6, 1, 1,1);
    BUF3D_ALLOC(work_array_7, 1, 1,1);

    // allocate enough for 1 item per work group, and then a bit extra for the reduction
    // 1.5 should work even if wg size is 2
    BUF_ALLOC(reduce_buf_1, 1.5*((sizeof(double)*reduced_cells)/(LOCAL_X*LOCAL_Y*LOCAL_Z)));
    BUF_ALLOC(reduce_buf_2, 1.5*((sizeof(double)*reduced_cells)/(LOCAL_X*LOCAL_Y*LOCAL_Z)));
    BUF_ALLOC(reduce_buf_3, 1.5*((sizeof(double)*reduced_cells)/(LOCAL_X*LOCAL_Y*LOCAL_Z)));
    BUF_ALLOC(reduce_buf_4, 1.5*((sizeof(double)*reduced_cells)/(LOCAL_X*LOCAL_Y*LOCAL_Z)));
    BUF_ALLOC(reduce_buf_5, 1.5*((sizeof(double)*reduced_cells)/(LOCAL_X*LOCAL_Y*LOCAL_Z)));
    BUF_ALLOC(reduce_buf_6, 1.5*((sizeof(double)*reduced_cells)/(LOCAL_X*LOCAL_Y*LOCAL_Z)));
    BUF_ALLOC(PdV_reduce_buf, 1.5*((sizeof(int)*reduced_cells)/(LOCAL_X*LOCAL_Y*LOCAL_Z)));

    fprintf(DBGOUT, "Buffers allocated\n");
}

void CloverChunk::allocateMPIBuffers
(int * lr_align_elems, int * bt_align_elems, int * fb_align_elems)
{
    const std::vector<double> zeros(total_cells, 0.0);

    int device_alignment;

    // get the (device-specific) minimum alignment for the subbuffers
    try
    {
        device_alignment = device.getInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>();
    }
    catch (cl::Error e)
    {
        DIE("%d (%s) when trying to get device alignment\n");
    }

    // set initial (ideal) size for buffers and increment untl it hits alignment
    lr_mpi_buf_sz = sizeof(double)*(y_max + 5)*(z_max + 5);
    bt_mpi_buf_sz = sizeof(double)*(x_max + 5)*(z_max + 5);
    fb_mpi_buf_sz = sizeof(double)*(x_max + 5)*(y_max + 5);

    while (lr_mpi_buf_sz % device_alignment)
        lr_mpi_buf_sz++;
    while (bt_mpi_buf_sz % device_alignment)
        bt_mpi_buf_sz++;
    while (fb_mpi_buf_sz % device_alignment)
        fb_mpi_buf_sz++;

    // enough for 1 for each array - overkill, but not that much extra space
    BUF_ALLOC(left_buffer, NUM_BUFFERED_FIELDS*2*lr_mpi_buf_sz);
    BUF_ALLOC(right_buffer, NUM_BUFFERED_FIELDS*2*lr_mpi_buf_sz);
    BUF_ALLOC(bottom_buffer, NUM_BUFFERED_FIELDS*2*bt_mpi_buf_sz);
    BUF_ALLOC(top_buffer, NUM_BUFFERED_FIELDS*2*bt_mpi_buf_sz);
    BUF_ALLOC(back_buffer, NUM_BUFFERED_FIELDS*2*fb_mpi_buf_sz);
    BUF_ALLOC(front_buffer, NUM_BUFFERED_FIELDS*2*fb_mpi_buf_sz);

    // needs to be 2 sets, one for each depth
    // fortran expects it to be (sort of) contiguous depending on depth
    for (int depth = 1; depth <= 2; depth++)
    {
        // start off with 0 offset, big enough for 1 exchange buffer
        cl_buffer_region left_right_region = {0, depth*lr_mpi_buf_sz};
        cl_buffer_region bottom_top_region = {0, depth*bt_mpi_buf_sz};
        cl_buffer_region back_front_region = {0, depth*fb_mpi_buf_sz};

        // for every offset, create the sub buffer then increment the origin
        for (int ii = 0; ii < NUM_BUFFERED_FIELDS; ii++)
        {
            #define SUBBUFF_CREATE(orig_buf, new_region)    \
                orig_buf.createSubBuffer(CL_MEM_READ_WRITE, \
                    CL_BUFFER_CREATE_TYPE_REGION,           \
                    &new_region)

            try
            {
                left_subbuffers[depth-1].push_back(SUBBUFF_CREATE(left_buffer, left_right_region));
                right_subbuffers[depth-1].push_back(SUBBUFF_CREATE(right_buffer, left_right_region));

                bottom_subbuffers[depth-1].push_back(SUBBUFF_CREATE(bottom_buffer, bottom_top_region));
                top_subbuffers[depth-1].push_back(SUBBUFF_CREATE(top_buffer, bottom_top_region));

                back_subbuffers[depth-1].push_back(SUBBUFF_CREATE(back_buffer, back_front_region));
                front_subbuffers[depth-1].push_back(SUBBUFF_CREATE(front_buffer, back_front_region));
            }
            catch (cl::Error e)
            {
                DIE("Error %d (%s) when allocating subbuffers\n", e.err(), e.what());
            }
            
            left_right_region.origin += left_right_region.size;
            bottom_top_region.origin += bottom_top_region.size;
            back_front_region.origin += back_front_region.size;
        }
    }

    *lr_align_elems = lr_mpi_buf_sz/sizeof(double);
    *bt_align_elems = bt_mpi_buf_sz/sizeof(double);
    *fb_align_elems = fb_mpi_buf_sz/sizeof(double);

    #undef BUF3D_ALLOC
    #undef BUF2D_ALLOC
    #undef BUF1DX_ALLOC
    #undef BUF1DY_ALLOC
    #undef BUF1DZ_ALLOC
    #undef BUF_ALLOC
}

