#include "ocl_common.hpp"

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

    #define BUF2D_ALLOC(name, x_e, y_e) \
        BUF_ALLOC(name, (x_max+4+x_e) * (y_max+4+y_e) * sizeof(double))

    BUF2D_ALLOC(density0, 0, 0);
    BUF2D_ALLOC(density1, 0, 0);
    BUF2D_ALLOC(energy0, 0, 0);
    BUF2D_ALLOC(energy1, 0, 0);

    BUF2D_ALLOC(pressure, 0, 0);
    BUF2D_ALLOC(soundspeed, 0, 0);
    BUF2D_ALLOC(viscosity, 0, 0);
    BUF2D_ALLOC(volume, 0, 0);

    BUF2D_ALLOC(xvel0, 1, 1);
    BUF2D_ALLOC(xvel1, 1, 1);
    BUF2D_ALLOC(yvel0, 1, 1);
    BUF2D_ALLOC(yvel1, 1, 1);

    BUF2D_ALLOC(xarea, 1, 0);
    BUF2D_ALLOC(vol_flux_x, 1, 0);
    BUF2D_ALLOC(mass_flux_x, 1, 0);

    BUF2D_ALLOC(yarea, 0, 1);
    BUF2D_ALLOC(vol_flux_y, 0, 1);
    BUF2D_ALLOC(mass_flux_y, 0, 1);

    BUF1DX_ALLOC(cellx, 0);
    BUF1DX_ALLOC(celldx, 0);
    BUF1DX_ALLOC(vertexx, 1);
    BUF1DX_ALLOC(vertexdx, 1);

    BUF1DY_ALLOC(celly, 0);
    BUF1DY_ALLOC(celldy, 0);
    BUF1DY_ALLOC(vertexy, 1);
    BUF1DY_ALLOC(vertexdy, 1);

    // work arrays used in various kernels (post_vol, pre_vol, mom_flux, etc)
    BUF2D_ALLOC(work_array_1, 1, 1);
    BUF2D_ALLOC(work_array_2, 1, 1);
    BUF2D_ALLOC(work_array_3, 1, 1);
    BUF2D_ALLOC(work_array_4, 1, 1);
    BUF2D_ALLOC(work_array_5, 1, 1);

    // allocate enough for 1 item per work group, and then a bit extra for the reduction
    // 1.5 should work even if wg size is 2
    BUF_ALLOC(reduce_buf_1, 1.5*((sizeof(double)*reduced_cells)/(LOCAL_X*LOCAL_Y)));
    BUF_ALLOC(reduce_buf_2, 1.5*((sizeof(double)*reduced_cells)/(LOCAL_X*LOCAL_Y)));
    BUF_ALLOC(reduce_buf_3, 1.5*((sizeof(double)*reduced_cells)/(LOCAL_X*LOCAL_Y)));
    BUF_ALLOC(reduce_buf_4, 1.5*((sizeof(double)*reduced_cells)/(LOCAL_X*LOCAL_Y)));
    BUF_ALLOC(reduce_buf_5, 1.5*((sizeof(double)*reduced_cells)/(LOCAL_X*LOCAL_Y)));
    BUF_ALLOC(reduce_buf_6, 1.5*((sizeof(double)*reduced_cells)/(LOCAL_X*LOCAL_Y)));
    BUF_ALLOC(PdV_reduce_buf, 1.5*((sizeof(int)*reduced_cells)/(LOCAL_X*LOCAL_Y)));

    // enough for 1 for each array - overkill, but not that much extra space
    BUF_ALLOC(left_buffer, NUM_BUFFERED_FIELDS*2*sizeof(double)*(y_max + 5));
    BUF_ALLOC(right_buffer, NUM_BUFFERED_FIELDS*2*sizeof(double)*(y_max + 5));
    BUF_ALLOC(bottom_buffer, NUM_BUFFERED_FIELDS*2*sizeof(double)*(x_max + 5));
    BUF_ALLOC(top_buffer, NUM_BUFFERED_FIELDS*2*sizeof(double)*(x_max + 5));

    // needs to be 2 sets, one for each depth
    // fortran expects it to be (sort of) contiguous depending on depth
    for (int depth = 1; depth <= 2; depth++)
    {
        // start off with 0 offset, big enough for 1 exchange buffer
        cl_buffer_region left_right_region = {0, depth*sizeof(double)*(y_max+5)};
        cl_buffer_region bottom_top_region = {0, depth*sizeof(double)*(x_max+5)};

        // for every offset, create the sub buffer then increment the origin
        for (int ii = 0; ii < NUM_BUFFERED_FIELDS; ii++)
        {
            left_subbuffers[depth-1].push_back(
                left_buffer.createSubBuffer(CL_MEM_READ_WRITE,
                    CL_BUFFER_CREATE_TYPE_REGION,
                    &left_right_region));
            right_subbuffers[depth-1].push_back(
                right_buffer.createSubBuffer(CL_MEM_READ_WRITE,
                    CL_BUFFER_CREATE_TYPE_REGION,
                    &left_right_region));
            
            left_right_region.origin += left_right_region.size;

            bottom_subbuffers[depth-1].push_back(
                bottom_buffer.createSubBuffer(CL_MEM_READ_WRITE,
                    CL_BUFFER_CREATE_TYPE_REGION,
                    &bottom_top_region));
            top_subbuffers[depth-1].push_back(
                top_buffer.createSubBuffer(CL_MEM_READ_WRITE,
                    CL_BUFFER_CREATE_TYPE_REGION,
                    &bottom_top_region));
            
            bottom_top_region.origin += bottom_top_region.size;
        }
    }

    #undef BUF2D_ALLOC
    #undef BUF1DX_ALLOC
    #undef BUF1DY_ALLOC
    #undef BUF_ALLOC

    fprintf(DBGOUT, "Buffers allocated\n");
}

