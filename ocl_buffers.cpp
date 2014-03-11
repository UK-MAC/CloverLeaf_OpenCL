#include "ocl_common.hpp"

void CloverChunk::initBuffers
(void)
{
    const std::vector<double> zeros(total_cells, 0.0);

    #define BUF_ALLOC(name, size)                               \
        try                                                     \
        {                                                       \
            name = cl::Buffer(context,                          \
                              CL_MEM_READ_WRITE,                \
                              (size));                          \
            queue.enqueueWriteBuffer(name,\
                                     CL_TRUE,\
                                     0,\
                                     (size),\
                                     &zeros[0]);\
        }                                                       \
        catch (cl::Error e)                                     \
        {                                                       \
            DIE("Error in creating %s buffer %d\n", \
                    #name, e.err());                            \
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

#if defined(NO_KERNEL_REDUCTIONS)
    // reduction arrays
    /*
    BUF2D_ALLOC(reduce_buf_1, 1, 1);
    BUF2D_ALLOC(reduce_buf_2, 1, 1);
    BUF2D_ALLOC(reduce_buf_3, 1, 1);
    BUF2D_ALLOC(reduce_buf_4, 1, 1);
    BUF2D_ALLOC(reduce_buf_5, 1, 1);
    BUF2D_ALLOC(reduce_buf_6, 1, 1);
    BUF_ALLOC(PdV_reduce_buf, sizeof(int)*total_cells);
    */
    BUF_ALLOC(reduce_buf_1, sizeof(double)*total_cells);
    BUF_ALLOC(reduce_buf_2, sizeof(double)*total_cells);
    BUF_ALLOC(reduce_buf_3, sizeof(double)*total_cells);
    BUF_ALLOC(reduce_buf_4, sizeof(double)*total_cells);
    BUF_ALLOC(reduce_buf_5, sizeof(double)*total_cells);
    BUF_ALLOC(reduce_buf_6, sizeof(double)*total_cells);
    BUF_ALLOC(PdV_reduce_buf, sizeof(int)*total_cells)
#else
    // allocate enough for 1 item per work group, and then a bit extra for the reduction
    // 1.5 should work even if wg size is 2
    BUF_ALLOC(reduce_buf_1, 1.5*((sizeof(double)*total_cells)/(LOCAL_X*LOCAL_Y)))
    BUF_ALLOC(reduce_buf_2, 1.5*((sizeof(double)*total_cells)/(LOCAL_X*LOCAL_Y)))
    BUF_ALLOC(reduce_buf_3, 1.5*((sizeof(double)*total_cells)/(LOCAL_X*LOCAL_Y)))
    BUF_ALLOC(reduce_buf_4, 1.5*((sizeof(double)*total_cells)/(LOCAL_X*LOCAL_Y)))
    BUF_ALLOC(reduce_buf_5, 1.5*((sizeof(double)*total_cells)/(LOCAL_X*LOCAL_Y)))
    BUF_ALLOC(reduce_buf_6, 1.5*((sizeof(double)*total_cells)/(LOCAL_X*LOCAL_Y)))
    BUF_ALLOC(PdV_reduce_buf, sizeof(int)*(x_max+4)*(y_max+4))
#endif

    #undef BUF2D_ALLOC
    #undef BUF1DX_ALLOC
    #undef BUF1DY_ALLOC
    #undef BUF_ALLOC

    fprintf(DBGOUT, "Buffers allocated\n");
}

void CloverChunk::initArgs
(void)
{
    #define SETARG_CHECK(knl, idx, buf) \
        try \
        { \
            knl.setArg(idx, buf); \
        } \
        catch (cl::Error e) \
        { \
            DIE("Error in setting argument index %d to %s for kernel %s (%s - %d)", \
                idx, #buf, #knl, \
                e.what(), e.err()); \
        }

    // ideal_gas
    ideal_gas_device.setArg(2, pressure);
    ideal_gas_device.setArg(3, soundspeed);

    // accelerate
    accelerate_device.setArg(1, xarea);
    accelerate_device.setArg(2, yarea);
    accelerate_device.setArg(3, volume);
    accelerate_device.setArg(4, density0);
    accelerate_device.setArg(5, pressure);
    accelerate_device.setArg(6, viscosity);
    accelerate_device.setArg(7, xvel0);
    accelerate_device.setArg(8, yvel0);
    accelerate_device.setArg(9, xvel1);
    accelerate_device.setArg(10, yvel1);

    // flux calc
    flux_calc_device.setArg(1, xarea);
    flux_calc_device.setArg(2, yarea);
    flux_calc_device.setArg(3, xvel0);
    flux_calc_device.setArg(4, yvel0);
    flux_calc_device.setArg(5, xvel1);
    flux_calc_device.setArg(6, yvel1);
    flux_calc_device.setArg(7, vol_flux_x);
    flux_calc_device.setArg(8, vol_flux_y);

    // viscosity
    viscosity_device.setArg(0, celldx);
    viscosity_device.setArg(1, celldy);
    viscosity_device.setArg(2, density0);
    viscosity_device.setArg(3, pressure);
    viscosity_device.setArg(4, viscosity);
    viscosity_device.setArg(5, xvel0);
    viscosity_device.setArg(6, yvel0);

    // revert
    revert_device.setArg(0, density0);
    revert_device.setArg(1, density1);
    revert_device.setArg(2, energy0);
    revert_device.setArg(3, energy1);

    // initialise chunk
    initialise_chunk_first_device.setArg(4, vertexx);
    initialise_chunk_first_device.setArg(5, vertexdx);
    initialise_chunk_first_device.setArg(6, vertexy);
    initialise_chunk_first_device.setArg(7, vertexdy);

    initialise_chunk_second_device.setArg(4, vertexx);
    initialise_chunk_second_device.setArg(5, vertexdx);
    initialise_chunk_second_device.setArg(6, vertexy);
    initialise_chunk_second_device.setArg(7, vertexdy);
    initialise_chunk_second_device.setArg(8, cellx);
    initialise_chunk_second_device.setArg(9, celldx);
    initialise_chunk_second_device.setArg(10, celly);
    initialise_chunk_second_device.setArg(11, celldy);
    initialise_chunk_second_device.setArg(12, volume);
    initialise_chunk_second_device.setArg(13, xarea);
    initialise_chunk_second_device.setArg(14, yarea);

    // advec_mom
    /*
    post_vol = work array 1
    node_flux = work array 2 _AND_ pre_vol = work array 2
    node_mass_post = work array 3
    node_mass_pre = work array 4
    mom_flux = work array 5
    */
    advec_mom_vol_device.setArg(1, work_array_1);
    advec_mom_vol_device.setArg(2, work_array_2);
    advec_mom_vol_device.setArg(3, volume);
    advec_mom_vol_device.setArg(4, vol_flux_x);
    advec_mom_vol_device.setArg(5, vol_flux_y);

    // x kernels
    advec_mom_node_flux_post_x_device.setArg(0, work_array_2);
    advec_mom_node_flux_post_x_device.setArg(1, work_array_3);
    advec_mom_node_flux_post_x_device.setArg(2, mass_flux_x);
    advec_mom_node_flux_post_x_device.setArg(3, work_array_1);
    advec_mom_node_flux_post_x_device.setArg(4, density1);

    advec_mom_node_pre_x_device.setArg(0, work_array_2);
    advec_mom_node_pre_x_device.setArg(1, work_array_3);
    advec_mom_node_pre_x_device.setArg(2, work_array_4);

    advec_mom_flux_x_device.setArg(0, work_array_2);
    advec_mom_flux_x_device.setArg(1, work_array_3);
    advec_mom_flux_x_device.setArg(2, work_array_4);
    advec_mom_flux_x_device.setArg(4, celldx);
    advec_mom_flux_x_device.setArg(5, work_array_5);

    advec_mom_xvel_device.setArg(0, work_array_3);
    advec_mom_xvel_device.setArg(1, work_array_4);
    advec_mom_xvel_device.setArg(2, work_array_5);

    // y kernels
    advec_mom_node_flux_post_y_device.setArg(0, work_array_2);
    advec_mom_node_flux_post_y_device.setArg(1, work_array_3);
    advec_mom_node_flux_post_y_device.setArg(2, mass_flux_y);
    advec_mom_node_flux_post_y_device.setArg(3, work_array_1);
    advec_mom_node_flux_post_y_device.setArg(4, density1);

    advec_mom_node_pre_y_device.setArg(0, work_array_2);
    advec_mom_node_pre_y_device.setArg(1, work_array_3);
    advec_mom_node_pre_y_device.setArg(2, work_array_4);

    advec_mom_flux_y_device.setArg(0, work_array_2);
    advec_mom_flux_y_device.setArg(1, work_array_3);
    advec_mom_flux_y_device.setArg(2, work_array_4);
    advec_mom_flux_y_device.setArg(4, celldy);
    advec_mom_flux_y_device.setArg(5, work_array_5);

    advec_mom_yvel_device.setArg(0, work_array_3);
    advec_mom_yvel_device.setArg(1, work_array_4);
    advec_mom_yvel_device.setArg(2, work_array_5);

    // advec cell
    /*
    post_vol = work array 1 _AND_ ener_flux = work_array_1
    pre_vol = work array 2
    */

    #define SET_SHARED(knl)             \
        knl.setArg(1, volume);          \
        knl.setArg(2, vol_flux_x);      \
        knl.setArg(3, vol_flux_y);      \
        knl.setArg(4, work_array_2);    \
        knl.setArg(5, density1);        \
        knl.setArg(6, energy1);         \
        knl.setArg(7, work_array_1);

    // x kernels
    advec_cell_pre_vol_x_device.setArg(1, work_array_2);
    advec_cell_pre_vol_x_device.setArg(2, work_array_1);
    advec_cell_pre_vol_x_device.setArg(3, volume);
    advec_cell_pre_vol_x_device.setArg(4, vol_flux_x);
    advec_cell_pre_vol_x_device.setArg(5, vol_flux_y);

    SET_SHARED(advec_cell_ener_flux_x_device)
    advec_cell_ener_flux_x_device.setArg(8, vertexdx);
    advec_cell_ener_flux_x_device.setArg(9, mass_flux_x);

    SET_SHARED(advec_cell_x_device)
    advec_cell_x_device.setArg(8, mass_flux_x);

    // y kernels
    advec_cell_pre_vol_y_device.setArg(1, work_array_2);
    advec_cell_pre_vol_y_device.setArg(2, work_array_1);
    advec_cell_pre_vol_y_device.setArg(3, volume);
    advec_cell_pre_vol_y_device.setArg(4, vol_flux_x);
    advec_cell_pre_vol_y_device.setArg(5, vol_flux_y);

    SET_SHARED(advec_cell_ener_flux_y_device)
    advec_cell_ener_flux_y_device.setArg(8, vertexdy);
    advec_cell_ener_flux_y_device.setArg(9, mass_flux_y);

    SET_SHARED(advec_cell_y_device)
    advec_cell_y_device.setArg(8, mass_flux_y);

    #undef SET_SHARED

    // reset field
    reset_field_device.setArg(0, density0);
    reset_field_device.setArg(1, density1);
    reset_field_device.setArg(2, energy0);
    reset_field_device.setArg(3, energy1);
    reset_field_device.setArg(4, xvel0);
    reset_field_device.setArg(5, xvel1);
    reset_field_device.setArg(6, yvel0);
    reset_field_device.setArg(7, yvel1);

    // set field
    set_field_device.setArg(0, density0);
    set_field_device.setArg(1, density1);
    set_field_device.setArg(2, energy0);
    set_field_device.setArg(3, energy1);

    // generate chunk
    generate_chunk_init_device.setArg(0, density0);
    generate_chunk_init_device.setArg(1, energy0);
    generate_chunk_init_device.setArg(2, xvel0);
    generate_chunk_init_device.setArg(3, yvel0);

    generate_chunk_device.setArg(0, vertexx);
    generate_chunk_device.setArg(1, vertexy);
    generate_chunk_device.setArg(2, cellx);
    generate_chunk_device.setArg(3, celly);
    generate_chunk_device.setArg(4, density0);
    generate_chunk_device.setArg(5, energy0);
    generate_chunk_device.setArg(6, xvel0);
    generate_chunk_device.setArg(7, yvel0);

    // PdV
    PdV_predict_device.setArg(1, PdV_reduce_buf);
    PdV_predict_device.setArg(2, xarea);
    PdV_predict_device.setArg(3, yarea);
    PdV_predict_device.setArg(4, volume);
    PdV_predict_device.setArg(5, density0);
    PdV_predict_device.setArg(6, density1);
    PdV_predict_device.setArg(7, energy0);
    PdV_predict_device.setArg(8, energy1);
    PdV_predict_device.setArg(9, pressure);
    PdV_predict_device.setArg(10, viscosity);
    PdV_predict_device.setArg(11, xvel0);
    PdV_predict_device.setArg(12, yvel0);
    PdV_predict_device.setArg(13, xvel1);
    PdV_predict_device.setArg(14, yvel1);

    PdV_not_predict_device.setArg(1, PdV_reduce_buf);
    PdV_not_predict_device.setArg(2, xarea);
    PdV_not_predict_device.setArg(3, yarea);
    PdV_not_predict_device.setArg(4, volume);
    PdV_not_predict_device.setArg(5, density0);
    PdV_not_predict_device.setArg(6, density1);
    PdV_not_predict_device.setArg(7, energy0);
    PdV_not_predict_device.setArg(8, energy1);
    PdV_not_predict_device.setArg(9, pressure);
    PdV_not_predict_device.setArg(10, viscosity);
    PdV_not_predict_device.setArg(11, xvel0);
    PdV_not_predict_device.setArg(12, yvel0);
    PdV_not_predict_device.setArg(13, xvel1);
    PdV_not_predict_device.setArg(14, yvel1);

    // field summary
    field_summary_device.setArg(0, volume);
    field_summary_device.setArg(1, density0);
    field_summary_device.setArg(2, energy0);
    field_summary_device.setArg(3, pressure);
    field_summary_device.setArg(4, xvel0);
    field_summary_device.setArg(5, yvel0);

    field_summary_device.setArg(6, reduce_buf_1);
    field_summary_device.setArg(7, reduce_buf_2);
    field_summary_device.setArg(8, reduce_buf_3);
    field_summary_device.setArg(9, reduce_buf_4);
    field_summary_device.setArg(10, reduce_buf_5);

    // calc dt
    /*
    work_array_1 = jk_ctrl
    work_array_2 = dt_min
    */
    calc_dt_device.setArg(7, xarea);
    calc_dt_device.setArg(8, yarea);
    calc_dt_device.setArg(9, celldx);
    calc_dt_device.setArg(10, celldy);
    calc_dt_device.setArg(11, volume);
    calc_dt_device.setArg(12, density0);
    calc_dt_device.setArg(13, viscosity);
    calc_dt_device.setArg(14, soundspeed);
    calc_dt_device.setArg(15, xvel0);
    calc_dt_device.setArg(16, xvel0);
    calc_dt_device.setArg(17, reduce_buf_1);
    calc_dt_device.setArg(18, reduce_buf_2);

    // no parameters set for update_halo here

    fprintf(DBGOUT, "Kernel arguments set\n");
}

