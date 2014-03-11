#include "ocl_common.hpp"
#include "ocl_kernel_hdr.hpp"
#include <sstream>

void CloverChunk::initProgram
(void)
{
    // options
    std::stringstream options("");

    // FIXME check to make sure its nvidia
    if (desired_type == CL_DEVICE_TYPE_GPU)
    {
        // for nvidia architecture
        //options << "-cl-nv-arch " << NV_ARCH << " ";
        //options << "-cl-nv-maxrregcount=20 ";
    }

#ifdef __arm__
    // on ARM, don't use built in functions as they don't exist
    options << "-DCLOVER_NO_BUILTINS ";
#endif

#if defined(NO_KERNEL_REDUCTIONS)
    // don't do any reductions inside the kernels
    options << "-D NO_KERNEL_REDUCTIONS ";
#endif

#ifdef ONED_KERNEL_LAUNCHES
    // launch kernels with 1d work group size
    options << "-DONED_KERNEL_LAUNCHES ";
#endif

    // pass in these values so you don't have to pass them in to every kernel
    options << "-Dx_min=" << x_min << " ";
    options << "-Dx_max=" << x_max << " ";
    options << "-Dy_min=" << y_min << " ";
    options << "-Dy_max=" << y_max << " ";

    // local sizes
    options << "-DBLOCK_SZ=" << LOCAL_X*LOCAL_Y << " ";
    options << "-DLOCAL_X=" << LOCAL_X << " ";
    options << "-DLOCAL_Y=" << LOCAL_Y << " ";

    // for update halo
    options << "-DCELL_DATA=" << CELL_DATA << " ";
    options << "-DVERTEX_DATA=" << VERTEX_DATA << " ";
    options << "-DX_FACE_DATA=" << X_FACE_DATA << " ";
    options << "-DY_FACE_DATA=" << Y_FACE_DATA << " ";

    // include current directory
    options << "-I. ";

    // device type in the form "-D..."
    options << device_type_prepro;

    const std::string options_str = options.str();

    fprintf(DBGOUT, "Compiling kernels with options:\n%s\n", options_str.c_str());
    fprintf(stdout, "Compiling kernels (may take some time)...");
    fflush(stdout);

    compileKernel(options_str, src_ideal_gas_cl, "ideal_gas", ideal_gas_device);
    compileKernel(options_str, src_accelerate_cl, "accelerate", accelerate_device);
    compileKernel(options_str, src_flux_calc_cl, "flux_calc", flux_calc_device);
    compileKernel(options_str, src_viscosity_cl, "viscosity", viscosity_device);
    compileKernel(options_str, src_revert_cl, "revert", revert_device);

    compileKernel(options_str, src_initialise_chunk_cl, "initialise_chunk_first", initialise_chunk_first_device);
    compileKernel(options_str, src_initialise_chunk_cl, "initialise_chunk_second", initialise_chunk_second_device);
    compileKernel(options_str, src_generate_chunk_cl, "generate_chunk_init", generate_chunk_init_device);
    compileKernel(options_str, src_generate_chunk_cl, "generate_chunk", generate_chunk_device);

    compileKernel(options_str, src_reset_field_cl, "reset_field", reset_field_device);
    compileKernel(options_str, src_set_field_cl, "set_field", set_field_device);

    compileKernel(options_str, src_PdV_cl, "PdV_predict", PdV_predict_device);
    compileKernel(options_str, src_PdV_cl, "PdV_not_predict", PdV_not_predict_device);

    compileKernel(options_str, src_field_summary_cl, "field_summary", field_summary_device);
    compileKernel(options_str, src_calc_dt_cl, "calc_dt", calc_dt_device);

    compileKernel(options_str, src_update_halo_cl, "update_halo_top", update_halo_top_device);
    compileKernel(options_str, src_update_halo_cl, "update_halo_bottom", update_halo_bottom_device);
    compileKernel(options_str, src_update_halo_cl, "update_halo_left", update_halo_left_device);
    compileKernel(options_str, src_update_halo_cl, "update_halo_right", update_halo_right_device);

    compileKernel(options_str, src_advec_mom_cl, "advec_mom_vol", advec_mom_vol_device);
    compileKernel(options_str, src_advec_mom_cl, "advec_mom_node_flux_post_x", advec_mom_node_flux_post_x_device);
    compileKernel(options_str, src_advec_mom_cl, "advec_mom_node_pre_x", advec_mom_node_pre_x_device);
    compileKernel(options_str, src_advec_mom_cl, "advec_mom_flux_x", advec_mom_flux_x_device);
    compileKernel(options_str, src_advec_mom_cl, "advec_mom_xvel", advec_mom_xvel_device);
    compileKernel(options_str, src_advec_mom_cl, "advec_mom_node_flux_post_y", advec_mom_node_flux_post_y_device);
    compileKernel(options_str, src_advec_mom_cl, "advec_mom_node_pre_y", advec_mom_node_pre_y_device);
    compileKernel(options_str, src_advec_mom_cl, "advec_mom_flux_y", advec_mom_flux_y_device);
    compileKernel(options_str, src_advec_mom_cl, "advec_mom_yvel", advec_mom_yvel_device);

    compileKernel(options_str, src_advec_cell_cl, "advec_cell_pre_vol_x", advec_cell_pre_vol_x_device);
    compileKernel(options_str, src_advec_cell_cl, "advec_cell_ener_flux_x", advec_cell_ener_flux_x_device);
    compileKernel(options_str, src_advec_cell_cl, "advec_cell_x", advec_cell_x_device);
    compileKernel(options_str, src_advec_cell_cl, "advec_cell_pre_vol_y", advec_cell_pre_vol_y_device);
    compileKernel(options_str, src_advec_cell_cl, "advec_cell_ener_flux_y", advec_cell_ener_flux_y_device);
    compileKernel(options_str, src_advec_cell_cl, "advec_cell_y", advec_cell_y_device);

    fprintf(stdout, "done.\n");
    fprintf(DBGOUT, "All kernels compiled\n");
}

void CloverChunk::compileKernel
(const std::string& options,
 const std::string& source_name,
 const char* kernel_name,
 cl::Kernel& kernel)
{
    const std::string source_str(source_name);
    fprintf(DBGOUT, "Compiling %s...", kernel_name);
    cl::Program program;

    if (built_programs.find(source_name) == built_programs.end())
    {
        try
        {
            program = compileProgram(source_str, options);
        }
        catch (std::string errs)
        {
            DIE("Errors in compiling %s:\n%s\n", kernel_name, errs.c_str());
        }

        built_programs[source_name] = program;
    }
    else
    {
        program = built_programs.at(source_name);
    }

    size_t max_wg_size;

    try
    {
        kernel = cl::Kernel(program, kernel_name);
    }
    catch (cl::Error e)
    {
        fprintf(DBGOUT, "Failed\n");
        DIE("Error in creating %s kernel %d\n",
                kernel_name, e.err());
    }
    cl::detail::errHandler(
        clGetKernelWorkGroupInfo(kernel(),
                                 device(),
                                 CL_KERNEL_WORK_GROUP_SIZE,
                                 sizeof(size_t),
                                 &max_wg_size,
                                 NULL));
    if ((LOCAL_X*LOCAL_Y) > max_wg_size)
    {
        DIE("Work group size %zux%zu is too big for kernel %s"
            " - maximum is %zu\n",
                LOCAL_X, LOCAL_Y, kernel_name,
                max_wg_size);
    }

    fprintf(DBGOUT, "Done\n");
    fflush(DBGOUT);
}

cl::Program CloverChunk::compileProgram
(const std::string& source,
 const std::string& options)
{
    // catches any warnings/errors in the build
    std::stringstream errstream("");

    // very verbose
    //fprintf(stderr, "Making with source:\n%s\n", source.c_str());
    //fprintf(DBGOUT, "Making with options string:\n%s\n", options.c_str());
    fflush(DBGOUT);
    cl::Program program;

    cl::Program::Sources sources;
    sources = cl::Program::Sources(1, std::make_pair(source.c_str(), source.length()));

    try
    {
        program = cl::Program(context, sources);
        std::vector<cl::Device> dev_vec(1, device);
        program.build(dev_vec, options.c_str());
    }
    catch (cl::Error e)
    {
        fprintf(stderr, "Errors in creating program\n");

        try
        {
            errstream << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        }
        catch (cl::Error ie)
        {
            DIE("Error in retrieving build info\n");
        }

        std::string errs(errstream.str());
        //DIE("%s\n", errs.c_str());
        throw errs;
    }

    // return
    errstream << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
    std::string errs(errstream.str());

    // some will print out an empty warning log
    if (errs.size() > 10)
    {
        fprintf(DBGOUT, "Warnings:\n%s\n", errs.c_str());
    }

    return program;
}

void CloverChunk::initSizes
(void)
{
#if defined(ONED_KERNEL_LAUNCHES)
    size_t glob_x = x_max+5;
    size_t glob_y = y_max+5;
    total_cells = glob_x*glob_y;

    // pad as below
    while (total_cells % LOCAL_X)
    {
        total_cells++;
    }

    fprintf(DBGOUT, "Global size = %zu\n", total_cells);
    global_size = cl::NDRange(total_cells);
#else
    fprintf(DBGOUT, "Local size = %zux%zu\n", LOCAL_X, LOCAL_Y);

    // pad the global size so the local size fits
    size_t glob_x = x_max+5 +
        (((x_max+5)%LOCAL_X == 0) ? 0 : (LOCAL_X - ((x_max+5)%LOCAL_X)));
    size_t glob_y = y_max+5 +
        (((y_max+5)%LOCAL_Y == 0) ? 0 : (LOCAL_Y - ((y_max+5)%LOCAL_Y)));
    total_cells = glob_x*glob_y;

    fprintf(DBGOUT, "Global size = %zux%zu\n", glob_x, glob_y);
    global_size = cl::NDRange(glob_x, glob_y);
#endif

    /*
     *  update halo kernels need specific work group sizes - not doing a
     *  reduction, so can just fit it to the row/column even if its not a pwoer
     *  of 2
     */
    // get max local size for the update kernels
    size_t max_update_wg_sz;
    cl::detail::errHandler(
        clGetKernelWorkGroupInfo(update_halo_bottom_device(),
                                 device(),
                                 CL_KERNEL_WORK_GROUP_SIZE,
                                 sizeof(size_t),
                                 &max_update_wg_sz,
                                 NULL));
    fprintf(DBGOUT, "Max work group size for update halo is %zu\n", max_update_wg_sz);

    // subdivide row size until it will fit
    size_t local_row_size = x_max+5;
    while (local_row_size > max_update_wg_sz/2)
    {
        local_row_size = local_row_size/2;
    }
    fprintf(DBGOUT, "Local row work group size is %zu\n", local_row_size);

    update_ud_local_size[0] = cl::NDRange(local_row_size, 1);
    update_ud_local_size[1] = cl::NDRange(local_row_size, 2);

    size_t global_row_size = local_row_size;
    while (global_row_size < x_max+5)
    {
        global_row_size += local_row_size;
    }
    update_ud_global_size[0] = cl::NDRange(global_row_size, 1);
    update_ud_global_size[1] = cl::NDRange(global_row_size, 2);

    // same for column
    size_t local_column_size = y_max+5;
    while (local_column_size > max_update_wg_sz/2)
    {
        local_column_size = local_column_size/2;
    }

    if (CL_DEVICE_TYPE_ACCELERATOR == desired_type)
    {
        // on xeon phi, needs to be 16 so that update left/right kernels dont go really slow
        local_column_size = 16;
    }

    fprintf(DBGOUT, "Local column work group size is %zu\n", local_column_size);

    update_lr_local_size[0] = cl::NDRange(1, local_column_size);
    update_lr_local_size[1] = cl::NDRange(2, local_column_size);

    size_t global_column_size = local_column_size;
    while (global_column_size < y_max+5)
    {
        global_column_size += local_column_size;
    }
    update_lr_global_size[0] = cl::NDRange(1, global_column_size);
    update_lr_global_size[1] = cl::NDRange(2, global_column_size);

    fprintf(DBGOUT, "Update halo parameters calculated\n");

    /*
     *  figure out offset launch sizes for the various kernels
     *  no 'smart' way to do this?
     */
    #define FIND_PADDING_SIZE(knl, vmin, vmax, hmin, hmax)                      \
    {                                                                            \
        size_t global_horz_size = (-(hmin)) + (hmax) + x_max;                   \
        while (global_horz_size % LOCAL_X) global_horz_size++;                  \
        size_t global_vert_size = (-(vmin)) + (vmax) + y_max;                   \
        while (global_vert_size % LOCAL_Y) global_vert_size++;                  \
        launch_specs_t cur_specs;                                               \
        cur_specs.global = cl::NDRange(global_horz_size, global_vert_size);     \
        cur_specs.offset = cl::NDRange(x_min + 1 + (vmin), y_min + 1 + (hmin)); \
        launch_specs[#knl"_device"] = cur_specs;                                \
    }

    FIND_PADDING_SIZE(ideal_gas, 0, 0, 0, 0); // works
    FIND_PADDING_SIZE(accelerate, 0, 1, 0, 1); // works
    FIND_PADDING_SIZE(flux_calc, 0, 0, 1, 1);
    FIND_PADDING_SIZE(viscosity, 0, 0, 0, 0); // works
    FIND_PADDING_SIZE(revert, 0, 0, 0, 0); // works
    FIND_PADDING_SIZE(reset_field, 0, 1, 0, 1); // works
    FIND_PADDING_SIZE(set_field, 0, 1, 0, 1); // works
    FIND_PADDING_SIZE(field_summary, 0, 0, 0, 0);
    FIND_PADDING_SIZE(calc_dt, 0, 0, 0, 0);

    FIND_PADDING_SIZE(advec_mom_vol, -2, 2, -2, 2); // works
    FIND_PADDING_SIZE(advec_mom_node_flux_post_x, -1, 1, -1, 2);
    FIND_PADDING_SIZE(advec_mom_node_pre_x, 0, 1, -1, 2); // works
    FIND_PADDING_SIZE(advec_mom_flux_x, 0, 1, -1, 1); // works
    FIND_PADDING_SIZE(advec_mom_xvel, 0, 1, 0, 1); // works
    FIND_PADDING_SIZE(advec_mom_node_flux_post_y, -1, 2, -1, 1);
    FIND_PADDING_SIZE(advec_mom_node_pre_y, -1, 2, 0, 1); // works
    FIND_PADDING_SIZE(advec_mom_flux_y, -1, 1, 0, 1); // works
    FIND_PADDING_SIZE(advec_mom_yvel, 0, 1, 0, 1); // works

    FIND_PADDING_SIZE(advec_cell_pre_vol_x, -2, 2, -2, 2); // works
    FIND_PADDING_SIZE(advec_cell_ener_flux_x, 0, 0, 0, 2); // works
    FIND_PADDING_SIZE(advec_cell_x, 0, 0, 0, 0); // works
    FIND_PADDING_SIZE(advec_cell_pre_vol_y, -2, 2, -2, 2); // works
    FIND_PADDING_SIZE(advec_cell_ener_flux_y, 0, 2, 0, 2); // works
    FIND_PADDING_SIZE(advec_cell_y, 0, 0, 0, 0); // works

    FIND_PADDING_SIZE(PdV_predict, 0, 0, 0, 0); // works
    FIND_PADDING_SIZE(PdV_not_predict, 0, 0, 0, 0); // works

    FIND_PADDING_SIZE(initialise_chunk_first, 0, 3, 0, 3);
    FIND_PADDING_SIZE(initialise_chunk_second, -2, 2, -2, 2);
    FIND_PADDING_SIZE(generate_chunk_init, -2, 2, -2, 2);
    FIND_PADDING_SIZE(generate_chunk, -2, 2, -2, 2);

    FIND_PADDING_SIZE(generate_chunk, -2, 2, -2, 2);
}

