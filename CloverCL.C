/*Crown Copyright 2012 AWE.
*
* This file is part of CloverLeaf.
*
* CloverLeaf is free software: you can redistribute it and/or modify it under
* the terms of the GNU General Public License as published by the
* Free Software Foundation, either version 3 of the License, or (at your option)
* any later version.
*
* CloverLeaf is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
* details.
*
* You should have received a copy of the GNU General Public License along with
* CloverLeaf. If not, see http://www.gnu.org/licenses/. */

/**
 *  @brief CloverCL static class.
 *  @author Andrew Mallinson, David Beckingsale
 *  @details Contains common functionality required by all OCL kernels 
*/

#include "mpi.h"
#include "CloverCL.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <sstream>
#include <math.h>

bool CloverCL::initialised;

cl::Platform CloverCL::platform;
cl::Context CloverCL::context;
cl::Device CloverCL::device;
cl::Program CloverCL::program;

cl::CommandQueue CloverCL::queue;
cl::CommandQueue CloverCL::outoforder_queue;

size_t CloverCL::prefer_wg_multiple;
size_t CloverCL::max_reduction_wg_size;
cl_uint CloverCL::device_procs;
size_t CloverCL::device_max_wg_size;
cl_ulong CloverCL::device_local_mem_size;
cl_device_type CloverCL::device_type; 
int CloverCL::number_of_red_levels;

int CloverCL::xmax_plusfour_rounded_comms;
int CloverCL::xmax_plusfive_rounded_comms;
int CloverCL::ymax_plusfour_rounded_comms;
int CloverCL::ymax_plusfive_rounded_comms;

int CloverCL::xmax_plusfour_rounded_updatehalo;
int CloverCL::xmax_plusfive_rounded_updatehalo;
int CloverCL::ymax_plusfour_rounded_updatehalo;
int CloverCL::ymax_plusfive_rounded_updatehalo;

int CloverCL::mpi_rank; 
int CloverCL::xmax_c;
int CloverCL::ymax_c;

cl::Buffer CloverCL::density0_buffer;
cl::Buffer CloverCL::density1_buffer;
cl::Buffer CloverCL::energy0_buffer;
cl::Buffer CloverCL::energy1_buffer;
cl::Buffer CloverCL::pressure_buffer;
cl::Buffer CloverCL::soundspeed_buffer;
cl::Buffer CloverCL::celldx_buffer;
cl::Buffer CloverCL::celldy_buffer;
cl::Buffer CloverCL::viscosity_buffer;
cl::Buffer CloverCL::xvel0_buffer;
cl::Buffer CloverCL::yvel0_buffer;
cl::Buffer CloverCL::xvel1_buffer;
cl::Buffer CloverCL::yvel1_buffer;
cl::Buffer CloverCL::vol_flux_x_buffer;
cl::Buffer CloverCL::vol_flux_y_buffer;
cl::Buffer CloverCL::mass_flux_x_buffer;
cl::Buffer CloverCL::mass_flux_y_buffer;

cl::Buffer CloverCL::volume_buffer;
cl::Buffer CloverCL::vertexdx_buffer;
cl::Buffer CloverCL::vertexx_buffer;
cl::Buffer CloverCL::vertexdy_buffer;
cl::Buffer CloverCL::vertexy_buffer;
cl::Buffer CloverCL::cellx_buffer;
cl::Buffer CloverCL::celly_buffer;
cl::Buffer CloverCL::xarea_buffer;
cl::Buffer CloverCL::yarea_buffer;

cl::Buffer CloverCL::dt_min_val_buffer;
cl::Buffer CloverCL::vol_sum_val_buffer;
cl::Buffer CloverCL::mass_sum_val_buffer;
cl::Buffer CloverCL::ie_sum_val_buffer;
cl::Buffer CloverCL::ke_sum_val_buffer;
cl::Buffer CloverCL::press_sum_val_buffer;

cl::Buffer CloverCL::state_density_buffer;
cl::Buffer CloverCL::state_energy_buffer;
cl::Buffer CloverCL::state_xvel_buffer;
cl::Buffer CloverCL::state_yvel_buffer;
cl::Buffer CloverCL::state_xmin_buffer;
cl::Buffer CloverCL::state_xmax_buffer;
cl::Buffer CloverCL::state_ymin_buffer;
cl::Buffer CloverCL::state_ymax_buffer;
cl::Buffer CloverCL::state_radius_buffer;
cl::Buffer CloverCL::state_geometry_buffer;

cl::Buffer CloverCL::cpu_min_red_buffer;
cl::Buffer CloverCL::cpu_vol_red_buffer;
cl::Buffer CloverCL::cpu_mass_red_buffer;
cl::Buffer CloverCL::cpu_ie_red_buffer;
cl::Buffer CloverCL::cpu_ke_red_buffer;
cl::Buffer CloverCL::cpu_press_red_buffer;

cl::Buffer CloverCL::work_array1_buffer;
cl::Buffer CloverCL::work_array2_buffer;
cl::Buffer CloverCL::work_array3_buffer;
cl::Buffer CloverCL::work_array4_buffer;
cl::Buffer CloverCL::work_array5_buffer;
cl::Buffer CloverCL::work_array6_buffer;
cl::Buffer CloverCL::work_array7_buffer;

cl::Buffer CloverCL::top_send_buffer;
cl::Buffer CloverCL::top_recv_buffer;
cl::Buffer CloverCL::bottom_send_buffer;
cl::Buffer CloverCL::bottom_recv_buffer;
cl::Buffer CloverCL::left_send_buffer;
cl::Buffer CloverCL::left_recv_buffer;
cl::Buffer CloverCL::right_send_buffer;
cl::Buffer CloverCL::right_recv_buffer;

cl::Kernel CloverCL::ideal_gas_predict_knl;
cl::Kernel CloverCL::ideal_gas_NO_predict_knl;
cl::Kernel CloverCL::viscosity_knl;
cl::Kernel CloverCL::flux_calc_knl;
cl::Kernel CloverCL::accelerate_knl;
cl::Kernel CloverCL::advec_mom_vol_knl;
cl::Kernel CloverCL::advec_mom_node_x_knl;
cl::Kernel CloverCL::advec_mom_node_mass_pre_x_knl;
cl::Kernel CloverCL::advec_mom_flux_x_vec1_knl;
cl::Kernel CloverCL::advec_mom_flux_x_vecnot1_knl;
cl::Kernel CloverCL::advec_mom_vel_x_knl;
cl::Kernel CloverCL::advec_mom_node_y_knl;
cl::Kernel CloverCL::advec_mom_node_mass_pre_y_knl;
cl::Kernel CloverCL::advec_mom_flux_y_vec1_knl;
cl::Kernel CloverCL::advec_mom_flux_y_vecnot1_knl;
cl::Kernel CloverCL::advec_mom_vel_y_knl;
cl::Kernel CloverCL::dt_calc_knl;
cl::Kernel CloverCL::advec_cell_xdir_sec1_s1_knl;
cl::Kernel CloverCL::advec_cell_xdir_sec1_s2_knl;
cl::Kernel CloverCL::advec_cell_xdir_sec2_knl;
cl::Kernel CloverCL::advec_cell_xdir_sec3_knl;
cl::Kernel CloverCL::advec_cell_ydir_sec1_s1_knl;
cl::Kernel CloverCL::advec_cell_ydir_sec1_s2_knl;
cl::Kernel CloverCL::advec_cell_ydir_sec2_knl;
cl::Kernel CloverCL::advec_cell_ydir_sec3_knl;
cl::Kernel CloverCL::pdv_correct_knl;
cl::Kernel CloverCL::pdv_predict_knl;
cl::Kernel CloverCL::reset_field_knl;
cl::Kernel CloverCL::revert_knl;
cl::Kernel CloverCL::generate_chunk_knl;
cl::Kernel CloverCL::initialise_chunk_cell_x_knl;
cl::Kernel CloverCL::initialise_chunk_cell_y_knl;
cl::Kernel CloverCL::initialise_chunk_vertex_x_knl;
cl::Kernel CloverCL::initialise_chunk_vertex_y_knl;
cl::Kernel CloverCL::initialise_chunk_volume_area_knl;
cl::Kernel CloverCL::field_summary_knl;
cl::Kernel CloverCL::update_halo_left_cell_knl;
cl::Kernel CloverCL::update_halo_right_cell_knl;
cl::Kernel CloverCL::update_halo_top_cell_knl;
cl::Kernel CloverCL::update_halo_bottom_cell_knl;
cl::Kernel CloverCL::update_halo_left_vel_knl;
cl::Kernel CloverCL::update_halo_right_vel_knl;
cl::Kernel CloverCL::update_halo_top_vel_knl;
cl::Kernel CloverCL::update_halo_bottom_vel_knl;
cl::Kernel CloverCL::update_halo_left_flux_x_knl;
cl::Kernel CloverCL::update_halo_right_flux_x_knl;
cl::Kernel CloverCL::update_halo_top_flux_x_knl;
cl::Kernel CloverCL::update_halo_bottom_flux_x_knl;
cl::Kernel CloverCL::update_halo_left_flux_y_knl;
cl::Kernel CloverCL::update_halo_right_flux_y_knl;
cl::Kernel CloverCL::update_halo_top_flux_y_knl;
cl::Kernel CloverCL::update_halo_bottom_flux_y_knl;
cl::Kernel CloverCL::read_top_buffer_knl;
cl::Kernel CloverCL::read_right_buffer_knl;
cl::Kernel CloverCL::read_bottom_buffer_knl;
cl::Kernel CloverCL::read_left_buffer_knl;
cl::Kernel CloverCL::write_top_buffer_knl;
cl::Kernel CloverCL::write_right_buffer_knl;
cl::Kernel CloverCL::write_bottom_buffer_knl;
cl::Kernel CloverCL::write_left_buffer_knl;
cl::Kernel CloverCL::minimum_red_cpu_knl;
cl::Kernel CloverCL::vol_sum_red_cpu_knl; 
cl::Kernel CloverCL::mass_sum_red_cpu_knl; 
cl::Kernel CloverCL::ie_sum_red_cpu_knl; 
cl::Kernel CloverCL::ke_sum_red_cpu_knl; 
cl::Kernel CloverCL::press_sum_red_cpu_knl; 

std::vector<cl::Kernel> CloverCL::min_reduction_kernels;
std::vector<cl::Kernel> CloverCL::vol_sum_reduction_kernels;
std::vector<cl::Kernel> CloverCL::mass_sum_reduction_kernels;
std::vector<cl::Kernel> CloverCL::ie_sum_reduction_kernels;
std::vector<cl::Kernel> CloverCL::ke_sum_reduction_kernels;
std::vector<cl::Kernel> CloverCL::press_sum_reduction_kernels;

std::vector<int> CloverCL::num_workitems_tolaunch;
std::vector<int> CloverCL::num_workitems_per_wg;
std::vector<int> CloverCL::local_mem_size;
std::vector<int> CloverCL::size_limits;
std::vector<int> CloverCL::buffer_sizes;
std::vector<bool> CloverCL::input_even;
std::vector<int> CloverCL::num_elements_per_wi;

std::vector<cl::Buffer> CloverCL::min_interBuffers;
std::vector<cl::Buffer> CloverCL::vol_interBuffers;
std::vector<cl::Buffer> CloverCL::mass_interBuffers;
std::vector<cl::Buffer> CloverCL::ie_interBuffers;
std::vector<cl::Buffer> CloverCL::ke_interBuffers;
std::vector<cl::Buffer> CloverCL::press_interBuffers;

std::vector<cl::LocalSpaceArg> CloverCL::min_local_memory_objects;
std::vector<cl::LocalSpaceArg> CloverCL::vol_local_memory_objects;
std::vector<cl::LocalSpaceArg> CloverCL::mass_local_memory_objects;
std::vector<cl::LocalSpaceArg> CloverCL::ie_local_memory_objects;
std::vector<cl::LocalSpaceArg> CloverCL::ke_local_memory_objects;
std::vector<cl::LocalSpaceArg> CloverCL::press_local_memory_objects;

std::vector<cl::Event> CloverCL::global_events;
cl::Event CloverCL::last_event;

#if PROFILE_OCL_KERNELS
long CloverCL::accelerate_time;
long CloverCL::advec_cell_time;
long CloverCL::advec_mom_time;
long CloverCL::calc_dt_time;
long CloverCL::comms_buffers_time;
long CloverCL::field_summ_time;
long CloverCL::flux_calc_time;
long CloverCL::generate_chunk_time;
long CloverCL::ideal_gas_time;
long CloverCL::initialise_chunk_time;
long CloverCL::pdv_time;
long CloverCL::reset_field_time;
long CloverCL::revert_time;
long CloverCL::udpate_halo_time;
long CloverCL::viscosity_time;

double CloverCL::accelerate_count;
double CloverCL::advec_cell_count;
double CloverCL::advec_mom_count;
double CloverCL::calc_dt_count;
double CloverCL::comms_buffers_count;
double CloverCL::field_summ_count;
double CloverCL::flux_calc_count;
double CloverCL::generate_chunk_count;
double CloverCL::ideal_gas_count;
double CloverCL::initialise_chunk_count;
double CloverCL::pdv_count;
double CloverCL::reset_field_count;
double CloverCL::revert_count;
double CloverCL::udpate_halo_count;
double CloverCL::viscosity_count;
#endif


void CloverCL::init(
        std::string platform_name,
        std::string platform_type,
        int x_min,
        int x_max,
        int y_min,
        int y_max,
        int num_states,
        double g_small,
        double g_big,
        double dtmin,
        double dtc_safe,
        double dtu_safe,
        double dtv_safe,
        double dtdiv_safe) 
{
#ifdef OCL_VERBOSE
    std::cout << "num states = " << num_states << std::endl;
    std::cout << "x_max = " << x_max << std::endl;
    std::cout << "y_max = " << y_max << std::endl;

    printDeviceInformation();

    //insert printouts of all the autotuning parameters
#endif
    initPlatform(platform_name);
    initContext(platform_type);
    initDevice(0);
    initCommandQueue();
    loadProgram(x_min, x_max, y_min, y_max);
    determineWorkGroupSizeInfo();

    calculateKernelLaunchParams(x_max, y_max);

    calculateReductionStructure(x_max, y_max);

    createBuffers(x_max, y_max, num_states);
    allocateReductionInterBuffers();
    allocateLocalMemoryObjects();
    build_reduction_kernel_objects(); 

#ifdef DUMP_BINARY
    dumpBinary();
#endif

    initialiseKernelArgs(x_min, x_max, y_min, y_max,
                         g_small, g_big, dtmin, dtc_safe,
                         dtu_safe, dtv_safe, dtdiv_safe);
    initialised = true;

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    xmax_c = x_max;
    ymax_c = y_max; 

#if PROFILE_OCL_KERNELS
    accelerate_time = 0;
    advec_cell_time = 0;
    advec_mom_time = 0;
    calc_dt_time = 0;
    comms_buffers_time = 0;
    field_summ_time = 0;
    flux_calc_time = 0;
    generate_chunk_time = 0;
    ideal_gas_time = 0;
    initialise_chunk_time = 0;
    pdv_time = 0;
    reset_field_time = 0;
    revert_time = 0;
    udpate_halo_time = 0;
    viscosity_time = 0;
    
    accelerate_count = 0;
    advec_cell_count = 0;
    advec_mom_count = 0;
    calc_dt_count = 0;
    comms_buffers_count = 0;
    field_summ_count = 0;
    flux_calc_count = 0;
    generate_chunk_count = 0;
    ideal_gas_count = 0;
    initialise_chunk_count = 0;
    pdv_count = 0;
    reset_field_count = 0;
    revert_count = 0;
    udpate_halo_count = 0;
    viscosity_count = 0;
#endif
}


void CloverCL::calculateKernelLaunchParams(int xmax, int ymax) {

    int x_rnd, y_rnd; 


    x_rnd = ( (xmax+4) / local_wg_largedim_updatehalo ) * local_wg_largedim_updatehalo;

    if (x_rnd != xmax+4) {
        x_rnd = x_rnd + local_wg_largedim_updatehalo; 
    }
    
    xmax_plusfour_rounded_updatehalo = x_rnd; 

    x_rnd = ( (xmax+4) / local_wg_largedim_comms) * local_wg_largedim_comms;

    if (x_rnd != xmax+4) {
        x_rnd = x_rnd + local_wg_largedim_comms; 
    }
    
    xmax_plusfour_rounded_comms = x_rnd; 


    x_rnd = ( (xmax+5) / local_wg_largedim_updatehalo) * local_wg_largedim_updatehalo;

    if (x_rnd != xmax+5) {
        x_rnd = x_rnd + local_wg_largedim_updatehalo; 
    }

    xmax_plusfive_rounded_updatehalo = x_rnd; 

    x_rnd = ( (xmax+5) / local_wg_largedim_comms) * local_wg_largedim_comms;

    if (x_rnd != xmax+5) {
        x_rnd = x_rnd + local_wg_largedim_comms; 
    }

    xmax_plusfive_rounded_comms = x_rnd; 




    y_rnd = ( (ymax+4) / local_wg_largedim_updatehalo) * local_wg_largedim_updatehalo;

    if (y_rnd != ymax+4) {
        y_rnd = y_rnd + local_wg_largedim_updatehalo; 
    }

    ymax_plusfour_rounded_updatehalo = y_rnd; 

    y_rnd = ( (ymax+4) / local_wg_largedim_comms) * local_wg_largedim_comms;

    if (y_rnd != ymax+4) {
        y_rnd = y_rnd + local_wg_largedim_comms; 
    }

    ymax_plusfour_rounded_comms = y_rnd; 



    y_rnd = ( (ymax+5) / local_wg_largedim_updatehalo) * local_wg_largedim_updatehalo;

    if (y_rnd != ymax+5) {
        y_rnd = y_rnd + local_wg_largedim_updatehalo; 
    }

    ymax_plusfive_rounded_updatehalo = y_rnd; 

    y_rnd = ( (ymax+5) / local_wg_largedim_comms) * local_wg_largedim_comms;

    if (y_rnd != ymax+5) {
        y_rnd = y_rnd + local_wg_largedim_comms; 
    }

    ymax_plusfive_rounded_comms = y_rnd; 



#ifdef OCL_VERBOSE
    std::cout << "Kernel launch xmaxplusfour_comms rounded: " << xmax_plusfour_rounded_comms << std::endl;
    std::cout << "Kernel launch xmaxplusfive_comms rounded: " << xmax_plusfive_rounded_comms << std::endl;
    std::cout << "Kernel launch ymaxplusfour_comms rounded: " << ymax_plusfour_rounded_comms << std::endl;
    std::cout << "Kernel launch ymaxplusfive_comms rounded: " << ymax_plusfive_rounded_comms << std::endl;

    std::cout << "Kernel launch xmaxplusfour_updatehalo rounded: " << xmax_plusfour_rounded_updatehalo << std::endl;
    std::cout << "Kernel launch xmaxplusfive_updatehalo rounded: " << xmax_plusfive_rounded_updatehalo << std::endl;
    std::cout << "Kernel launch ymaxplusfour_updatehalo rounded: " << ymax_plusfour_rounded_updatehalo << std::endl;
    std::cout << "Kernel launch ymaxplusfive_updatehalo rounded: " << ymax_plusfive_rounded_updatehalo << std::endl;
#endif
}

void CloverCL::determineWorkGroupSizeInfo() {

    ideal_gas_predict_knl.getWorkGroupInfo(device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, &prefer_wg_multiple);
    ideal_gas_predict_knl.getWorkGroupInfo(device, CL_KERNEL_WORK_GROUP_SIZE, &max_reduction_wg_size);
    
    device.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &device_procs);
    device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &device_max_wg_size);
    device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &device_local_mem_size);

#ifdef OCL_VERBOSE
    std::cout << "Size of min reduction work group multiple: " << prefer_wg_multiple << std::endl;
    std::cout << "Size of min reduction max work group size: " << max_reduction_wg_size << std::endl;
    std::cout << "Device Num of compute units: " << device_procs << std::endl;
    std::cout << "Device Max WG Size: " << device_max_wg_size << std::endl;
    std::cout << "Device Local memmory size: " << device_local_mem_size << std::endl;

    if (device_type == CL_DEVICE_TYPE_CPU) {
        std::cout << "Device Type selected: CPU" << std::endl;
    }
    else if (device_type == CL_DEVICE_TYPE_GPU) {
        std::cout << "Device Type selected: GPU" << std::endl;
    }
    else if (device_type == CL_DEVICE_TYPE_ACCELERATOR) {
        std::cout << "Device Type selected: ACCELERATOR" << std::endl;
    }
    else if (device_type == CL_DEVICE_TYPE_DEFAULT) {
        std::cout << "Device Type selected: DEFAULT" << std::endl;
    }
    else {
        std::cout << "ERROR Device Type selected: NOT SUPPORTED" << std::endl;
    }
#endif
}

void CloverCL::build_reduction_kernel_objects() {

    cl_int err; 

    min_reduction_kernels.clear();
    vol_sum_reduction_kernels.clear();
    mass_sum_reduction_kernels.clear();
    ie_sum_reduction_kernels.clear();
    ke_sum_reduction_kernels.clear();
    press_sum_reduction_kernels.clear();

    if ( (device_type == CL_DEVICE_TYPE_CPU) || (device_type == CL_DEVICE_TYPE_ACCELERATOR) ) {
        //build the CPU and Phi reduction objects 

        if ( number_of_red_levels == 1 ) { 

            //build level 1 of CPU reduction 
            min_reduction_kernels.push_back( cl::Kernel(program, "reduction_minimum_cpu_ocl_kernel", &err) );
            vol_sum_reduction_kernels.push_back( cl::Kernel(program, "reduction_sum_cpu_ocl_kernel", &err));
            mass_sum_reduction_kernels.push_back( cl::Kernel(program, "reduction_sum_cpu_ocl_kernel", &err));
            ie_sum_reduction_kernels.push_back( cl::Kernel(program, "reduction_sum_cpu_ocl_kernel", &err));
            ke_sum_reduction_kernels.push_back( cl::Kernel(program, "reduction_sum_cpu_ocl_kernel", &err));
            press_sum_reduction_kernels.push_back( cl::Kernel(program, "reduction_sum_cpu_ocl_kernel", &err));

            min_reduction_kernels[0].setArg(      0, CloverCL::work_array1_buffer);
            vol_sum_reduction_kernels[0].setArg(  0, CloverCL::work_array1_buffer);
            mass_sum_reduction_kernels[0].setArg( 0, CloverCL::work_array2_buffer);
            ie_sum_reduction_kernels[0].setArg(   0, CloverCL::work_array3_buffer);
            ke_sum_reduction_kernels[0].setArg(   0, CloverCL::work_array4_buffer);
            press_sum_reduction_kernels[0].setArg(0, CloverCL::work_array5_buffer);

            min_reduction_kernels[1].setArg(      1, CloverCL::dt_min_val_buffer);
            vol_sum_reduction_kernels[1].setArg(1, CloverCL::vol_sum_val_buffer); 
            mass_sum_reduction_kernels[1].setArg(1, CloverCL::mass_sum_val_buffer); 
            ie_sum_reduction_kernels[1].setArg(1, CloverCL::ie_sum_val_buffer);
            ke_sum_reduction_kernels[1].setArg(1, CloverCL::ke_sum_val_buffer); 
            press_sum_reduction_kernels[1].setArg(1, CloverCL::press_sum_val_buffer);

            min_reduction_kernels[0].setArg(      2, CloverCL::num_elements_per_wi[0]);
            vol_sum_reduction_kernels[0].setArg(2, CloverCL::num_elements_per_wi[0]); 
            mass_sum_reduction_kernels[0].setArg(2, CloverCL::num_elements_per_wi[0]);
            ie_sum_reduction_kernels[0].setArg(2, CloverCL::num_elements_per_wi[0]); 
            ke_sum_reduction_kernels[0].setArg(2, CloverCL::num_elements_per_wi[0]); 
            press_sum_reduction_kernels[0].setArg(2, CloverCL::num_elements_per_wi[0]); 

        }
        else {

            //build level 1 of CPU reduction 
            min_reduction_kernels.push_back( cl::Kernel(program, "reduction_minimum_cpu_ocl_kernel", &err) );
            vol_sum_reduction_kernels.push_back( cl::Kernel(program, "reduction_sum_cpu_ocl_kernel", &err));
            mass_sum_reduction_kernels.push_back( cl::Kernel(program, "reduction_sum_cpu_ocl_kernel", &err));
            ie_sum_reduction_kernels.push_back( cl::Kernel(program, "reduction_sum_cpu_ocl_kernel", &err));
            ke_sum_reduction_kernels.push_back( cl::Kernel(program, "reduction_sum_cpu_ocl_kernel", &err));
            press_sum_reduction_kernels.push_back( cl::Kernel(program, "reduction_sum_cpu_ocl_kernel", &err));

            min_reduction_kernels[0].setArg(      0, CloverCL::work_array1_buffer);
            vol_sum_reduction_kernels[0].setArg(  0, CloverCL::work_array1_buffer);
            mass_sum_reduction_kernels[0].setArg( 0, CloverCL::work_array2_buffer);
            ie_sum_reduction_kernels[0].setArg(   0, CloverCL::work_array3_buffer);
            ke_sum_reduction_kernels[0].setArg(   0, CloverCL::work_array4_buffer);
            press_sum_reduction_kernels[0].setArg(0, CloverCL::work_array5_buffer);
            
            min_reduction_kernels[0].setArg(      1, CloverCL::cpu_min_red_buffer);
            vol_sum_reduction_kernels[0].setArg(1, CloverCL::cpu_vol_red_buffer); 
            mass_sum_reduction_kernels[0].setArg(1, CloverCL::cpu_mass_red_buffer); 
            ie_sum_reduction_kernels[0].setArg(1, CloverCL::cpu_ie_red_buffer); 
            ke_sum_reduction_kernels[0].setArg(1, CloverCL::cpu_ke_red_buffer); 
            press_sum_reduction_kernels[0].setArg(1, CloverCL::cpu_press_red_buffer); 

            min_reduction_kernels[0].setArg(      2, CloverCL::num_elements_per_wi[0]);
            vol_sum_reduction_kernels[0].setArg(2, CloverCL::num_elements_per_wi[0]); 
            mass_sum_reduction_kernels[0].setArg(2, CloverCL::num_elements_per_wi[0]);
            ie_sum_reduction_kernels[0].setArg(2, CloverCL::num_elements_per_wi[0]); 
            ke_sum_reduction_kernels[0].setArg(2, CloverCL::num_elements_per_wi[0]); 
            press_sum_reduction_kernels[0].setArg(2, CloverCL::num_elements_per_wi[0]); 


            //build level 2 of CPU reduction 
            min_reduction_kernels.push_back( cl::Kernel(program, "reduction_minimum_cpu_ocl_kernel", &err) );
            vol_sum_reduction_kernels.push_back( cl::Kernel(program, "reduction_sum_cpu_ocl_kernel", &err));
            mass_sum_reduction_kernels.push_back( cl::Kernel(program, "reduction_sum_cpu_ocl_kernel", &err));
            ie_sum_reduction_kernels.push_back( cl::Kernel(program, "reduction_sum_cpu_ocl_kernel", &err));
            ke_sum_reduction_kernels.push_back( cl::Kernel(program, "reduction_sum_cpu_ocl_kernel", &err));
            press_sum_reduction_kernels.push_back( cl::Kernel(program, "reduction_sum_cpu_ocl_kernel", &err));

            min_reduction_kernels[1].setArg(      0, CloverCL::cpu_min_red_buffer);
            vol_sum_reduction_kernels[1].setArg(0, CloverCL::cpu_vol_red_buffer); 
            mass_sum_reduction_kernels[1].setArg(0, CloverCL::cpu_mass_red_buffer); 
            ie_sum_reduction_kernels[1].setArg(0, CloverCL::cpu_ie_red_buffer); 
            ke_sum_reduction_kernels[1].setArg(0, CloverCL::cpu_ke_red_buffer);
            press_sum_reduction_kernels[1].setArg(0, CloverCL::cpu_press_red_buffer); 
            
            min_reduction_kernels[1].setArg(      1, CloverCL::dt_min_val_buffer);
            vol_sum_reduction_kernels[1].setArg(1, CloverCL::vol_sum_val_buffer); 
            mass_sum_reduction_kernels[1].setArg(1, CloverCL::mass_sum_val_buffer); 
            ie_sum_reduction_kernels[1].setArg(1, CloverCL::ie_sum_val_buffer);
            ke_sum_reduction_kernels[1].setArg(1, CloverCL::ke_sum_val_buffer); 
            press_sum_reduction_kernels[1].setArg(1, CloverCL::press_sum_val_buffer);

            min_reduction_kernels[1].setArg(      2, CloverCL::num_elements_per_wi[1]);
            vol_sum_reduction_kernels[1].setArg(2, CloverCL::num_elements_per_wi[1]); 
            mass_sum_reduction_kernels[1].setArg(2, CloverCL::num_elements_per_wi[1]); 
            ie_sum_reduction_kernels[1].setArg(2, CloverCL::num_elements_per_wi[1]); 
            ke_sum_reduction_kernels[1].setArg(2, CloverCL::num_elements_per_wi[1]); 
            press_sum_reduction_kernels[1].setArg(2, CloverCL::num_elements_per_wi[1]); 

        }
        
    }
    else if (CloverCL::device_type == CL_DEVICE_TYPE_GPU) {
        //build the GPU reduction objects

        for (int i=1; i<=CloverCL::number_of_red_levels; i++) {

            if (CloverCL::size_limits[i-1] == -1) { 

                //build a normal GPU reduction kernel
                min_reduction_kernels.push_back( cl::Kernel(program, "reduction_minimum_ocl_kernel", &err) );
                vol_sum_reduction_kernels.push_back( cl::Kernel(program, "reduction_sum_ocl_kernel", &err));
                mass_sum_reduction_kernels.push_back( cl::Kernel(program, "reduction_sum_ocl_kernel", &err));
                ie_sum_reduction_kernels.push_back( cl::Kernel(program, "reduction_sum_ocl_kernel", &err));
                ke_sum_reduction_kernels.push_back( cl::Kernel(program, "reduction_sum_ocl_kernel", &err));
                press_sum_reduction_kernels.push_back( cl::Kernel(program, "reduction_sum_ocl_kernel", &err));

                if (i==1) {
                    min_reduction_kernels[i-1].setArg(      0, CloverCL::work_array1_buffer);
                    vol_sum_reduction_kernels[i-1].setArg(  0, CloverCL::work_array1_buffer);
                    mass_sum_reduction_kernels[i-1].setArg( 0, CloverCL::work_array2_buffer);
                    ie_sum_reduction_kernels[i-1].setArg(   0, CloverCL::work_array3_buffer);
                    ke_sum_reduction_kernels[i-1].setArg(   0, CloverCL::work_array4_buffer);
                    press_sum_reduction_kernels[i-1].setArg(0, CloverCL::work_array5_buffer);
                }
                else {
                    min_reduction_kernels[i-1].setArg(0, CloverCL::min_interBuffers[i-2]);
                    vol_sum_reduction_kernels[i-1].setArg(  0, CloverCL::vol_interBuffers[i-2]);
                    mass_sum_reduction_kernels[i-1].setArg( 0, CloverCL::mass_interBuffers[i-2]);
                    ie_sum_reduction_kernels[i-1].setArg(   0, CloverCL::ie_interBuffers[i-2]);
                    ke_sum_reduction_kernels[i-1].setArg(   0, CloverCL::ke_interBuffers[i-2]);
                    press_sum_reduction_kernels[i-1].setArg(0, CloverCL::press_interBuffers[i-2]);
                }

                min_reduction_kernels[i-1].setArg(1, CloverCL::min_local_memory_objects[i-1]);
                vol_sum_reduction_kernels[i-1].setArg(1,   CloverCL::vol_local_memory_objects[i-1]);
                mass_sum_reduction_kernels[i-1].setArg(1,  CloverCL::mass_local_memory_objects[i-1]);
                ie_sum_reduction_kernels[i-1].setArg(1,    CloverCL::ie_local_memory_objects[i-1]);
                ke_sum_reduction_kernels[i-1].setArg(1,    CloverCL::ke_local_memory_objects[i-1]);
                press_sum_reduction_kernels[i-1].setArg(1, CloverCL::press_local_memory_objects[i-1]);

                if (i==CloverCL::number_of_red_levels) {
                    min_reduction_kernels[i-1].setArg(2, CloverCL::dt_min_val_buffer);
                    vol_sum_reduction_kernels[i-1].setArg(2,   CloverCL::vol_sum_val_buffer);
                    mass_sum_reduction_kernels[i-1].setArg(2,  CloverCL::mass_sum_val_buffer);
                    ie_sum_reduction_kernels[i-1].setArg(2,    CloverCL::ie_sum_val_buffer);
                    ke_sum_reduction_kernels[i-1].setArg(2,    CloverCL::ke_sum_val_buffer);
                    press_sum_reduction_kernels[i-1].setArg(2, CloverCL::press_sum_val_buffer);
                }
                else {
                    min_reduction_kernels[i-1].setArg(2, CloverCL::min_interBuffers[i-1]);
                    vol_sum_reduction_kernels[i-1].setArg(2,   CloverCL::vol_interBuffers[i-1]);
                    mass_sum_reduction_kernels[i-1].setArg(2,  CloverCL::mass_interBuffers[i-1]);
                    ie_sum_reduction_kernels[i-1].setArg(2,    CloverCL::ie_interBuffers[i-1]);
                    ke_sum_reduction_kernels[i-1].setArg(2,    CloverCL::ke_interBuffers[i-1]);
                    press_sum_reduction_kernels[i-1].setArg(2, CloverCL::press_interBuffers[i-1]);
                }
            }
            else {

                //build a last level GPU reduction kernel
                min_reduction_kernels.push_back( cl::Kernel(program, "reduction_minimum_last_ocl_kernel", &err)  );
                vol_sum_reduction_kernels.push_back( cl::Kernel(program, "reduction_sum_last_ocl_kernel", &err));
                mass_sum_reduction_kernels.push_back( cl::Kernel(program, "reduction_sum_last_ocl_kernel", &err));
                ie_sum_reduction_kernels.push_back( cl::Kernel(program, "reduction_sum_last_ocl_kernel", &err));
                ke_sum_reduction_kernels.push_back( cl::Kernel(program, "reduction_sum_last_ocl_kernel", &err));
                press_sum_reduction_kernels.push_back( cl::Kernel(program, "reduction_sum_last_ocl_kernel", &err));

                if (i==1) {
                    //if on first level then set input to equal source buffer
                    min_reduction_kernels[i-1].setArg(0, CloverCL::work_array1_buffer);
                    vol_sum_reduction_kernels[i-1].setArg(  0, CloverCL::work_array1_buffer);
                    mass_sum_reduction_kernels[i-1].setArg( 0, CloverCL::work_array2_buffer);
                    ie_sum_reduction_kernels[i-1].setArg(   0, CloverCL::work_array3_buffer);
                    ke_sum_reduction_kernels[i-1].setArg(   0, CloverCL::work_array4_buffer);
                    press_sum_reduction_kernels[i-1].setArg(0, CloverCL::work_array5_buffer);
                }
                else {
                    min_reduction_kernels[i-1].setArg(0, CloverCL::min_interBuffers[i-2]);
                    vol_sum_reduction_kernels[i-1].setArg(  0, CloverCL::vol_interBuffers[i-2]);
                    mass_sum_reduction_kernels[i-1].setArg( 0, CloverCL::mass_interBuffers[i-2]);
                    ie_sum_reduction_kernels[i-1].setArg(   0, CloverCL::ie_interBuffers[i-2]);
                    ke_sum_reduction_kernels[i-1].setArg(   0, CloverCL::ke_interBuffers[i-2]);
                    press_sum_reduction_kernels[i-1].setArg(0, CloverCL::press_interBuffers[i-2]);
                }

                min_reduction_kernels[i-1].setArg(1, CloverCL::min_local_memory_objects[i-1]);
                vol_sum_reduction_kernels[i-1].setArg(1,   CloverCL::vol_local_memory_objects[i-1]);
                mass_sum_reduction_kernels[i-1].setArg(1,  CloverCL::mass_local_memory_objects[i-1]);
                ie_sum_reduction_kernels[i-1].setArg(1,    CloverCL::ie_local_memory_objects[i-1]);
                ke_sum_reduction_kernels[i-1].setArg(1,    CloverCL::ke_local_memory_objects[i-1]);
                press_sum_reduction_kernels[i-1].setArg(1, CloverCL::press_local_memory_objects[i-1]);

                if (i==CloverCL::number_of_red_levels) {
                    //if last level of reduction set output to be output buffer
                    min_reduction_kernels[i-1].setArg(2, CloverCL::dt_min_val_buffer);
                    vol_sum_reduction_kernels[i-1].setArg(2,   CloverCL::vol_sum_val_buffer);
                    mass_sum_reduction_kernels[i-1].setArg(2,  CloverCL::mass_sum_val_buffer);
                    ie_sum_reduction_kernels[i-1].setArg(2,    CloverCL::ie_sum_val_buffer);
                    ke_sum_reduction_kernels[i-1].setArg(2,    CloverCL::ke_sum_val_buffer);
                    press_sum_reduction_kernels[i-1].setArg(2, CloverCL::press_sum_val_buffer);
                }
                else {
                    min_reduction_kernels[i-1].setArg(2, CloverCL::min_interBuffers[i-1]);
                    vol_sum_reduction_kernels[i-1].setArg(2,   CloverCL::vol_interBuffers[i-1]);
                    mass_sum_reduction_kernels[i-1].setArg(2,  CloverCL::mass_interBuffers[i-1]);
                    ie_sum_reduction_kernels[i-1].setArg(2,    CloverCL::ie_interBuffers[i-1]);
                    ke_sum_reduction_kernels[i-1].setArg(2,    CloverCL::ke_interBuffers[i-1]);
                    press_sum_reduction_kernels[i-1].setArg(2, CloverCL::press_interBuffers[i-1]);
                }

                min_reduction_kernels[i-1].setArg(3, CloverCL::size_limits[i-1]);
                vol_sum_reduction_kernels[i-1].setArg(3, CloverCL::size_limits[i-1]);
                mass_sum_reduction_kernels[i-1].setArg(3, CloverCL::size_limits[i-1]);
                ie_sum_reduction_kernels[i-1].setArg(3, CloverCL::size_limits[i-1]);
                ke_sum_reduction_kernels[i-1].setArg(3, CloverCL::size_limits[i-1]);
                press_sum_reduction_kernels[i-1].setArg(3, CloverCL::size_limits[i-1]);

                if (CloverCL::input_even[i-1]==true) {
                    min_reduction_kernels[i-1].setArg(4, 1);
                    vol_sum_reduction_kernels[i-1].setArg(4, 1);
                    mass_sum_reduction_kernels[i-1].setArg(4, 1);
                    ie_sum_reduction_kernels[i-1].setArg(4, 1);
                    ke_sum_reduction_kernels[i-1].setArg(4, 1);
                    press_sum_reduction_kernels[i-1].setArg(4, 1);
                }
                else {
                    min_reduction_kernels[i-1].setArg(4, 0);
                    vol_sum_reduction_kernels[i-1].setArg(4, 0);
                    mass_sum_reduction_kernels[i-1].setArg(4, 0);
                    ie_sum_reduction_kernels[i-1].setArg(4, 0);
                    ke_sum_reduction_kernels[i-1].setArg(4, 0);
                    press_sum_reduction_kernels[i-1].setArg(4, 0);
                }
            }
        }

    }
    else {
        std::cout << "ERROR in CloverCL build reduction structure objectes method: device type is unsupported" 
                  << std::endl;
    }
}

void CloverCL::calculateReductionStructure(int xmax, int ymax) {

    int x_rnd = ((xmax+2) / local_wg_x_calcdt_fieldsumm) * local_wg_x_calcdt_fieldsumm;

    if ((x_rnd != xmax+2))
        x_rnd = x_rnd + local_wg_x_calcdt_fieldsumm;

    int y_rnd = ((ymax+2) / local_wg_y_calcdt_fieldsumm ) * local_wg_y_calcdt_fieldsumm;

    if ((y_rnd != ymax+2))
        y_rnd = y_rnd + local_wg_y_calcdt_fieldsumm;

    int num_elements = (x_rnd / local_wg_x_calcdt_fieldsumm) * (y_rnd / local_wg_y_calcdt_fieldsumm);

    num_workitems_tolaunch.clear();
    num_workitems_per_wg.clear();
    local_mem_size.clear();
    size_limits.clear();
    buffer_sizes.clear();
    input_even.clear();
    num_elements_per_wi.clear();

    if ( (device_type == CL_DEVICE_TYPE_CPU) || (device_type == CL_DEVICE_TYPE_ACCELERATOR) ) {

        if ( num_elements < device_procs*2) { 
            //just launch one level of reduction as not enough elements

            number_of_red_levels = 1;

            num_workitems_tolaunch.push_back(1);
            num_workitems_per_wg.push_back(1);
            num_elements_per_wi.push_back(num_elements);
        }
        else {

            number_of_red_levels = 2;
            
            num_workitems_tolaunch.push_back(device_procs);
            num_workitems_per_wg.push_back(1);
            num_elements_per_wi.push_back(num_elements);

            num_workitems_tolaunch.push_back(1);
            num_workitems_per_wg.push_back(1);
            num_elements_per_wi.push_back(device_procs);
        }

#ifdef OCL_VERBOSE
        std::cout << "number_of_red_levels after loop: " << number_of_red_levels << std::endl;
        std::cout << "number of workitems to launch vector size: " << num_workitems_tolaunch.size() << std::endl;
        std::cout << "number of workitems per wg vector size: " << num_workitems_per_wg.size() << std::endl;
        std::cout << "number of local_mem_size vector size: " << local_mem_size.size() << std::endl;
        std::cout << "number of size_limits vector size: " << size_limits.size() << std::endl;
        std::cout << "number of buffer_sizes vector size: " << buffer_sizes.size() << std::endl;
        std::cout << "number of input_even vector size: " << input_even.size() << std::endl;
        std::cout << "number of num_elements_per_wi vector size: " << num_elements_per_wi.size() << std::endl;

        for (int i=0; i<number_of_red_levels; i++) {
            std::cout << "Red level:            " << i+1 << std::endl;
            std::cout << "Work items to launch: " << num_workitems_tolaunch[i] << std::endl;
            std::cout << "Work items per wg:    " << num_workitems_per_wg[i] << std::endl;
            //std::cout << "Size limit:           " << size_limits[i] << std::endl;
            std::cout << "Num Element per wi:   " << num_elements_per_wi[i] << std::endl;
            std::cout << std::endl;
        }
#endif

    }
    else if (device_type == CL_DEVICE_TYPE_GPU) {

        int wg_ingest_value, temp_wg_ingest_size, remaining_wis;
        int normal_wg_size;
        number_of_red_levels = 0;

        normal_wg_size = CloverCL::local_wg_x_reduction;
        wg_ingest_value = 2*normal_wg_size;

        if ( fmod(log2(max_reduction_wg_size),1)!=0  ) {
            //reduction workgroup size selected is not a power of 2
            std::cerr << "Error: reduction local workgroup size is NOT a power of 2" << std::endl;
            exit(EXIT_FAILURE);
        }

        if ( normal_wg_size > device_max_wg_size ) {
            std::cerr << "Error: reduction local workgroup size is greater than device maximum" << std::endl;
            exit(EXIT_FAILURE);
        }

        //add initial starting buffer to buffers vector
        buffer_sizes.push_back(num_elements);

        do {
            number_of_red_levels++;

            if (buffer_sizes.back() <= wg_ingest_value) {
            //only one workgroup required 

                if (buffer_sizes.back() == wg_ingest_value) {
                    num_workitems_tolaunch.push_back(normal_wg_size);
            	    num_workitems_per_wg.push_back(normal_wg_size);
            	    local_mem_size.push_back(normal_wg_size);
            	    size_limits.push_back(-1);
            	    input_even.push_back(true);
                }
                else {
            	    temp_wg_ingest_size = wg_ingest_value / 2;

            	    while( (temp_wg_ingest_size > buffer_sizes.back()) && (temp_wg_ingest_size >= prefer_wg_multiple*2 )  ) {
                        wg_ingest_value = temp_wg_ingest_size; 
            	        temp_wg_ingest_size = temp_wg_ingest_size / 2; 
            	    }
            	    normal_wg_size = wg_ingest_value / 2;

            	    num_workitems_tolaunch.push_back(normal_wg_size);
            	    num_workitems_per_wg.push_back(normal_wg_size);
            	    local_mem_size.push_back(normal_wg_size);

                    if (buffer_sizes.back() == wg_ingest_value) {
            	    //last level is a multiple of 2 there don't need a limit 
            	    size_limits.push_back(-1);
            	    input_even.push_back(true);
            	    }
            	    else if (buffer_sizes.back() % 2 == 0) {
            	        //last level input is even amount
            	        size_limits.push_back(buffer_sizes.back() / 2);
            	        input_even.push_back(true);
            	    }
            	    else {
            	        //last level input is odd amount
            	        size_limits.push_back(buffer_sizes.back() / 2);
            	        input_even.push_back(false);
            	    }
                }

                buffer_sizes.push_back(1);

            }
            else if ( buffer_sizes.back() % wg_ingest_value==0 ) {
                num_workitems_tolaunch.push_back(buffer_sizes.back() / wg_ingest_value * normal_wg_size);
                num_workitems_per_wg.push_back(normal_wg_size);
                local_mem_size.push_back(normal_wg_size);
                size_limits.push_back(-1);
                buffer_sizes.push_back(buffer_sizes.back() / wg_ingest_value);
                input_even.push_back(true);
            }
            else {
                //basic strategy is currently to use the maximum possible size of workgroup and then 
                //limit the number reduced in the final workgroup to allow for arbitrary sizes
                //this approach may well need changing but will do for now 
                //eg may be better to balance things across the GPU eg 1WG / MP

                num_workitems_tolaunch.push_back( (buffer_sizes.back() / wg_ingest_value + 1) * normal_wg_size);
                num_workitems_per_wg.push_back(normal_wg_size);
                local_mem_size.push_back(normal_wg_size);

                remaining_wis = buffer_sizes.back() % wg_ingest_value;
                size_limits.push_back(remaining_wis / 2);
                buffer_sizes.push_back(buffer_sizes.back() / wg_ingest_value + 1);
                if (remaining_wis % 2 == 0) {
                    input_even.push_back(true);
                }
                else {
                    input_even.push_back(false);
                }
            }

        } while(buffer_sizes.back() != 1);

#ifdef OCL_VERBOSE
        std::cout << "number_of_red_levels after loop: " << number_of_red_levels << std::endl;
        std::cout << "number of workitems to launch vector size: " << num_workitems_tolaunch.size() << std::endl;
        std::cout << "number of workitems per wg vector size: " << num_workitems_per_wg.size() << std::endl;
        std::cout << "number of local_mem_size vector size: " << local_mem_size.size() << std::endl;
        std::cout << "number of size_limits vector size: " << size_limits.size() << std::endl;
        std::cout << "number of buffer_sizes vector size: " << buffer_sizes.size() << std::endl;
        std::cout << "number of input_even vector size: " << input_even.size() << std::endl;

        for (int i=0; i<number_of_red_levels; i++) {
            std::cout << "Red level:            " << i+1 << std::endl;
            std::cout << "buffer input size:    " << buffer_sizes[i] << std::endl;
            std::cout << "Work items to launch: " << num_workitems_tolaunch[i] << std::endl;
            std::cout << "Work items per wg:    " << num_workitems_per_wg[i] << std::endl;
            std::cout << "Local memory size:    " << local_mem_size[i] << std::endl;
            std::cout << "Size limit:           " << size_limits[i] << std::endl;
            std::cout << "Input Even:           " << std::boolalpha << input_even[i] << std::endl;
            std::cout << "Buffer output size:   " << buffer_sizes[i+1] << std::endl;
            std::cout << std::endl;
        }
#endif
    }
    else {
        std::cout << "ERROR in CloverCL reduction structure: device type is unsupported" << std::endl;
    }

}


void CloverCL::allocateLocalMemoryObjects() {

    if (device_type == CL_DEVICE_TYPE_CPU) {
#ifdef OCL_VERBOSE
        std::cout << "No local memory objects to create as device type is CPU" << std::endl;
#endif
    }
    else if (device_type == CL_DEVICE_TYPE_ACCELERATOR) {
#ifdef OCL_VERBOSE
        std::cout << "No local memory objects to create as device type is ACCELERATOR" << std::endl;
#endif
    }
    else if (device_type == CL_DEVICE_TYPE_GPU) {
        for (int i=0; i<number_of_red_levels; i++) {
            
            min_local_memory_objects.push_back(   cl::Local(local_mem_size[i]*sizeof(cl_double))  );
            vol_local_memory_objects.push_back(   cl::Local(local_mem_size[i]*sizeof(cl_double))  );
            mass_local_memory_objects.push_back(  cl::Local(local_mem_size[i]*sizeof(cl_double))  );
            ie_local_memory_objects.push_back(    cl::Local(local_mem_size[i]*sizeof(cl_double))  );
            ke_local_memory_objects.push_back(    cl::Local(local_mem_size[i]*sizeof(cl_double))  );
            press_local_memory_objects.push_back( cl::Local(local_mem_size[i]*sizeof(cl_double))  );
        }

#ifdef OCL_VERBOSE
        std::cout << "min local memory objects vector size: "     << min_local_memory_objects.size() << std::endl;
        std::cout << "vol local memory objects vector size: "     << vol_local_memory_objects.size() << std::endl;
        std::cout << "mass local memory objects vector size: "    << mass_local_memory_objects.size() << std::endl;
        std::cout << "ie local memory objects vector size: "      << ie_local_memory_objects.size() << std::endl;
        std::cout << "ke local memory objects vector size: "      << ke_local_memory_objects.size() << std::endl;
        std::cout << "press local memory objects vector size: "   << press_local_memory_objects.size() << std::endl;

        for (int i=0; i<number_of_red_levels; i++) {
           std::cout << "reduction level " << i+1 << "min local object size: "   << min_local_memory_objects[i].size_/sizeof(double) << std::endl;
           std::cout << "reduction level " << i+1 << "vol local object size: "   << vol_local_memory_objects[i].size_/sizeof(double) << std::endl;
           std::cout << "reduction level " << i+1 << "mass local object size: "  << mass_local_memory_objects[i].size_/sizeof(double) << std::endl;
           std::cout << "reduction level " << i+1 << "ie local object size: "    << ie_local_memory_objects[i].size_/sizeof(double) << std::endl;
           std::cout << "reduction level " << i+1 << "ke local object size: "    << ke_local_memory_objects[i].size_/sizeof(double) << std::endl;
           std::cout << "reduction level " << i+1 << "press local object size: " << press_local_memory_objects[i].size_/sizeof(double) << std::endl;
        }
#endif
    }
    else {
        std::cout << "ERROR in CloverCL.C local memory object creation: device type not supported " << std::endl;
    }

}

void CloverCL::allocateReductionInterBuffers() {

    cl_int err;

    if ( (device_type == CL_DEVICE_TYPE_CPU) || (device_type==CL_DEVICE_TYPE_ACCELERATOR) ) {

        if ( number_of_red_levels == 1 ) { 
#ifdef OCL_VERBOSE
            std::cout << "No intermediate reduction buffers required for CPU reduction as num_elements too small"  << std::endl;
#endif
        }
        else {
            cpu_min_red_buffer = cl::Buffer(context, CL_MEM_READ_WRITE, device_procs*sizeof(double), NULL, &err);
            cpu_vol_red_buffer = cl::Buffer(context, CL_MEM_READ_WRITE, device_procs*sizeof(double), NULL, &err);
            cpu_mass_red_buffer = cl::Buffer(context, CL_MEM_READ_WRITE, device_procs*sizeof(double), NULL, &err);
            cpu_ie_red_buffer = cl::Buffer(context, CL_MEM_READ_WRITE, device_procs*sizeof(double), NULL, &err);
            cpu_ke_red_buffer = cl::Buffer(context, CL_MEM_READ_WRITE, device_procs*sizeof(double), NULL, &err);
            cpu_press_red_buffer = cl::Buffer(context, CL_MEM_READ_WRITE, device_procs*sizeof(double), NULL, &err);

#ifdef OCL_VERBOSE
            std::cout << "Intermediate reduction buffers on CPU created with size: " << device_procs << std::endl;
#endif
        }

    }
    else if (device_type == CL_DEVICE_TYPE_GPU) {

        for (int i=1; i<=number_of_red_levels-1; i++) {

            min_interBuffers.push_back(  cl::Buffer( context, CL_MEM_READ_WRITE, buffer_sizes[i]*sizeof(double), NULL, &err));
            vol_interBuffers.push_back(  cl::Buffer( context, CL_MEM_READ_WRITE, buffer_sizes[i]*sizeof(double), NULL, &err));
            mass_interBuffers.push_back( cl::Buffer( context, CL_MEM_READ_WRITE, buffer_sizes[i]*sizeof(double), NULL, &err));
            ie_interBuffers.push_back(   cl::Buffer( context, CL_MEM_READ_WRITE, buffer_sizes[i]*sizeof(double), NULL, &err));
            ke_interBuffers.push_back(   cl::Buffer( context, CL_MEM_READ_WRITE, buffer_sizes[i]*sizeof(double), NULL, &err));
            press_interBuffers.push_back(cl::Buffer( context, CL_MEM_READ_WRITE, buffer_sizes[i]*sizeof(double), NULL, &err));

        }

#ifdef OCL_VERBOSE
        size_t size;
        std::cout << "min inter buffers vector size: "   << min_interBuffers.size() << std::endl;
        std::cout << "vol inter buffers vector size: "   << vol_interBuffers.size() << std::endl;
        std::cout << "mass inter buffers vector size: "  << mass_interBuffers.size() << std::endl;
        std::cout << "ie inter buffers vector size: "    << ie_interBuffers.size() << std::endl;
        std::cout << "ke inter buffers vector size: "    << ke_interBuffers.size() << std::endl;
        std::cout << "press inter buffers vector size: " << press_interBuffers.size() << std::endl;

        for (int i=0; i<=number_of_red_levels-2; i++) {
            min_interBuffers[i].getInfo(CL_MEM_SIZE, &size);
            std::cout << "min inter buffers level: "   << i << " buffer elements: " << size/sizeof(double) << std::endl;
            vol_interBuffers[i].getInfo(CL_MEM_SIZE, &size);
            std::cout << "vol inter buffers level: "   << i << " buffer elements: " << size/sizeof(double) << std::endl;
            mass_interBuffers[i].getInfo(CL_MEM_SIZE, &size);
            std::cout << "mass inter buffers level: "  << i << " buffer elements: " << size/sizeof(double) << std::endl;
            ie_interBuffers[i].getInfo(CL_MEM_SIZE, &size);
            std::cout << "ie inter buffers level: "    << i << " buffer elements: " << size/sizeof(double) << std::endl;
            ke_interBuffers[i].getInfo(CL_MEM_SIZE, &size);
            std::cout << "ke inter buffers level: "    << i << " buffer elements: " << size/sizeof(double) << std::endl;
            press_interBuffers[i].getInfo(CL_MEM_SIZE, &size);
            std::cout << "press inter buffers level: " << i << " buffer elements: " << size/sizeof(double) << std::endl;
        }
#endif
    } else {
        std::cout << "ERROR in CloverCL.C allocate inter buffers: device type not supported" << std::endl;
    }

}

void CloverCL::printDeviceInformation() {
    int i, j;
    char* value;
    size_t valueSize;
    cl_uint platformCount;
    cl_platform_id* platforms;
    cl_uint deviceCount;
    cl_device_id* devices;
    cl_uint maxComputeUnits;
  
    // get all platforms
    clGetPlatformIDs(0, NULL, &platformCount);
    platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);
    clGetPlatformIDs(platformCount, platforms, NULL);
  
    for (i = 0; i < platformCount; i++) {
  
        // get all devices
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
        devices = (cl_device_id*) malloc(sizeof(cl_device_id) * deviceCount);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);
  
        // for each device print critical attributes
        for (j = 0; j < deviceCount; j++) {
  
            // print device name
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, valueSize, value, NULL);
            printf("%d. Device: %s\n", j+1, value);
            free(value);
  
            // print hardware device version
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, valueSize, value, NULL);
            printf(" %d.%d Hardware version: %s\n", j+1, 1, value);
            free(value);
  
            // print software driver version
            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, valueSize, value, NULL);
            printf(" %d.%d Software version: %s\n", j+1, 2, value);
            free(value);
  
            // print c version supported by compiler for device
            clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
            printf(" %d.%d OpenCL C version: %s\n", j+1, 3, value);
            free(value);
  
            // print parallel compute units
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS,
                    sizeof(maxComputeUnits), &maxComputeUnits, NULL);
            printf(" %d.%d Parallel compute units: %d\n", j+1, 4, maxComputeUnits);
    
            //print max number of work items 
	        int mwgs;
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(mwgs), &mwgs, NULL);
            printf(" %d.%d Max num work items: %d\n", j+1, 5, mwgs);

            //print global memory size 
            cl_ulong global_mem;
            clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE,
                    sizeof(cl_ulong), &global_mem, NULL);
            printf(" %d.%d Global memory size: %lu\n", j+1, 6, global_mem);

            //print clock speed
            cl_uint clock_speed;
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_CLOCK_FREQUENCY,
                    sizeof(cl_uint), &clock_speed, NULL);
            printf(" %d.%d Clock speed: %u\n", j+1, 7, clock_speed);
        }
    }
}
      
void CloverCL::initPlatform(
        std::string name) 
{
    cl_int err;
    std::vector< cl::Platform > platformList;

    /*
     * Lowercase the name provided
     */
    std::transform(name.begin(), name.end(), name.begin(), ::tolower);

    cl::Platform::get(&platformList);

    checkErr(platformList.size() != 0 ? CL_SUCCESS : -1, "cl::Platform::get");
    checkErr(0 < platformList.size() ? CL_SUCCESS : -1, "0 < number of platforms");


    std::string platformVendor;
    for (int i = 0; i < platformList.size(); i++) {
        platformList[i].getInfo((cl_platform_info)CL_PLATFORM_VENDOR, &platformVendor);
#ifdef OCL_VERBOSE
        std::cout << "Found platform = " << platformVendor << std::endl;
#endif
        std::transform(platformVendor.begin(), platformVendor.end(), platformVendor.begin(), ::tolower);
        if(platformVendor.find(name) != std::string::npos) {
            platform = platformList[i];
            break;
        }
    }
}

void CloverCL::initContext(
        std::string preferred_type)
{
    cl_int err;

    /*
     * Lowercase the type provided
     */
    std::transform(preferred_type.begin(), preferred_type.end(), preferred_type.begin(), ::tolower);

    cl_context_properties cprops[3] =
        { CL_CONTEXT_PLATFORM,
          (cl_context_properties)(platform)(),
          0
        };

    device_type = CL_DEVICE_TYPE_DEFAULT;

    if (preferred_type == "gpu") {
        device_type = CL_DEVICE_TYPE_GPU;
    } else if(preferred_type == "cpu") {
        device_type = CL_DEVICE_TYPE_CPU;
    } else if(preferred_type == "phi") {
        device_type = CL_DEVICE_TYPE_ACCELERATOR;
    }

#ifdef OCL_VERBOSE
    if (preferred_type == "gpu") {
        std::cout << "Device Type selected: GPU" << std::endl;
    } else if(preferred_type == "cpu") {
        std::cout << "Device Type selected: CPU" << std::endl;
    } else if(preferred_type == "phi") {
        std::cout << "Device Type selected: PHI Accelerator" << std::endl;
    }
#endif

    /*
     * Get the device context.
     */
    try {

        context = cl::Context ( device_type, cprops, NULL, NULL, &err);

    } catch (cl::Error err) {
        reportError(err, "Creating context");
    }
}

void CloverCL::initDevice(int id)
{
    std::vector<cl::Device> devices;
    devices = context.getInfo<CL_CONTEXT_DEVICES>();

    checkErr(devices.size() > 0 ? CL_SUCCESS : -1, "devices.size() > 0");

    device = devices[id];
}

void CloverCL::initCommandQueue()
{
    cl_int err;

#if PROFILE_OCL_KERNELS
    queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);

    outoforder_queue = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE|CL_QUEUE_PROFILING_ENABLE, &err);
#else
    queue = cl::CommandQueue(context, device, 0, &err);

    outoforder_queue = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
#endif
}

void CloverCL::createBuffers(int x_max, int y_max, int num_states)
{
    cl_int err;

    density0_buffer = cl::Buffer( context, CL_MEM_READ_WRITE, (x_max+4)*(y_max+4)*sizeof(double), NULL, &err);
    density1_buffer = cl::Buffer( context, CL_MEM_READ_WRITE, (x_max+4)*(y_max+4)*sizeof(double), NULL, &err);
    energy0_buffer = cl::Buffer( context, CL_MEM_READ_WRITE, (x_max+4)*(y_max+4)*sizeof(double), NULL, &err);
    energy1_buffer = cl::Buffer( context, CL_MEM_READ_WRITE, (x_max+4)*(y_max+4)*sizeof(double), NULL, &err);
    pressure_buffer = cl::Buffer( context, CL_MEM_READ_WRITE, (x_max+4)*(y_max+4)*sizeof(double), NULL, &err);
    viscosity_buffer = cl::Buffer( context, CL_MEM_READ_WRITE, (x_max+4)*(y_max+4)*sizeof(double), NULL, &err);
    soundspeed_buffer = cl::Buffer( context, CL_MEM_READ_WRITE, (x_max+4)*(y_max+4)*sizeof(double), NULL, &err);
    xvel0_buffer = cl::Buffer( context, CL_MEM_READ_WRITE, (x_max+5)*(y_max+5)*sizeof(double), NULL, &err);
    yvel0_buffer = cl::Buffer( context, CL_MEM_READ_WRITE, (x_max+5)*(y_max+5)*sizeof(double), NULL, &err);
    xvel1_buffer = cl::Buffer( context, CL_MEM_READ_WRITE, (x_max+5)*(y_max+5)*sizeof(double), NULL, &err);
    yvel1_buffer = cl::Buffer( context, CL_MEM_READ_WRITE, (x_max+5)*(y_max+5)*sizeof(double), NULL, &err);
    vol_flux_x_buffer = cl::Buffer( context, CL_MEM_READ_WRITE, (x_max+5)*(y_max+4)*sizeof(double), NULL, &err);
    vol_flux_y_buffer = cl::Buffer( context, CL_MEM_READ_WRITE, (x_max+4)*(y_max+5)*sizeof(double), NULL, &err);
    mass_flux_x_buffer = cl::Buffer( context, CL_MEM_READ_WRITE, (x_max+5)*(y_max+4)*sizeof(double), NULL, &err);
    mass_flux_y_buffer = cl::Buffer( context, CL_MEM_READ_WRITE, (x_max+4)*(y_max+5)*sizeof(double), NULL, &err);

    cellx_buffer = cl::Buffer( context, CL_MEM_READ_WRITE, (x_max+4)*sizeof(double), NULL, &err);
    celly_buffer = cl::Buffer( context, CL_MEM_READ_WRITE, (y_max+4)*sizeof(double), NULL, &err);
    vertexx_buffer = cl::Buffer( context, CL_MEM_READ_WRITE, (x_max+5)*sizeof(double), NULL, &err);
    vertexy_buffer = cl::Buffer( context, CL_MEM_READ_WRITE, (y_max+5)*sizeof(double), NULL, &err);
    celldx_buffer = cl::Buffer( context, CL_MEM_READ_ONLY, (x_max+4)*(y_max+4)*sizeof(double), NULL, &err);
    celldy_buffer = cl::Buffer( context, CL_MEM_READ_ONLY, (x_max+4)*(y_max+4)*sizeof(double), NULL, &err);
    vertexdx_buffer = cl::Buffer( context, CL_MEM_READ_WRITE, (x_max+5)*sizeof(double), NULL, &err);
    vertexdy_buffer = cl::Buffer( context, CL_MEM_READ_WRITE, (y_max+5)*sizeof(double), NULL, &err);
    volume_buffer = cl::Buffer( context, CL_MEM_READ_WRITE, (x_max+4)*(y_max+4)*sizeof(double), NULL, &err);
    xarea_buffer = cl::Buffer( context, CL_MEM_READ_WRITE, (x_max+5)*(y_max+4)*sizeof(double), NULL, &err);
    yarea_buffer = cl::Buffer( context, CL_MEM_READ_ONLY, (x_max+4)*(y_max+5)*sizeof(double), NULL, &err);

    work_array1_buffer = cl::Buffer(context, CL_MEM_READ_WRITE, (x_max+5)*(y_max+5)*sizeof(double), NULL, &err);
    work_array2_buffer = cl::Buffer(context, CL_MEM_READ_WRITE, (x_max+5)*(y_max+5)*sizeof(double), NULL, &err);
    work_array3_buffer = cl::Buffer(context, CL_MEM_READ_WRITE, (x_max+5)*(y_max+5)*sizeof(double), NULL, &err);
    work_array4_buffer = cl::Buffer(context, CL_MEM_READ_WRITE, (x_max+5)*(y_max+5)*sizeof(double), NULL, &err);
    work_array5_buffer = cl::Buffer(context, CL_MEM_READ_WRITE, (x_max+5)*(y_max+5)*sizeof(double), NULL, &err);
    work_array6_buffer = cl::Buffer(context, CL_MEM_READ_WRITE, (x_max+5)*(y_max+5)*sizeof(double), NULL, &err);
    work_array7_buffer = cl::Buffer(context, CL_MEM_READ_WRITE, (x_max+5)*(y_max+5)*sizeof(double), NULL, &err);

    dt_min_val_buffer = cl::Buffer( context, CL_MEM_READ_WRITE, sizeof(double), NULL, &err);
    vol_sum_val_buffer = cl::Buffer( context, CL_MEM_READ_WRITE, sizeof(double), NULL, &err);
    mass_sum_val_buffer = cl::Buffer( context, CL_MEM_READ_WRITE, sizeof(double), NULL, &err);
    ie_sum_val_buffer = cl::Buffer( context, CL_MEM_READ_WRITE, sizeof(double), NULL, &err);
    ke_sum_val_buffer = cl::Buffer( context, CL_MEM_READ_WRITE, sizeof(double), NULL, &err);
    press_sum_val_buffer = cl::Buffer( context, CL_MEM_READ_WRITE, sizeof(double), NULL, &err);

    state_density_buffer = cl::Buffer( context, CL_MEM_READ_WRITE, num_states*sizeof(double), NULL); 
    state_energy_buffer = cl::Buffer( context, CL_MEM_READ_WRITE, num_states*sizeof(double), NULL); 
    state_xvel_buffer = cl::Buffer( context, CL_MEM_READ_WRITE, num_states*sizeof(double), NULL); 
    state_yvel_buffer = cl::Buffer( context, CL_MEM_READ_WRITE, num_states*sizeof(double), NULL); 
    state_xmin_buffer = cl::Buffer( context, CL_MEM_READ_WRITE, num_states*sizeof(double), NULL); 
    state_xmax_buffer = cl::Buffer( context, CL_MEM_READ_WRITE, num_states*sizeof(double), NULL); 
    state_ymin_buffer = cl::Buffer( context, CL_MEM_READ_WRITE, num_states*sizeof(double), NULL); 
    state_ymax_buffer = cl::Buffer( context, CL_MEM_READ_WRITE, num_states*sizeof(double), NULL); 
    state_radius_buffer = cl::Buffer( context, CL_MEM_READ_WRITE, num_states*sizeof(double), NULL); 
    state_geometry_buffer = cl::Buffer( context, CL_MEM_READ_WRITE, num_states*sizeof(int), NULL); 

    top_send_buffer = cl::Buffer( context, CL_MEM_READ_WRITE, (x_max+5)*2*sizeof(double), NULL, &err);
    top_recv_buffer = cl::Buffer( context, CL_MEM_READ_WRITE, (x_max+5)*2*sizeof(double), NULL, &err);
    bottom_send_buffer = cl::Buffer( context, CL_MEM_READ_WRITE, (x_max+5)*2*sizeof(double), NULL, &err);
    bottom_recv_buffer = cl::Buffer( context, CL_MEM_READ_WRITE, (x_max+5)*2*sizeof(double), NULL, &err);
    left_send_buffer = cl::Buffer( context, CL_MEM_READ_WRITE, (y_max+5)*2*sizeof(double), NULL, &err);
    left_recv_buffer = cl::Buffer( context, CL_MEM_READ_WRITE, (y_max+5)*2*sizeof(double), NULL, &err);
    right_send_buffer = cl::Buffer( context, CL_MEM_READ_WRITE, (y_max+5)*2*sizeof(double), NULL, &err);
    right_recv_buffer = cl::Buffer( context, CL_MEM_READ_WRITE, (y_max+5)*2*sizeof(double), NULL, &err);
}


void CloverCL::initialiseKernelArgs(int x_min, int x_max, int y_min, int y_max,
                                    double g_small, double g_big, double dtmin,
                                    double dtc_safe, double dtu_safe, 
                                    double dtv_safe, double dtdiv_safe)
{
    try {
        viscosity_knl.setArg(0,  celldx_buffer);
        viscosity_knl.setArg(1,  celldy_buffer);
        viscosity_knl.setArg(2,  density0_buffer);
        viscosity_knl.setArg(3,  pressure_buffer);
        viscosity_knl.setArg(4,  viscosity_buffer);
        viscosity_knl.setArg(5,  xvel0_buffer);
        viscosity_knl.setArg(6, yvel0_buffer);

        accelerate_knl.setArg(1, xarea_buffer);
        accelerate_knl.setArg(2, yarea_buffer);
        accelerate_knl.setArg(3, volume_buffer);
        accelerate_knl.setArg(4, density0_buffer);
        accelerate_knl.setArg(5, pressure_buffer);
        accelerate_knl.setArg(6, viscosity_buffer);
        accelerate_knl.setArg(7, xvel0_buffer);
        accelerate_knl.setArg(8, yvel0_buffer);
        accelerate_knl.setArg(9, xvel1_buffer);
        accelerate_knl.setArg(10, yvel1_buffer);

        field_summary_knl.setArg(0, volume_buffer);
        field_summary_knl.setArg(1, density0_buffer);
        field_summary_knl.setArg(2, energy0_buffer);
        field_summary_knl.setArg(3, pressure_buffer);
        field_summary_knl.setArg(4, xvel0_buffer);
        field_summary_knl.setArg(5, yvel0_buffer);
        field_summary_knl.setArg(6,  work_array1_buffer);
        field_summary_knl.setArg(7,  work_array2_buffer);
        field_summary_knl.setArg(8,  work_array3_buffer);
        field_summary_knl.setArg(9,  work_array4_buffer);
        field_summary_knl.setArg(10, work_array5_buffer);

        reset_field_knl.setArg(0, density0_buffer);
        reset_field_knl.setArg(1, density1_buffer);
        reset_field_knl.setArg(2, energy0_buffer);
        reset_field_knl.setArg(3, energy1_buffer);
        reset_field_knl.setArg(4, xvel0_buffer);
        reset_field_knl.setArg(5, xvel1_buffer);
        reset_field_knl.setArg(6, yvel0_buffer);
        reset_field_knl.setArg(7, yvel1_buffer);

        revert_knl.setArg(0, density0_buffer);
        revert_knl.setArg(1, density1_buffer);
        revert_knl.setArg(2, energy0_buffer);
        revert_knl.setArg(3, energy1_buffer);

        flux_calc_knl.setArg(1, xarea_buffer);
        flux_calc_knl.setArg(2, xvel0_buffer);
        flux_calc_knl.setArg(3, xvel1_buffer);
        flux_calc_knl.setArg(4, vol_flux_x_buffer);
        flux_calc_knl.setArg(5, yarea_buffer);
        flux_calc_knl.setArg(6, yvel0_buffer);
        flux_calc_knl.setArg(7, yvel1_buffer);
        flux_calc_knl.setArg(8, vol_flux_y_buffer);

        initialise_chunk_cell_x_knl.setArg(1, vertexx_buffer);
        initialise_chunk_cell_x_knl.setArg(2, cellx_buffer);
        initialise_chunk_cell_x_knl.setArg(3, celldx_buffer);

        initialise_chunk_cell_y_knl.setArg(1, vertexy_buffer);
        initialise_chunk_cell_y_knl.setArg(2, celly_buffer);
        initialise_chunk_cell_y_knl.setArg(3, celldy_buffer);

        initialise_chunk_vertex_x_knl.setArg(2, vertexx_buffer);
        initialise_chunk_vertex_x_knl.setArg(3, vertexdx_buffer);

        initialise_chunk_vertex_y_knl.setArg(2, vertexy_buffer);
        initialise_chunk_vertex_y_knl.setArg(3, vertexdy_buffer);

        initialise_chunk_volume_area_knl.setArg(2, volume_buffer);
        initialise_chunk_volume_area_knl.setArg(3, celldx_buffer);
        initialise_chunk_volume_area_knl.setArg(4, celldy_buffer);
        initialise_chunk_volume_area_knl.setArg(5, xarea_buffer);
        initialise_chunk_volume_area_knl.setArg(6, yarea_buffer);

        generate_chunk_knl.setArg(0, vertexx_buffer);
        generate_chunk_knl.setArg(1, vertexy_buffer);
        generate_chunk_knl.setArg(2, cellx_buffer);
        generate_chunk_knl.setArg(3, celly_buffer);
        generate_chunk_knl.setArg(4, density0_buffer);
        generate_chunk_knl.setArg(5, energy0_buffer);
        generate_chunk_knl.setArg(6, xvel0_buffer);
        generate_chunk_knl.setArg(7, yvel0_buffer);
        generate_chunk_knl.setArg(9, state_density_buffer);
        generate_chunk_knl.setArg(10, state_energy_buffer);
        generate_chunk_knl.setArg(11, state_xvel_buffer);
        generate_chunk_knl.setArg(12, state_yvel_buffer);
        generate_chunk_knl.setArg(13, state_xmin_buffer);
        generate_chunk_knl.setArg(14, state_xmax_buffer);
        generate_chunk_knl.setArg(15, state_ymin_buffer);
        generate_chunk_knl.setArg(16, state_ymax_buffer);
        generate_chunk_knl.setArg(17, state_radius_buffer);
        generate_chunk_knl.setArg(18, state_geometry_buffer);

        pdv_correct_knl.setArg(1, xarea_buffer);
        pdv_correct_knl.setArg(2, yarea_buffer);
        pdv_correct_knl.setArg(3, volume_buffer);
        pdv_correct_knl.setArg(4, density0_buffer);
        pdv_correct_knl.setArg(5, density1_buffer);
        pdv_correct_knl.setArg(6, energy0_buffer);
        pdv_correct_knl.setArg(7, energy1_buffer);
        pdv_correct_knl.setArg(8, pressure_buffer);
        pdv_correct_knl.setArg(9, viscosity_buffer);
        pdv_correct_knl.setArg(10, xvel0_buffer);
        pdv_correct_knl.setArg(11, xvel1_buffer);
        pdv_correct_knl.setArg(12, yvel0_buffer);
        pdv_correct_knl.setArg(13, yvel1_buffer);

        pdv_predict_knl.setArg(1, xarea_buffer);
        pdv_predict_knl.setArg(2, yarea_buffer);
        pdv_predict_knl.setArg(3, volume_buffer);
        pdv_predict_knl.setArg(4, density0_buffer);
        pdv_predict_knl.setArg(5, density1_buffer);
        pdv_predict_knl.setArg(6, energy0_buffer);
        pdv_predict_knl.setArg(7, energy1_buffer);
        pdv_predict_knl.setArg(8, pressure_buffer);
        pdv_predict_knl.setArg(9, viscosity_buffer);
        pdv_predict_knl.setArg(10, xvel0_buffer);
        pdv_predict_knl.setArg(11, xvel1_buffer);
        pdv_predict_knl.setArg(12, yvel0_buffer);
        pdv_predict_knl.setArg(13, yvel1_buffer);

        dt_calc_knl.setArg(0, g_small);
        dt_calc_knl.setArg(1, g_big);
        dt_calc_knl.setArg(2, dtmin);
        dt_calc_knl.setArg(3, dtc_safe);
        dt_calc_knl.setArg(4, dtu_safe);
        dt_calc_knl.setArg(5, dtv_safe);
        dt_calc_knl.setArg(6, dtdiv_safe);
        dt_calc_knl.setArg(7, xarea_buffer);
        dt_calc_knl.setArg(8, yarea_buffer);
        dt_calc_knl.setArg(9, cellx_buffer);
        dt_calc_knl.setArg(10, celly_buffer);
        dt_calc_knl.setArg(11, celldx_buffer);
        dt_calc_knl.setArg(12, celldy_buffer);
        dt_calc_knl.setArg(13, volume_buffer);
        dt_calc_knl.setArg(14, density0_buffer);
        dt_calc_knl.setArg(15, energy0_buffer);
        dt_calc_knl.setArg(16, pressure_buffer);
        dt_calc_knl.setArg(17, viscosity_buffer);
        dt_calc_knl.setArg(18, soundspeed_buffer);
        dt_calc_knl.setArg(19, xvel0_buffer);
        dt_calc_knl.setArg(20, yvel0_buffer);
        dt_calc_knl.setArg(21, work_array1_buffer);

        ideal_gas_predict_knl.setArg(0, density1_buffer);
        ideal_gas_predict_knl.setArg(1, energy1_buffer);
        ideal_gas_predict_knl.setArg(2, pressure_buffer);
        ideal_gas_predict_knl.setArg(3, soundspeed_buffer);

        ideal_gas_NO_predict_knl.setArg(0, density0_buffer);
        ideal_gas_NO_predict_knl.setArg(1, energy0_buffer);
        ideal_gas_NO_predict_knl.setArg(2, pressure_buffer);
        ideal_gas_NO_predict_knl.setArg(3, soundspeed_buffer);

        advec_cell_xdir_sec1_s1_knl.setArg(0, volume_buffer);
        advec_cell_xdir_sec1_s1_knl.setArg(1, vol_flux_x_buffer);
        advec_cell_xdir_sec1_s1_knl.setArg(2, vol_flux_y_buffer);
        advec_cell_xdir_sec1_s1_knl.setArg(3, work_array1_buffer);
        advec_cell_xdir_sec1_s1_knl.setArg(4, work_array2_buffer);

        advec_cell_xdir_sec1_s2_knl.setArg(0, volume_buffer);
        advec_cell_xdir_sec1_s2_knl.setArg(1, vol_flux_x_buffer);
        advec_cell_xdir_sec1_s2_knl.setArg(2, work_array1_buffer);
        advec_cell_xdir_sec1_s2_knl.setArg(3, work_array2_buffer);

        advec_cell_xdir_sec2_knl.setArg(0, vertexdx_buffer);
        advec_cell_xdir_sec2_knl.setArg(1, density1_buffer);
        advec_cell_xdir_sec2_knl.setArg(2, energy1_buffer);
        advec_cell_xdir_sec2_knl.setArg(3, mass_flux_x_buffer);
        advec_cell_xdir_sec2_knl.setArg(4, vol_flux_x_buffer);
        advec_cell_xdir_sec2_knl.setArg(5, work_array1_buffer);
        advec_cell_xdir_sec2_knl.setArg(6, work_array7_buffer);

        advec_cell_xdir_sec3_knl.setArg(0, density1_buffer);
        advec_cell_xdir_sec3_knl.setArg(1, energy1_buffer);
        advec_cell_xdir_sec3_knl.setArg(2, mass_flux_x_buffer);
        advec_cell_xdir_sec3_knl.setArg(3, vol_flux_x_buffer);
        advec_cell_xdir_sec3_knl.setArg(4, work_array1_buffer);
        advec_cell_xdir_sec3_knl.setArg(5, work_array3_buffer);
        advec_cell_xdir_sec3_knl.setArg(6, work_array4_buffer);
        advec_cell_xdir_sec3_knl.setArg(7, work_array5_buffer);
        advec_cell_xdir_sec3_knl.setArg(8, work_array6_buffer);
        advec_cell_xdir_sec3_knl.setArg(9, work_array7_buffer);

        advec_cell_ydir_sec1_s1_knl.setArg(0, volume_buffer);
        advec_cell_ydir_sec1_s1_knl.setArg(1, vol_flux_x_buffer);
        advec_cell_ydir_sec1_s1_knl.setArg(2, vol_flux_y_buffer);
        advec_cell_ydir_sec1_s1_knl.setArg(3, work_array1_buffer);
        advec_cell_ydir_sec1_s1_knl.setArg(4, work_array2_buffer);

        advec_cell_ydir_sec1_s2_knl.setArg(0, volume_buffer);
        advec_cell_ydir_sec1_s2_knl.setArg(1, vol_flux_y_buffer);
        advec_cell_ydir_sec1_s2_knl.setArg(2, work_array1_buffer);
        advec_cell_ydir_sec1_s2_knl.setArg(3, work_array2_buffer);

        advec_cell_ydir_sec2_knl.setArg(0, vertexdy_buffer);
        advec_cell_ydir_sec2_knl.setArg(1, density1_buffer);
        advec_cell_ydir_sec2_knl.setArg(2, energy1_buffer);
        advec_cell_ydir_sec2_knl.setArg(3, mass_flux_y_buffer);
        advec_cell_ydir_sec2_knl.setArg(4, vol_flux_y_buffer);
        advec_cell_ydir_sec2_knl.setArg(5, work_array1_buffer);
        advec_cell_ydir_sec2_knl.setArg(6, work_array7_buffer);

        advec_cell_ydir_sec3_knl.setArg(0, density1_buffer);
        advec_cell_ydir_sec3_knl.setArg(1, energy1_buffer);
        advec_cell_ydir_sec3_knl.setArg(2, mass_flux_y_buffer);
        advec_cell_ydir_sec3_knl.setArg(3, vol_flux_y_buffer);
        advec_cell_ydir_sec3_knl.setArg(4, work_array1_buffer);
        advec_cell_ydir_sec3_knl.setArg(5, work_array3_buffer);
        advec_cell_ydir_sec3_knl.setArg(6, work_array4_buffer);
        advec_cell_ydir_sec3_knl.setArg(7, work_array5_buffer);
        advec_cell_ydir_sec3_knl.setArg(8, work_array6_buffer);
        advec_cell_ydir_sec3_knl.setArg(9, work_array7_buffer);

        advec_mom_vol_knl.setArg(0, volume_buffer);
        advec_mom_vol_knl.setArg(1, vol_flux_x_buffer);
        advec_mom_vol_knl.setArg(2, vol_flux_y_buffer);
        advec_mom_vol_knl.setArg(3, work_array6_buffer);
        advec_mom_vol_knl.setArg(4, work_array7_buffer);

        advec_mom_node_x_knl.setArg(0, CloverCL::mass_flux_x_buffer);
        advec_mom_node_x_knl.setArg(1, work_array1_buffer);
        advec_mom_node_x_knl.setArg(2, density1_buffer);
        advec_mom_node_x_knl.setArg(3, work_array7_buffer);
        advec_mom_node_x_knl.setArg(4, work_array2_buffer);

        advec_mom_node_mass_pre_x_knl.setArg(0, work_array3_buffer);
        advec_mom_node_mass_pre_x_knl.setArg(1, work_array2_buffer);
        advec_mom_node_mass_pre_x_knl.setArg(2, work_array1_buffer);

        advec_mom_node_y_knl.setArg(0, mass_flux_y_buffer);
        advec_mom_node_y_knl.setArg(1, work_array1_buffer);
        advec_mom_node_y_knl.setArg(2, work_array2_buffer);
        advec_mom_node_y_knl.setArg(3, density1_buffer);
        advec_mom_node_y_knl.setArg(4, work_array7_buffer);

        advec_mom_node_mass_pre_y_knl.setArg(0, work_array3_buffer);
        advec_mom_node_mass_pre_y_knl.setArg(1, work_array2_buffer);
        advec_mom_node_mass_pre_y_knl.setArg(2, work_array1_buffer);

        advec_mom_flux_x_vec1_knl.setArg(0, work_array1_buffer);
        advec_mom_flux_x_vec1_knl.setArg(1, work_array3_buffer);
        advec_mom_flux_x_vec1_knl.setArg(3, work_array4_buffer);
        advec_mom_flux_x_vec1_knl.setArg(4, work_array5_buffer);
        advec_mom_flux_x_vec1_knl.setArg(5, celldx_buffer);

        advec_mom_flux_x_vecnot1_knl.setArg(0, work_array1_buffer);
        advec_mom_flux_x_vecnot1_knl.setArg(1, work_array3_buffer);
        advec_mom_flux_x_vecnot1_knl.setArg(3, work_array4_buffer);
        advec_mom_flux_x_vecnot1_knl.setArg(4, work_array5_buffer);
        advec_mom_flux_x_vecnot1_knl.setArg(5, celldx_buffer);

        advec_mom_flux_y_vec1_knl.setArg(0, work_array1_buffer);
        advec_mom_flux_y_vec1_knl.setArg(1, work_array3_buffer);
        advec_mom_flux_y_vec1_knl.setArg(3, work_array4_buffer);
        advec_mom_flux_y_vec1_knl.setArg(4, work_array5_buffer);
        advec_mom_flux_y_vec1_knl.setArg(5, celldy_buffer);

        advec_mom_flux_y_vecnot1_knl.setArg(0, work_array1_buffer);
        advec_mom_flux_y_vecnot1_knl.setArg(1, work_array3_buffer);
        advec_mom_flux_y_vecnot1_knl.setArg(3, work_array4_buffer);
        advec_mom_flux_y_vecnot1_knl.setArg(4, work_array5_buffer);
        advec_mom_flux_y_vecnot1_knl.setArg(5, celldy_buffer);

        advec_mom_vel_x_knl.setArg(0, work_array2_buffer);
        advec_mom_vel_x_knl.setArg(1, work_array3_buffer);
        advec_mom_vel_x_knl.setArg(2, work_array5_buffer);

        advec_mom_vel_y_knl.setArg(0, work_array2_buffer);
        advec_mom_vel_y_knl.setArg(1, work_array3_buffer);
        advec_mom_vel_y_knl.setArg(2, work_array5_buffer);

    } catch (cl::Error err) {
        CloverCL::reportError(err, "Setting Kernel Args in CloverCL.C");
    }

}

void CloverCL::loadProgram(int xmin, int xmax, int ymin, int ymax)
{
    cl_int err;

    std::vector<cl::Device> devices;
    
    cl::Program::Sources source;

    devices.push_back(device);

    std::fstream sourceFile;
    std::string line;
    std::string sourceCode;
    std::stringstream ss;
    ss.str("");
    #define ADD_SOURCE(file) \
        sourceFile.open(file, std::ifstream::in); \
	    while ( sourceFile.good() ) \
	    { \
	        getline (sourceFile, line); \
	        ss << line << std::endl; \
	    } \
	    sourceFile.close();
    ADD_SOURCE("./viscosity_knl.cl");
    ADD_SOURCE("./ideal_gas_knl.cl");
    ADD_SOURCE("./flux_calc_knl.cl");
    ADD_SOURCE("./accelerate_knl.cl");
    ADD_SOURCE("./advec_cell_knl.cl");
    ADD_SOURCE("./advec_mom_knl.cl");
    ADD_SOURCE("./calc_dt_knl.cl");
    ADD_SOURCE("./pdv_knl.cl");
    ADD_SOURCE("./reset_field_knl.cl");
    ADD_SOURCE("./revert_knl.cl");
    ADD_SOURCE("./generate_chunk_knl.cl");
    ADD_SOURCE("./initialise_chunk_knl.cl");
    ADD_SOURCE("./field_summary_knl.cl");
    ADD_SOURCE("./update_halo_knl.cl");
    //ADD_SOURCE("./read_comm_buffers_knl.cl");
    //ADD_SOURCE("./write_comm_buffers_knl.cl");
    ADD_SOURCE("./min_reduction_knl.cl");
    ADD_SOURCE("./sum_reduction_knl.cl");
    ADD_SOURCE("./pack_comms_buffers_knl.cl");
    ADD_SOURCE("./unpack_comms_buffers_knl.cl");

    sourceCode = ss.str();
    cl::Program::Sources sources;
    sources = cl::Program::Sources(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));

    /* Build the program */
    #define BUILD_LOG() \
        std::string build_log; \
        program.getBuildInfo(devices[0], CL_PROGRAM_BUILD_LOG, &build_log);     \
        std::cout << "Build Log:" << std::endl; \
        std::cout << build_log << std::endl; \

    cl_int prog_err;
    char buildOptions [350];

    try {
        program = cl::Program(context, sources, &prog_err);
	    checkErr(prog_err, "Program object creation");

        int workgroup_size = CloverCL::local_wg_x_calcdt_fieldsumm * CloverCL::local_wg_y_calcdt_fieldsumm;

        if (device_type == CL_DEVICE_TYPE_GPU) {

#ifdef OCL_VERBOSE
            std::cout << "Executing GPU specific kernels " << std::endl;
#endif
            sprintf(buildOptions, 
                    "-DXMIN=%u -DXMINPLUSONE=%u -DXMAX=%u -DYMIN=%u -DYMINPLUSONE=%u -DYMINPLUSTWO=%u "
                    "-DYMAX=%u -DXMAXPLUSONE=%u -DXMAXPLUSTWO=%u -DXMAXPLUSTHREE=%u -DXMAXPLUSFOUR=%u "
                    "-DXMAXPLUSFIVE=%u -DYMAXPLUSONE=%u -DYMAXPLUSTWO=%u -DYMAXPLUSTHREE=%u -DWORKGROUP_SIZE=%u "
                    "-DWORKGROUP_SIZE_DIVTWO=%u -DGPU_REDUCTION -cl-strict-aliasing", 
                    xmin, xmin+1, xmax, ymin, ymin+1, ymin+2, ymax, xmax+1, xmax+2, xmax+3, xmax+4, xmax+5, 
                    ymax+1, ymax+2, ymax+3, workgroup_size, workgroup_size/2
                   );
        } else {

#ifdef OCL_VERBOSE
            std::cout << "Executing CPU specific kernels " << std::endl;
#endif
            sprintf(buildOptions, 
                    "-DXMIN=%u -DXMINPLUSONE=%u -DXMAX=%u -DYMIN=%u -DYMINPLUSONE=%u -DYMINPLUSTWO=%u "
                    "-DYMAX=%u -DXMAXPLUSONE=%u -DXMAXPLUSTWO=%u -DXMAXPLUSTHREE=%u -DXMAXPLUSFOUR=%u "
                    "-DXMAXPLUSFIVE=%u -DYMAXPLUSONE=%u -DYMAXPLUSTWO=%u -DYMAXPLUSTHREE=%u -DWORKGROUP_SIZE=%u "
                    "-DWORKGROUP_SIZE_DIVTWO=%u", 
                    xmin, xmin+1, xmax, ymin, ymin+1, ymin+2, ymax, xmax+1, xmax+2, xmax+3, xmax+4, xmax+5, 
                    ymax+1, ymax+2, ymax+3, workgroup_size, workgroup_size/2
                   );
        }

        err = program.build(devices, buildOptions); 

    } catch (cl::Error err) {
        std::cerr
            << "[ERROR]: " 
            << err.what()
            << "("
            << errToString(err.err())
            << ")"
            << std::endl;
        BUILD_LOG();
    }


    /*
     * Set up the kernels here!
     */
    try {
         ideal_gas_predict_knl = cl::Kernel(program, "ideal_gas_ocl_kernel", &err);
    } catch (cl::Error err) {
        reportError(err, "ideal_gas_predict_kernel");
    }

    try {
         ideal_gas_NO_predict_knl = cl::Kernel(program, "ideal_gas_ocl_kernel", &err);
    } catch (cl::Error err) {
        reportError(err, "ideal_gas_NO_predict_kernel");
    }

    try {
        viscosity_knl = cl::Kernel(program, "viscosity_ocl_kernel", &err);
    } catch (cl::Error err) {
        reportError(err, "viscosity_ocl_kernel");
    }

    try {
        flux_calc_knl = cl::Kernel(program, "flux_calc_ocl_kernel", &err);
    } catch (cl::Error err) {
        reportError(err, "flux_calc_ocl_kernel");
    }

    try {
        accelerate_knl = cl::Kernel(program, "accelerate_ocl_kernel", &err);
    } catch (cl::Error err) {
        reportError(err, "accelerate_ocl_kernel");
    }

    try {
        advec_cell_xdir_sec1_s1_knl = cl::Kernel(program, "advec_cell_xdir_section1_sweep1_kernel", &err);
    } catch (cl::Error err) {
        reportError(err, "advec_cell_xdir_section1_sweep1_kernel");
    }

    try {
        advec_cell_xdir_sec1_s2_knl = cl::Kernel(program, "advec_cell_xdir_section1_sweep2_kernel", &err);
    } catch (cl::Error err) {
        reportError(err, "advec_cell_xdir_section1_sweep2_kernel");
    }

    try {
        advec_cell_xdir_sec2_knl = cl::Kernel(program, "advec_cell_xdir_section2_kernel", &err);
    } catch (cl::Error err) {
        reportError(err, "advec_cell_xdir_section1_sweep2_kernel");
    }

    try {
        advec_cell_xdir_sec3_knl = cl::Kernel(program, "advec_cell_xdir_section3_kernel", &err);
    } catch (cl::Error err) {
        reportError(err, "advec_cell_xdir_section3_kernel");
    }

    try {
        advec_cell_ydir_sec1_s1_knl = cl::Kernel(program, "advec_cell_ydir_section1_sweep1_kernel", &err);
    } catch (cl::Error err) {
        reportError(err, "advec_cell_ydir_section1_sweep1_kernel");
    }

    try {
        advec_cell_ydir_sec1_s2_knl = cl::Kernel(program, "advec_cell_ydir_section1_sweep2_kernel", &err);
    } catch (cl::Error err) {
        reportError(err, "advec_cell_ydir_section1_sweep2_kernel");
    }

    try {
        advec_cell_ydir_sec2_knl = cl::Kernel(program, "advec_cell_ydir_section2_kernel", &err);
    } catch (cl::Error err) {
        reportError(err, "advec_cell_ydir_section2_kernel");
    }

    try {
        advec_cell_ydir_sec3_knl = cl::Kernel(program, "advec_cell_ydir_section3_kernel", &err);
    } catch (cl::Error err) {
        reportError(err, "advec_cell_ydir_section3_kernel");
    }

    try {
        advec_mom_vol_knl = cl::Kernel(program, "advec_mom_vol_ocl_kernel", &err);
    } catch (cl::Error err) {
        reportError(err, "advec_mom_vol_ocl_kernel");
    }

    try {
        advec_mom_node_x_knl = cl::Kernel(program, "advec_mom_node_ocl_kernel_x", &err);
    } catch (cl::Error err) {
        reportError(err, "advec_mom_vol_ocl_kernel");
    }

    try {
        advec_mom_node_mass_pre_x_knl = cl::Kernel(program, "advec_mom_node_mass_pre_ocl_kernel_x", &err);
    } catch (cl::Error err) {
        reportError(err, "advec_mom_vol_ocl_kernel");
    }

    try {
        advec_mom_flux_x_vec1_knl = cl::Kernel(program, "advec_mom_flux_ocl_kernel_x_vec1", &err);
    } catch (cl::Error err) {
        reportError(err, "advec_mom_ocl_kernel vec1");
    }

    try {
        advec_mom_flux_x_vecnot1_knl = cl::Kernel(program, "advec_mom_flux_ocl_kernel_x_notvec1", &err);
    } catch (cl::Error err) {
        reportError(err, "advec_mom_ocl_kernel vecnot1");
    }

    try {
        advec_mom_vel_x_knl = cl::Kernel(program, "advec_mom_vel_ocl_kernel_x", &err);
    } catch (cl::Error err) {
        reportError(err, "advec_mom_ocl_kernel");
    }

    try {
        advec_mom_node_y_knl = cl::Kernel(program, "advec_mom_node_ocl_kernel_y", &err);
    } catch (cl::Error err) {
        reportError(err, "advec_mom_vol_ocl_kernel");
    }

    try {
        advec_mom_node_mass_pre_y_knl = cl::Kernel(program, "advec_mom_node_mass_pre_ocl_kernel_y", &err);
    } catch (cl::Error err) {
        reportError(err, "advec_mom_vol_ocl_kernel");
    }

    try {
        advec_mom_flux_y_vec1_knl = cl::Kernel(program, "advec_mom_flux_ocl_kernel_y_vec1", &err);
    } catch (cl::Error err) {
        reportError(err, "advec_mom_ocl_kernel vec1");
    }

    try {
        advec_mom_flux_y_vecnot1_knl = cl::Kernel(program, "advec_mom_flux_ocl_kernel_y_notvec1", &err);
    } catch (cl::Error err) {
        reportError(err, "advec_mom_ocl_kernel vecnot1");
    }

    try {
        advec_mom_vel_y_knl = cl::Kernel(program, "advec_mom_vel_ocl_kernel_y", &err);
    } catch (cl::Error err) {
        reportError(err, "advec_mom_ocl_kernel");
    }

    try {
        pdv_correct_knl = cl::Kernel(program, "pdv_correct_ocl_kernel", &err);
    } catch (cl::Error err) {
        reportError(err, "creating pdv_ocl_kernel_correct");
    }

    try {
        pdv_predict_knl = cl::Kernel(program, "pdv_predict_ocl_kernel", &err);
    } catch (cl::Error err) {
        reportError(err, "creating pdv_ocl_kernel_predict");
    }

    try {
        dt_calc_knl = cl::Kernel(program, "calc_dt_ocl_kernel", &err);
    } catch (cl::Error err) {
        reportError(err, "calc_dt_ocl_kernel");
    }

    try {
        revert_knl = cl::Kernel(program, "revert_ocl_kernel", &err);
    } catch (cl::Error err) {
        reportError(err, "creating revert_ocl_kernel");
    }

    try {
        reset_field_knl = cl::Kernel(program, "reset_field_ocl_kernel", &err);
    } catch (cl::Error err) {
        reportError(err, "creating reset_field_ocl_kernel");
    }

    try {
        generate_chunk_knl = cl::Kernel(program, "generate_chunk_ocl_kernel");
    } catch (cl::Error err) {
        reportError(err, "creating generate_chunk_ocl_kernel");
    }

    try {
        initialise_chunk_cell_x_knl = cl::Kernel(program, "initialise_chunk_cell_x_ocl_kernel", &err);
    } catch (cl::Error err) {
        reportError(err, "creating initialise_chunk_cell_x_ocl_kernel");
    }

    try {
        initialise_chunk_cell_y_knl = cl::Kernel(program, "initialise_chunk_cell_y_ocl_kernel", &err);
    } catch (cl::Error err) {
        reportError(err, "creating initialise_chunk_cell_y_ocl_kernel");
    }

    try {
        initialise_chunk_vertex_x_knl = cl::Kernel(program, "initialise_chunk_vertex_x_ocl_kernel", &err);
    } catch (cl::Error err) {
        reportError(err, "creating initialise_chunk_vertex_x_ocl_kernel");
    }

    try {
        initialise_chunk_vertex_y_knl = cl::Kernel(program, "initialise_chunk_vertex_y_ocl_kernel", &err);
    } catch (cl::Error err) {
        reportError(err, "creating initialise_chunk_vertex_y_ocl_kernel");
    }

    try {
        initialise_chunk_volume_area_knl = cl::Kernel(program, "initialise_chunk_volume_area_ocl_kernel", &err);
    } catch (cl::Error err) {
        reportError(err, "creating initialise_chunk_volume_area_ocl_kernel");
    }

    try {
        field_summary_knl = cl::Kernel(program, "field_summary_ocl_kernel", &err);
    } catch (cl::Error err) {
        reportError(err, "creating field_summary_ocl_kernel");
    }

    try {
        update_halo_bottom_cell_knl = cl::Kernel(program, "update_halo_bottom_cell_ocl_kernel", &err);
        update_halo_bottom_vel_knl = cl::Kernel(program, "update_halo_bottom_vel_ocl_kernel", &err);
        update_halo_bottom_flux_x_knl = cl::Kernel(program, "update_halo_bottom_flux_x_ocl_kernel", &err);
        update_halo_bottom_flux_y_knl = cl::Kernel(program, "update_halo_bottom_flux_y_ocl_kernel", &err);
    } catch (cl::Error err) {
        reportError(err, "creating update_halo_bottom_ocl_kernel");
    }

    try {
        update_halo_top_cell_knl = cl::Kernel(program, "update_halo_top_cell_ocl_kernel", &err);
        update_halo_top_vel_knl = cl::Kernel(program, "update_halo_top_vel_ocl_kernel", &err);
        update_halo_top_flux_x_knl = cl::Kernel(program, "update_halo_top_flux_x_ocl_kernel", &err);
        update_halo_top_flux_y_knl = cl::Kernel(program, "update_halo_top_flux_y_ocl_kernel", &err);
    } catch (cl::Error err) {
        reportError(err, "creating update_halo_top_ocl_kernel");
    }

    try {
        update_halo_right_cell_knl = cl::Kernel(program, "update_halo_right_cell_ocl_kernel", &err);
        update_halo_right_vel_knl = cl::Kernel(program, "update_halo_right_vel_ocl_kernel", &err);
        update_halo_right_flux_x_knl = cl::Kernel(program, "update_halo_right_flux_x_ocl_kernel", &err);
        update_halo_right_flux_y_knl = cl::Kernel(program, "update_halo_right_flux_y_ocl_kernel", &err);
    } catch (cl::Error err) {
        reportError(err, "creating update_halo_right_ocl_kernel");
    }

    try {
        update_halo_left_cell_knl = cl::Kernel(program, "update_halo_left_cell_ocl_kernel", &err);
        update_halo_left_vel_knl = cl::Kernel(program, "update_halo_left_vel_ocl_kernel", &err);
        update_halo_left_flux_x_knl = cl::Kernel(program, "update_halo_left_flux_x_ocl_kernel", &err);
        update_halo_left_flux_y_knl = cl::Kernel(program, "update_halo_left_flux_y_ocl_kernel", &err);
    } catch (cl::Error err) {
        reportError(err, "creating update_halo_left_ocl_kernel");
    }

    try {
        read_top_buffer_knl = cl::Kernel(program, "top_comm_buffer_pack");
        read_bottom_buffer_knl = cl::Kernel(program, "bottom_comm_buffer_pack");
        read_right_buffer_knl = cl::Kernel(program, "right_comm_buffer_pack");
        read_left_buffer_knl = cl::Kernel(program, "left_comm_buffer_pack");
    } catch(cl::Error err) {
        reportError(err, "creating comms buffer pack kernels");
    }

    try {
        write_top_buffer_knl = cl::Kernel(program, "top_comm_buffer_unpack");
        write_bottom_buffer_knl = cl::Kernel(program, "bottom_comm_buffer_unpack");
        write_right_buffer_knl = cl::Kernel(program, "right_comm_buffer_unpack");
        write_left_buffer_knl = cl::Kernel(program, "left_comm_buffer_unpack");
    } catch(cl::Error err) {
        reportError(err, "creating comms buffer unpack kernels");
    }

}

void CloverCL::readVisualisationBuffers(
                int x_max,
                int y_max,
                double* vertexx,
                double* vertexy,
                double* density0,
                double* energy0,
                double* pressure,
                double* viscosity,
                double* xvel0,
                double* yvel0)
{

    cl::Event event1, event2, event3, event4, event5, event6, event7, event8;
    std::vector<cl::Event> events;

    try {
        queue.enqueueReadBuffer( CloverCL::vertexx_buffer, CL_FALSE, 0, (x_max+5)*sizeof(double), vertexx, NULL, &event1);
    } catch (cl::Error err) {
        reportError(err, "readVisualisationBuffers() vertexx");
    }

    try {
        queue.enqueueReadBuffer( CloverCL::vertexy_buffer, CL_FALSE, 0, (y_max+5)*sizeof(double), vertexy, NULL, &event2);
    } catch (cl::Error err) {
        reportError(err, "readVisualisationBuffers() vertexy");
    }

    try {
        queue.enqueueReadBuffer( CloverCL::density0_buffer, CL_FALSE, 0, (x_max+4)*(y_max+4)*sizeof(double), density0, NULL, &event3);
    } catch (cl::Error err) {
        reportError(err, "readVisualisationBuffers() density0");
    }

    try {
        queue.enqueueReadBuffer( CloverCL::energy0_buffer, CL_FALSE, 0, (x_max+4)*(y_max+4)*sizeof(double), energy0, NULL, &event4);
    } catch (cl::Error err) {
        reportError(err, "readVisualisationBuffers() energy0");
    }

    try {
        queue.enqueueReadBuffer( CloverCL::pressure_buffer, CL_FALSE, 0, (x_max+4)*(y_max+4)*sizeof(double), pressure, NULL, &event5);
    } catch (cl::Error err) {
        reportError(err, "readVisualisationBuffers() pressure");
    }

    try {
        queue.enqueueReadBuffer( CloverCL::viscosity_buffer, CL_FALSE, 0, (x_max+4)*(y_max+4)*sizeof(double), viscosity, NULL, &event6);
    } catch (cl::Error err) {
        reportError(err, "readVisualisationBuffers() viscosity");
    }

    try {
        queue.enqueueReadBuffer( CloverCL::xvel0_buffer, CL_FALSE, 0, (x_max+5)*(y_max+5)*sizeof(double), xvel0, NULL, &event7);
    } catch (cl::Error err) {
        reportError(err, "readVisualisationBuffers() xvel0");
    }

    try {
        queue.enqueueReadBuffer( CloverCL::yvel0_buffer, CL_FALSE, 0, (x_max+5)*(y_max+5)*sizeof(double), yvel0, NULL, &event8);
    } catch (cl::Error err) {
        reportError(err, "readVisualisationBuffers() yvel0");
    }

    events.push_back(event1);
    events.push_back(event2);
    events.push_back(event3);
    events.push_back(event4);
    events.push_back(event5);
    events.push_back(event6);
    events.push_back(event7);
    events.push_back(event8);

    cl::Event::waitForEvents(events);
}

void CloverCL::readCommunicationBuffer(
        int* xmin,
        int* xmax,
        int* ymin,
        int* ymax,
        int* depth,
        int* xinc,
        int* yinc,
        int* field_name,
        double* buffer,
        int* which_edge)
{
    cl::Buffer* field_buffer;
    cl::Buffer* comm_buffer;
    cl::Event event1;
    int buff_length;
    int buff_min;

    global_events.clear();
    global_events.push_back(last_event);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::stringstream ss_rank;
    ss_rank << rank;

    switch(*field_name) {
        case FIELD_DENSITY0: field_buffer = &density0_buffer; break;
        case FIELD_DENSITY1: field_buffer = &density1_buffer; break;
        case FIELD_ENERGY0: field_buffer = &energy0_buffer; break;
        case FIELD_ENERGY1: field_buffer = &energy1_buffer; break;
        case FIELD_PRESSURE: field_buffer = &pressure_buffer; break;
        case FIELD_VISCOSITY: field_buffer = &viscosity_buffer; break;
        case FIELD_SOUNDSPEED: field_buffer = &soundspeed_buffer; break;
        case FIELD_XVEL0: field_buffer = &xvel0_buffer; break;
        case FIELD_XVEL1: field_buffer = &xvel1_buffer; break;
        case FIELD_YVEL0: field_buffer = &yvel0_buffer; break;
        case FIELD_YVEL1: field_buffer = &yvel1_buffer; break;
        case FIELD_VOL_FLUX_X: field_buffer = &vol_flux_x_buffer; break;
        case FIELD_VOL_FLUX_Y: field_buffer = &vol_flux_y_buffer; break;
        case FIELD_MASS_FLUX_X: field_buffer = &mass_flux_x_buffer; break;
        case FIELD_MASS_FLUX_Y: field_buffer = &mass_flux_y_buffer; break;
    }

    cl::size_t<3> b_origin;
    cl::size_t<3> h_origin;
    cl::size_t<3> region;

    size_t b_row_pitch = sizeof(double) * (*xmax + *xinc + 4);
    size_t b_slice_pitch = 0;
    size_t h_row_pitch = 0;
    size_t h_slice_pitch = 0;

    h_origin[0] = 0;
    h_origin[1] = 0;
    h_origin[2] = 0;

    switch(*which_edge) {
        case 1: comm_buffer = &(top_send_buffer);
                buff_length = *xmax + *xinc + (2 * *depth);
                buff_min = *xmin; 
                b_origin[0] = ((*xmin+1) - *depth)*sizeof(double);
                b_origin[1] = ((*ymax+1)-(*depth-1));
                b_origin[2] = 0;
                region[0] = ((*xmax)+*xinc+2*(*depth))*sizeof(double);
                region[1] = *depth;
                region[2] = 1;
                break;
        case 2: comm_buffer = &right_send_buffer;
                buff_length = *ymax + *yinc + (2 * *depth);
                buff_min = *ymin; 
                b_origin[0] = ((*xmax+1)-(*depth-1))*sizeof(double);
                b_origin[1] = ((*ymin+1) - (*depth));
                b_origin[2] = 0;
                region[0] = (*depth)*sizeof(double);
                region[1] = (*ymax)+*yinc+(2* *depth);
                region[2] = 1;
                break;
        case 3: comm_buffer = &bottom_send_buffer;
                buff_length = *xmax + *xinc + (2 * *depth);
                buff_min = *xmin; 
                b_origin[0] = ((*xmin+1) - *depth)*sizeof(double);
                b_origin[1] = (*ymin+1+*yinc);
                b_origin[2] = 0;
                region[0] = ((*xmax)+*xinc+2*(*depth))*sizeof(double);
                region[1] = *depth;
                region[2] = 1;
                break;
        case 4: comm_buffer = &left_send_buffer;
                buff_length = *ymax + *yinc + (2 * *depth);
                buff_min = *ymin; 
                b_origin[0] = ((*xmin+1+*xinc)*sizeof(double));
                b_origin[1] = ((*ymin+1) - (*depth));
                b_origin[2] = 0;
                region[0] = (*depth)*sizeof(double);
                region[1] = (*ymax)+*yinc+(2* *depth);
                region[2] = 1;
                break;
    }

    buff_length = buff_length * *depth;

    try {
        queue.enqueueReadBufferRect( *field_buffer, CL_TRUE, b_origin, h_origin, region, b_row_pitch, 
                                     b_slice_pitch, h_row_pitch, h_slice_pitch, buffer, &global_events);
    } catch (cl::Error err) {
        reportError(err, "readCommunicationBuffer enqueueReadBufferRect");
    }
}

void CloverCL::writeCommunicationBuffer(
        int* xmin,
        int* xmax,
        int* ymin,
        int* ymax,
        int* depth,
        int* xinc,
        int* yinc,
        int* field_name,
        double* buffer,
        int* which_edge)
{
    cl::Buffer* field_buffer;
    cl::Buffer* comm_buffer;
    cl::Kernel* comm_kernel;
    cl::Event event1;
    int buff_length;
    int buff_min;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::stringstream ss_rank;
    ss_rank << rank;

    switch(*field_name) {
        case FIELD_DENSITY0: field_buffer = &density0_buffer; break;
        case FIELD_DENSITY1: field_buffer = &density1_buffer; break;
        case FIELD_ENERGY0: field_buffer = &energy0_buffer; break;
        case FIELD_ENERGY1: field_buffer = &energy1_buffer; break;
        case FIELD_PRESSURE: field_buffer = &pressure_buffer; break;
        case FIELD_VISCOSITY: field_buffer = &viscosity_buffer; break;
        case FIELD_SOUNDSPEED: field_buffer = &soundspeed_buffer; break;
        case FIELD_XVEL0: field_buffer = &xvel0_buffer; break;
        case FIELD_XVEL1: field_buffer = &xvel1_buffer; break;
        case FIELD_YVEL0: field_buffer = &yvel0_buffer; break;
        case FIELD_YVEL1: field_buffer = &yvel1_buffer; break;
        case FIELD_VOL_FLUX_X: field_buffer = &vol_flux_x_buffer; break;
        case FIELD_VOL_FLUX_Y: field_buffer = &vol_flux_y_buffer; break;
        case FIELD_MASS_FLUX_X: field_buffer = &mass_flux_x_buffer; break;
        case FIELD_MASS_FLUX_Y: field_buffer = &mass_flux_y_buffer; break;
    }

    cl::size_t<3> b_origin;
    cl::size_t<3> h_origin;
    cl::size_t<3> region;

    size_t b_row_pitch = sizeof(double) * (*xmax + *xinc + 4);
    size_t b_slice_pitch = 0;
    size_t h_row_pitch = 0;
    size_t h_slice_pitch = 0;

    h_origin[0] = 0;
    h_origin[1] = 0;
    h_origin[2] = 0;

    switch(*which_edge) {
        case 1: comm_buffer = &(top_send_buffer);
                buff_length = *xmax + *xinc + (2 * *depth);
                buff_min = *xmin; 
                b_origin[0] = ((*xmin+1) - *depth)*sizeof(double);
                b_origin[1] = ((*ymax+1)+1+*yinc);
                b_origin[2] = 0;
                region[0] = ((*xmax)+*xinc+2*(*depth))*sizeof(double);
                region[1] = *depth;
                region[2] = 1;
                break;
        case 2: comm_buffer = &right_send_buffer;
                buff_length = *ymax + *yinc + (2 * *depth);
                buff_min = *ymin; 
                b_origin[0] = ((*xmax+1)+1+*xinc)*sizeof(double);
                b_origin[1] = ((*ymin+1) - (*depth));
                b_origin[2] = 0;
                region[0] = (*depth)*sizeof(double);
                region[1] = (*ymax)+*yinc+(2* *depth);
                region[2] = 1;
                break;
        case 3: comm_buffer = &bottom_send_buffer;
                buff_length = *xmax + *xinc + (2 * *depth);
                buff_min = *xmin; 
                b_origin[0] = ((*xmin+1) - *depth)*sizeof(double);
                b_origin[1] = (*ymin+1)-*depth;
                b_origin[2] = 0;
                region[0] = ((*xmax)+*xinc+2*(*depth))*sizeof(double);
                region[1] = *depth;
                region[2] = 1;
                break;
        case 4: comm_buffer = &left_send_buffer;
                buff_length = *ymax + *yinc + (2 * *depth);
                buff_min = *ymin; 
                b_origin[0] = ((*xmin+1-(*depth))*sizeof(double));
                b_origin[1] = ((*ymin+1) - (*depth));
                b_origin[2] = 0;
                region[0] = (*depth)*sizeof(double);
                region[1] = (*ymax)+*yinc+(2* *depth);
                region[2] = 1;
                break;
    }

    buff_length = buff_length * *depth;

    try {
        queue.enqueueWriteBufferRect( *field_buffer, CL_TRUE, b_origin, h_origin, region, b_row_pitch, 
                                      b_slice_pitch, h_row_pitch, h_slice_pitch, buffer); 
    } catch (cl::Error err) {
        reportError(err, "writeCommunicationBuffer enqueueWriteBufferRect");
    }
}

void CloverCL::readAllCommunicationBuffers(
        int* x_max,
        int* y_max,
        double* density0,
        double* density1,
        double* energy0,
        double* energy1,
        double* pressure,
        double* viscosity,
        double* soundspeed,
        double* xvel0,
        double* xvel1,
        double* yvel0,
        double* yvel1,
        double* vol_flux_x,
        double* vol_flux_y,
        double* mass_flux_x,
        double* mass_flux_y)
{

    cl::Event event1;

    try {
        queue.enqueueReadBuffer( CloverCL::density0_buffer, CL_TRUE, 0, 
                                (*x_max+4)*(*y_max+4)*sizeof(double), density0, NULL, &event1);
    } catch (cl::Error err) {
        reportError(err, "readAllCommunicationBuffers() density0");
    }

    try {
        queue.enqueueReadBuffer( CloverCL::density1_buffer, CL_TRUE, 0, 
                                 (*x_max+4)*(*y_max+4)*sizeof(double), density1, NULL, &event1);
    } catch (cl::Error err) {
        reportError(err, "readAllCommunicationBuffers() density1");
    }

    try {
        queue.enqueueReadBuffer( CloverCL::energy0_buffer, CL_TRUE, 0, 
                                 (*x_max+4)*(*y_max+4)*sizeof(double), energy0, NULL, &event1);
    } catch (cl::Error err) {
        reportError(err, "readAllCommunicationBuffers() energy0");
    }

    try {
        queue.enqueueReadBuffer( CloverCL::energy1_buffer, CL_TRUE, 0, 
                                 (*x_max+4)*(*y_max+4)*sizeof(double), energy1, NULL, &event1);
    } catch (cl::Error err) {
        reportError(err, "readAllCommunicationBuffers() energy1");
    }

    try {
        queue.enqueueReadBuffer( CloverCL::pressure_buffer, CL_TRUE, 0, 
                                 (*x_max+4)*(*y_max+4)*sizeof(double), pressure, NULL, &event1);
    } catch (cl::Error err) {
        reportError(err, "readAllCommunicationBuffers() pressure");
    }

    try {
        queue.enqueueReadBuffer( CloverCL::viscosity_buffer, CL_TRUE, 0, 
                                 (*x_max+4)*(*y_max+4)*sizeof(double), viscosity, NULL, &event1);
    } catch (cl::Error err) {
        reportError(err, "readAllCommunicationBuffers() viscosity");
    }

    try {
        queue.enqueueReadBuffer( CloverCL::soundspeed_buffer, CL_TRUE, 0, 
                                 (*x_max+4)*(*y_max+4)*sizeof(double), soundspeed, NULL, &event1);
    } catch (cl::Error err) {
        reportError(err, "readAllCommunicationBuffers() soundspeed");
    }

    try {
        queue.enqueueReadBuffer( CloverCL::xvel0_buffer, CL_TRUE, 0, 
                                 (*x_max+5)*(*y_max+5)*sizeof(double), xvel0, NULL, &event1);
    } catch (cl::Error err) {
        reportError(err, "readAllCommunicationBuffers() xvel0");
    }

    try {
        queue.enqueueReadBuffer( CloverCL::xvel1_buffer, CL_TRUE, 0, 
                                 (*x_max+5)*(*y_max+5)*sizeof(double), xvel1, NULL, &event1);
    } catch (cl::Error err) {
        reportError(err, "readAllCommunicationBuffers() xvel1");
    }

    try {
        queue.enqueueReadBuffer( CloverCL::yvel0_buffer, CL_TRUE, 0, 
                                 (*x_max+5)*(*y_max+5)*sizeof(double), yvel0, NULL, &event1);
    } catch (cl::Error err) {
        reportError(err, "readAllCommunicationBuffers() yvel0");
    }

    try {
        queue.enqueueReadBuffer( CloverCL::yvel1_buffer, CL_TRUE, 0, 
                                 (*x_max+5)*(*y_max+5)*sizeof(double), yvel1, NULL, &event1);
    } catch (cl::Error err) {
        reportError(err, "readAllCommunicationBuffers() yvel1");
    }

    try {
        queue.enqueueReadBuffer( CloverCL::mass_flux_x_buffer, CL_TRUE, 0, 
                                 (*x_max+5)*(*y_max+4)*sizeof(double), mass_flux_x, NULL, &event1);
    } catch (cl::Error err) {
        reportError(err, "readAllCommunicationBuffers() mass_flux_x");
    }

    try {
        queue.enqueueReadBuffer( CloverCL::vol_flux_x_buffer, CL_TRUE, 0, 
                                 (*x_max+5)*(*y_max+4)*sizeof(double), vol_flux_x, NULL, &event1);
    } catch (cl::Error err) {
        reportError(err, "readAllCommunicationBuffers() vol_flux_x");
    }

    try {
        queue.enqueueReadBuffer( CloverCL::mass_flux_y_buffer, CL_TRUE, 0, 
                                 (*x_max+4)*(*y_max+5)*sizeof(double), mass_flux_y, NULL, &event1);
    } catch (cl::Error err) {
        reportError(err, "readAllCommunicationBuffers() mass_flux_y");
    }

    try {
        queue.enqueueReadBuffer( CloverCL::vol_flux_y_buffer, CL_TRUE, 0, 
                                 (*x_max+4)*(*y_max+5)*sizeof(double), vol_flux_y, NULL, &event1);
    } catch (cl::Error err) {
        reportError(err, "readAllCommunicationBuffers() vol_flux_y");
    }
}

void CloverCL::writeAllCommunicationBuffers(
        int* x_max,
        int* y_max,
        double* density0,
        double* density1,
        double* energy0,
        double* energy1,
        double* pressure,
        double* viscosity,
        double* soundspeed,
        double* xvel0,
        double* xvel1,
        double* yvel0,
        double* yvel1,
        double* vol_flux_x,
        double* vol_flux_y,
        double* mass_flux_x,
        double* mass_flux_y)
{

    cl::Event event1;

    try {
        queue.enqueueWriteBuffer( CloverCL::density0_buffer, CL_TRUE, 0, 
                                  (*x_max+4)*(*y_max+4)*sizeof(double), density0, NULL, &event1);
    } catch (cl::Error err) {
        reportError(err, "readAllCommunicationBuffers() density0");
    }

    try {
        queue.enqueueWriteBuffer( CloverCL::density1_buffer, CL_TRUE, 0, 
                                 (*x_max+4)*(*y_max+4)*sizeof(double), density1, NULL, &event1);
    } catch (cl::Error err) {
        reportError(err, "readAllCommunicationBuffers() density1");
    }

    try {
        queue.enqueueWriteBuffer( CloverCL::energy0_buffer, CL_TRUE, 0, 
                                 (*x_max+4)*(*y_max+4)*sizeof(double), energy0, NULL, &event1);
    } catch (cl::Error err) {
        reportError(err, "readAllCommunicationBuffers() energy0");
    }

    try {
        queue.enqueueWriteBuffer( CloverCL::energy1_buffer, CL_TRUE, 0, 
                                  (*x_max+4)*(*y_max+4)*sizeof(double), energy1, NULL, &event1);
    } catch (cl::Error err) {
        reportError(err, "readAllCommunicationBuffers() energy1");
    }

    try {
        queue.enqueueWriteBuffer( CloverCL::pressure_buffer, CL_TRUE, 0, 
                                  (*x_max+4)*(*y_max+4)*sizeof(double), pressure, NULL, &event1);
    } catch (cl::Error err) {
        reportError(err, "readAllCommunicationBuffers() pressure");
    }

    try {
        queue.enqueueWriteBuffer( CloverCL::viscosity_buffer, CL_TRUE, 0, 
                                  (*x_max+4)*(*y_max+4)*sizeof(double), viscosity, NULL, &event1);
    } catch (cl::Error err) {
        reportError(err, "readAllCommunicationBuffers() viscosity");
    }

    try {
        queue.enqueueWriteBuffer( CloverCL::soundspeed_buffer, CL_TRUE, 0, 
                                  (*x_max+4)*(*y_max+4)*sizeof(double), soundspeed, NULL, &event1);
    } catch (cl::Error err) {
        reportError(err, "readAllCommunicationBuffers() soundspeed");
    }

    try {
        queue.enqueueWriteBuffer( CloverCL::xvel0_buffer, CL_TRUE, 0, 
                                  (*x_max+5)*(*y_max+5)*sizeof(double), xvel0, NULL, &event1);
    } catch (cl::Error err) {
        reportError(err, "readAllCommunicationBuffers() xvel0");
    }

    try {
        queue.enqueueWriteBuffer( CloverCL::xvel1_buffer, CL_TRUE, 0, 
                                  (*x_max+5)*(*y_max+5)*sizeof(double), xvel1, NULL, &event1);
    } catch (cl::Error err) {
        reportError(err, "readAllCommunicationBuffers() xvel1");
    }

    try {
        queue.enqueueWriteBuffer( CloverCL::yvel0_buffer, CL_TRUE, 0, 
                                  (*x_max+5)*(*y_max+5)*sizeof(double), yvel0, NULL, &event1);
    } catch (cl::Error err) {
        reportError(err, "readAllCommunicationBuffers() yvel0");
    }

    try {
        queue.enqueueWriteBuffer( CloverCL::yvel1_buffer, CL_TRUE, 0, 
                                  (*x_max+5)*(*y_max+5)*sizeof(double), yvel1, NULL, &event1);
    } catch (cl::Error err) {
        reportError(err, "readAllCommunicationBuffers() yvel1");
    }

    try {
        queue.enqueueWriteBuffer( CloverCL::mass_flux_x_buffer, CL_TRUE, 0, 
                                  (*x_max+5)*(*y_max+4)*sizeof(double), mass_flux_x, NULL, &event1);
    } catch (cl::Error err) {
        reportError(err, "readAllCommunicationBuffers() mass_flux_x");
    }

    try {
        queue.enqueueWriteBuffer( CloverCL::vol_flux_x_buffer, CL_TRUE, 0, 
                                  (*x_max+5)*(*y_max+4)*sizeof(double), vol_flux_x, NULL, &event1);
    } catch (cl::Error err) {
        reportError(err, "readAllCommunicationBuffers() vol_flux_x");
    }

    try {
        queue.enqueueWriteBuffer( CloverCL::mass_flux_y_buffer, CL_TRUE, 0, 
                                  (*x_max+4)*(*y_max+5)*sizeof(double), mass_flux_y, NULL, &event1);
    } catch (cl::Error err) {
        reportError(err, "readAllCommunicationBuffers() mass_flux_y");
    }

    try {
        queue.enqueueWriteBuffer( CloverCL::vol_flux_y_buffer, CL_TRUE, 0, 
                                  (*x_max+4)*(*y_max+5)*sizeof(double), vol_flux_y, NULL, &event1);
    } catch (cl::Error err) {
        reportError(err, "readAllCommunicationBuffers() vol_flux_y");
    }
}

void CloverCL::enqueueKernel_nooffsets_localwg( cl::Kernel kernel, int num_x, int num_y, int wg_x, int wg_y)
{
    int x_rnd = (num_x / wg_x) * wg_x;

    if ((x_rnd != num_x))
        x_rnd = x_rnd + wg_x;


    int y_rnd = ( num_y / wg_y) * wg_y;

    if (y_rnd != num_y) {
        y_rnd = y_rnd + wg_y; 
    }

    try {

        queue.enqueueNDRangeKernel( kernel, cl::NullRange, cl::NDRange(x_rnd, y_rnd), 
                                    cl::NDRange(wg_x, wg_y), 
                                    NULL, NULL); 
    } catch(cl::Error err) {

        std::string kernel_name;
        kernel.getInfo(CL_KERNEL_FUNCTION_NAME, &kernel_name);
        std::cout << "launching kernel: " << kernel_name << "xnum: " << x_rnd << " ynum: " << y_rnd 
                  << " wg_x: " << wg_x << " wg_y: " << wg_y << std::endl;
        reportError(err, kernel_name);
    }
}


void CloverCL::enqueueKernel_nooffsets_recordevent_localwg( cl::Kernel kernel, int num_x, int num_y, int wg_x, int wg_y)
{
    int x_rnd = (num_x / wg_x) * wg_x;

    if ((x_rnd != num_x))
        x_rnd = x_rnd + wg_x;


    int y_rnd = ( num_y / wg_y) * wg_y;

    if (y_rnd != num_y) {
        y_rnd = y_rnd + wg_y; 
    }


    try {

        queue.enqueueNDRangeKernel( kernel, cl::NullRange, cl::NDRange(x_rnd, y_rnd), 
                                    cl::NDRange(wg_x, wg_y), 
                                    NULL, &last_event); 
    } catch(cl::Error err) {

        std::string kernel_name;
        kernel.getInfo(CL_KERNEL_FUNCTION_NAME, &kernel_name);
        std::cout << "launching kernel: " << kernel_name << "xnum: " << x_rnd << " ynum: " << y_rnd 
                  << " wg_x: " << wg_x << " wg_y: " << wg_y << std::endl;
        reportError(err, kernel_name);
    }
}

void CloverCL::enqueueKernel( cl::Kernel kernel, int x_min, int x_max, int y_min, int y_max)
{
    int x_max_opt;
    int x_tot = (x_max - x_min) + 1;

    int x_rnd = (x_tot / prefer_wg_multiple) * prefer_wg_multiple;

    if ((x_rnd != x_tot))
        x_rnd = x_rnd + prefer_wg_multiple;

    x_max_opt = x_rnd + x_min - 1;

    try {
        queue.enqueueNDRangeKernel( kernel, cl::NDRange(x_min, y_min), cl::NDRange(x_max_opt, y_max), 
                                    cl::NullRange, NULL, &last_event);
    } catch(cl::Error err) {

        std::string kernel_name;
        kernel.getInfo(CL_KERNEL_FUNCTION_NAME, &kernel_name);
        std::cout << "launching kernel: " << kernel_name << "xmin: " << x_min << " xmax: " << x_max_opt 
                                          << " ymin: " << y_min << " ymax: " << y_max << std::endl;
        reportError(err, kernel_name);
    }
}

void CloverCL::enqueueKernel( cl::Kernel kernel, int min, int max)
{
    int tot = (max - min) + 1;

    int rnd = (tot / prefer_wg_multiple) * prefer_wg_multiple;

    if ((rnd != tot))
        rnd = rnd + prefer_wg_multiple;

    int min_opt, max_opt;

    min_opt = min;

    max_opt = rnd + min - 1;

    try {
        queue.enqueueNDRangeKernel( kernel, cl::NDRange(min_opt), cl::NDRange(max_opt), cl::NullRange, NULL, &last_event);

    } catch(cl::Error err) {

        std::string kernel_name;
        kernel.getInfo(CL_KERNEL_FUNCTION_NAME, &kernel_name);
        reportError(err, kernel_name);
    }
}


void CloverCL::read_back_all_ocl_buffers(double* density0, double* density1, double* energy0, double* energy1,
                                         double* pressure, double* viscosity, double* soundspeed,
                                         double* xvel0, double* xvel1, double* yvel0, double* yvel1,
                                         double* vol_flux_x, double* mass_flux_x,
                                         double* vol_flux_y, double* mass_flux_y,
                                         double* celldx, double* celldy, double* volume)
{

    CloverCL::queue.finish();
    CloverCL::outoforder_queue.finish(); 

    CloverCL::outoforder_queue.enqueueReadBuffer(CloverCL::density0_buffer,    CL_FALSE, 0, (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), density0, NULL, NULL);
    CloverCL::outoforder_queue.enqueueReadBuffer(CloverCL::density1_buffer,    CL_FALSE, 0, (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), density1, NULL, NULL);
    CloverCL::outoforder_queue.enqueueReadBuffer(CloverCL::energy0_buffer,     CL_FALSE, 0, (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), energy0, NULL, NULL);
    CloverCL::outoforder_queue.enqueueReadBuffer(CloverCL::energy1_buffer,     CL_FALSE, 0, (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), energy1, NULL, NULL);
    CloverCL::outoforder_queue.enqueueReadBuffer(CloverCL::pressure_buffer,    CL_FALSE, 0, (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), pressure, NULL, NULL);
    CloverCL::outoforder_queue.enqueueReadBuffer(CloverCL::viscosity_buffer,   CL_FALSE, 0, (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), viscosity, NULL, NULL);
    CloverCL::outoforder_queue.enqueueReadBuffer(CloverCL::soundspeed_buffer,  CL_FALSE, 0, (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), soundspeed, NULL, NULL);
    CloverCL::outoforder_queue.enqueueReadBuffer(CloverCL::xvel0_buffer,       CL_FALSE, 0, (CloverCL::xmax_c+5)*(CloverCL::ymax_c+5)*sizeof(double), xvel0, NULL, NULL);
    CloverCL::outoforder_queue.enqueueReadBuffer(CloverCL::xvel1_buffer,       CL_FALSE, 0, (CloverCL::xmax_c+5)*(CloverCL::ymax_c+5)*sizeof(double), xvel1, NULL, NULL);
    CloverCL::outoforder_queue.enqueueReadBuffer(CloverCL::yvel0_buffer,       CL_FALSE, 0, (CloverCL::xmax_c+5)*(CloverCL::ymax_c+5)*sizeof(double), yvel0, NULL, NULL);
    CloverCL::outoforder_queue.enqueueReadBuffer(CloverCL::yvel1_buffer,       CL_FALSE, 0, (CloverCL::xmax_c+5)*(CloverCL::ymax_c+5)*sizeof(double), yvel1, NULL, NULL);
    CloverCL::outoforder_queue.enqueueReadBuffer(CloverCL::vol_flux_x_buffer,  CL_FALSE, 0, (CloverCL::xmax_c+5)*(CloverCL::ymax_c+4)*sizeof(double), vol_flux_x, NULL, NULL);
    CloverCL::outoforder_queue.enqueueReadBuffer(CloverCL::vol_flux_y_buffer,  CL_FALSE, 0, (CloverCL::xmax_c+4)*(CloverCL::ymax_c+5)*sizeof(double), vol_flux_y, NULL, NULL);
    CloverCL::outoforder_queue.enqueueReadBuffer(CloverCL::mass_flux_x_buffer, CL_FALSE, 0, (CloverCL::xmax_c+5)*(CloverCL::ymax_c+4)*sizeof(double), mass_flux_x, NULL, NULL);
    CloverCL::outoforder_queue.enqueueReadBuffer(CloverCL::mass_flux_y_buffer, CL_FALSE, 0, (CloverCL::xmax_c+4)*(CloverCL::ymax_c+5)*sizeof(double), mass_flux_y, NULL, NULL);

    CloverCL::outoforder_queue.enqueueReadBuffer(CloverCL::celldx_buffer, CL_FALSE, 0, (CloverCL::xmax_c+4)*sizeof(double), celldx, NULL, NULL);
    CloverCL::outoforder_queue.enqueueReadBuffer(CloverCL::celldy_buffer, CL_FALSE, 0, (CloverCL::ymax_c+4)*sizeof(double), celldy, NULL, NULL);
    CloverCL::outoforder_queue.enqueueReadBuffer(CloverCL::volume_buffer, CL_FALSE, 0, (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), volume, NULL, NULL);

    CloverCL::outoforder_queue.finish();
}


void CloverCL::write_back_all_ocl_buffers(double* density0, double* density1, double* energy0, double* energy1,
                                         double* pressure, double* viscosity, double* soundspeed,
                                         double* xvel0, double* xvel1, double* yvel0, double* yvel1,
                                         double* vol_flux_x, double* mass_flux_x,
                                         double* vol_flux_y, double* mass_flux_y,
                                         double* celldx, double* celldy, double* volume )
{

    CloverCL::queue.finish();
    CloverCL::outoforder_queue.finish(); 

    CloverCL::outoforder_queue.enqueueWriteBuffer(CloverCL::density0_buffer,    CL_FALSE, 0, (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), density0, NULL, NULL);
    CloverCL::outoforder_queue.enqueueWriteBuffer(CloverCL::density1_buffer,    CL_FALSE, 0, (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), density1, NULL, NULL);
    CloverCL::outoforder_queue.enqueueWriteBuffer(CloverCL::energy0_buffer,     CL_FALSE, 0, (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), energy0, NULL, NULL);
    CloverCL::outoforder_queue.enqueueWriteBuffer(CloverCL::energy1_buffer,     CL_FALSE, 0, (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), energy1, NULL, NULL);
    CloverCL::outoforder_queue.enqueueWriteBuffer(CloverCL::pressure_buffer,    CL_FALSE, 0, (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), pressure, NULL, NULL);
    CloverCL::outoforder_queue.enqueueWriteBuffer(CloverCL::viscosity_buffer,   CL_FALSE, 0, (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), viscosity, NULL, NULL);
    CloverCL::outoforder_queue.enqueueWriteBuffer(CloverCL::soundspeed_buffer,  CL_FALSE, 0, (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), soundspeed, NULL, NULL);
    CloverCL::outoforder_queue.enqueueWriteBuffer(CloverCL::xvel0_buffer,       CL_FALSE, 0, (CloverCL::xmax_c+5)*(CloverCL::ymax_c+5)*sizeof(double), xvel0, NULL, NULL);
    CloverCL::outoforder_queue.enqueueWriteBuffer(CloverCL::xvel1_buffer,       CL_FALSE, 0, (CloverCL::xmax_c+5)*(CloverCL::ymax_c+5)*sizeof(double), xvel1, NULL, NULL);
    CloverCL::outoforder_queue.enqueueWriteBuffer(CloverCL::yvel0_buffer,       CL_FALSE, 0, (CloverCL::xmax_c+5)*(CloverCL::ymax_c+5)*sizeof(double), yvel0, NULL, NULL);
    CloverCL::outoforder_queue.enqueueWriteBuffer(CloverCL::yvel1_buffer,       CL_FALSE, 0, (CloverCL::xmax_c+5)*(CloverCL::ymax_c+5)*sizeof(double), yvel1, NULL, NULL);
    CloverCL::outoforder_queue.enqueueWriteBuffer(CloverCL::vol_flux_x_buffer,  CL_FALSE, 0, (CloverCL::xmax_c+5)*(CloverCL::ymax_c+4)*sizeof(double), vol_flux_x, NULL, NULL);
    CloverCL::outoforder_queue.enqueueWriteBuffer(CloverCL::vol_flux_y_buffer,  CL_FALSE, 0, (CloverCL::xmax_c+4)*(CloverCL::ymax_c+5)*sizeof(double), vol_flux_y, NULL, NULL);
    CloverCL::outoforder_queue.enqueueWriteBuffer(CloverCL::mass_flux_x_buffer, CL_FALSE, 0, (CloverCL::xmax_c+5)*(CloverCL::ymax_c+4)*sizeof(double), mass_flux_x, NULL, NULL);
    CloverCL::outoforder_queue.enqueueWriteBuffer(CloverCL::mass_flux_y_buffer, CL_FALSE, 0, (CloverCL::xmax_c+4)*(CloverCL::ymax_c+5)*sizeof(double), mass_flux_y, NULL, NULL);

    CloverCL::outoforder_queue.enqueueWriteBuffer(CloverCL::celldx_buffer, CL_FALSE, 0, (CloverCL::xmax_c+4)*sizeof(double), celldx, NULL, NULL);
    CloverCL::outoforder_queue.enqueueWriteBuffer(CloverCL::celldy_buffer, CL_FALSE, 0, (CloverCL::ymax_c+4)*sizeof(double), celldy, NULL, NULL);
    CloverCL::outoforder_queue.enqueueWriteBuffer(CloverCL::volume_buffer, CL_FALSE, 0, (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), volume, NULL, NULL);

    CloverCL::outoforder_queue.finish();
}


inline void CloverCL::checkErr(cl_int err, std::string name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " << name
            << " (" << errToString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    } else {
#ifdef OCL_VERBOSE
        std::cerr << "SUCCESS: " << name << std::endl;
#endif
    }
}

void CloverCL::reportError( cl::Error err, std::string message)
{
    std::cerr << "[CloverCL] ERROR: " << message << " " << err.what() << "(" 
              << CloverCL::errToString(err.err()) << ")" << std::endl;
    exit(EXIT_FAILURE);
}

std::string CloverCL::errToString(cl_int err)
{
    switch (err) {
        case CL_SUCCESS:                            return "Success!";
        case CL_DEVICE_NOT_FOUND:                   return "Device not found.";
        case CL_DEVICE_NOT_AVAILABLE:               return "Device not available";
        case CL_COMPILER_NOT_AVAILABLE:             return "Compiler not available";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "Memory object allocation failure";
        case CL_OUT_OF_RESOURCES:                   return "Out of resources";
        case CL_OUT_OF_HOST_MEMORY:                 return "Out of host memory";
        case CL_PROFILING_INFO_NOT_AVAILABLE:       return "Profiling information not available";
        case CL_MEM_COPY_OVERLAP:                   return "Memory copy overlap";
        case CL_IMAGE_FORMAT_MISMATCH:              return "Image format mismatch";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "Image format not supported";
        case CL_BUILD_PROGRAM_FAILURE:              return "Program build failure";
        case CL_MAP_FAILURE:                        return "Map failure";
        case CL_INVALID_VALUE:                      return "Invalid value";
        case CL_INVALID_DEVICE_TYPE:                return "Invalid device type";
        case CL_INVALID_PLATFORM:                   return "Invalid platform";
        case CL_INVALID_PROPERTY:                   return "Invalid property";
        case CL_INVALID_DEVICE:                     return "Invalid device";
        case CL_INVALID_CONTEXT:                    return "Invalid context";
        case CL_INVALID_QUEUE_PROPERTIES:           return "Invalid queue properties";
        case CL_INVALID_COMMAND_QUEUE:              return "Invalid command queue";
        case CL_INVALID_HOST_PTR:                   return "Invalid host pointer";
        case CL_INVALID_MEM_OBJECT:                 return "Invalid memory object";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "Invalid image format descriptor";
        case CL_INVALID_IMAGE_SIZE:                 return "Invalid image size";
        case CL_INVALID_SAMPLER:                    return "Invalid sampler";
        case CL_INVALID_BINARY:                     return "Invalid binary";
        case CL_INVALID_BUILD_OPTIONS:              return "Invalid build options";
        case CL_INVALID_PROGRAM:                    return "Invalid program";
        case CL_INVALID_PROGRAM_EXECUTABLE:         return "Invalid program executable";
        case CL_INVALID_KERNEL_NAME:                return "Invalid kernel name";
        case CL_INVALID_KERNEL_DEFINITION:          return "Invalid kernel definition";
        case CL_INVALID_KERNEL:                     return "Invalid kernel";
        case CL_INVALID_ARG_INDEX:                  return "Invalid argument index";
        case CL_INVALID_ARG_VALUE:                  return "Invalid argument value";
        case CL_INVALID_ARG_SIZE:                   return "Invalid argument size";
        case CL_INVALID_KERNEL_ARGS:                return "Invalid kernel arguments";
        case CL_INVALID_WORK_DIMENSION:             return "Invalid work dimension";
        case CL_INVALID_WORK_GROUP_SIZE:            return "Invalid work group size";
        case CL_INVALID_WORK_ITEM_SIZE:             return "Invalid work item size";
        case CL_INVALID_GLOBAL_OFFSET:              return "Invalid global offset";
        case CL_INVALID_EVENT_WAIT_LIST:            return "Invalid event wait list";
        case CL_INVALID_EVENT:                      return "Invalid event";
        case CL_INVALID_OPERATION:                  return "Invalid operation";
        case CL_INVALID_GL_OBJECT:                  return "Invalid OpenGL object";
        case CL_INVALID_BUFFER_SIZE:                return "Invalid buffer size";
        case CL_INVALID_MIP_LEVEL:                  return "Invalid mip-map level";
        default: return "Unknown";
    }
}

void CloverCL::dumpBinary() {

    cl_int err;

    const std::string binary_name = "cloverleaf_ocl_binary";

    printf("Dumping binary to %s:\n", binary_name.c_str());
    try {
        unsigned int ndevices;
        program.getInfo(CL_PROGRAM_NUM_DEVICES, &ndevices);
        //printf(" ndevices in dumpBinary = %d\n", ndevices);
        
        std::vector<size_t> sizes = std::vector<size_t>(ndevices);
        program.getInfo(CL_PROGRAM_BINARY_SIZES, &sizes);
        //printf("DumpBinary sizes.size() = %d, sizes[0] = %d\n", sizes.size(), sizes[0]);
        
        std::vector<char*> binaries = std::vector<char*>(ndevices);
        binaries[0] = new char[sizes[0]];
        program.getInfo(CL_PROGRAM_BINARIES, &binaries);
        
        //printf("Binary:\n%s\n", binaries[0]);
        FILE* file = fopen(binary_name.c_str(), "wb");
        fwrite(binaries[0], sizes[0], sizeof(char), file);
        fclose(file);
        
        delete binaries[0];
        
        return;
        #if 0
        std::vector<size_t> sizes
        program.getInfo(CL_PROGRAM_BINARY_SIZES);
        assert(sizes.size() == 1);
        printf(" sizes.size() = %ld, sizes[0] = %ld\n", sizes.size(), sizes[0]);
        std::vector<unsigned char*> binaries = program.getInfo<CL_PROGRAM_BINARIES>();
        assert(binaries.size() == 1);
        for (int i = 0; i < binaries.size(); i++) {
            printf("%c", binaries[0][i]);
        }
        #endif
    } catch (cl::Error err) {
        reportError(err, "Dumping Binary");
    }
    
}

void CloverCL::print_profile_stats() {

#if PROFILE_OCL_KERNELS
    std::cout << "OpenCL [PROFILING] Information: " << std::endl;
    std::cout << "" << std::endl;
    
    std::cout << "[PROFILING] Total Kernel times: " << std::endl;
    std::cout << "Accelerate kernel      : " << accelerate_time*CloverCL::US_TO_SECONDS << " seconds (host time)" << std::endl;
    std::cout << "Advec Cell kernel      : " << advec_cell_time*CloverCL::US_TO_SECONDS << " seconds (host time)" << std::endl;
    std::cout << "Advec Mom kernel       : " << advec_mom_time*CloverCL::US_TO_SECONDS << " seconds (host time)" << std::endl;
    std::cout << "Calc dt kernel         : " << calc_dt_time*CloverCL::US_TO_SECONDS << " seconds (host time)" << std::endl;
    std::cout << "Comms kernel           : " << comms_buffers_time*CloverCL::US_TO_SECONDS << " seconds (host time)" << std::endl;
    std::cout << "Field Summary kernel   : " << field_summ_time*CloverCL::US_TO_SECONDS << " seconds (host time)" << std::endl;
    std::cout << "Flux Calc kernel       : " << flux_calc_time*CloverCL::US_TO_SECONDS << " seconds (host time)" << std::endl;
    //std::cout << "Generate Chunk kernel  : " << generate_chunk_time*CloverCL::US_TO_SECONDS << " seconds (host time)" << std::endl;
    std::cout << "Ideal Gas kernel       : " << ideal_gas_time*CloverCL::US_TO_SECONDS << " seconds (host time)" << std::endl;
    //std::cout << "Initialise Chunk kernel: " << initialise_chunk_time*CloverCL::US_TO_SECONDS << " seconds (host time)" << std::endl;
    std::cout << "PdV kernel             : " << pdv_time*CloverCL::US_TO_SECONDS << " seconds (host time)" << std::endl;
    std::cout << "Reset Field kernel     : " << reset_field_time*CloverCL::US_TO_SECONDS << " seconds (host time)" << std::endl;
    std::cout << "Revert kernel          : " << revert_time*CloverCL::US_TO_SECONDS << " seconds (host time)" << std::endl;
    std::cout << "Update Halo kernel     : " << udpate_halo_time*CloverCL::US_TO_SECONDS << " seconds (host time)" << std::endl;
    std::cout << "Viscosity kernel       : " << viscosity_time*CloverCL::US_TO_SECONDS << " seconds (host time)" << std::endl;

    std::cout << "" << std::endl;
    std::cout << "[PROFILING] Average Kernel times: " << std::endl;
    std::cout << "Accelerate kernel      : " << accelerate_time/accelerate_count*CloverCL::US_TO_SECONDS 
              << " seconds (host time)" << std::endl;
    std::cout << "Advec Cell kernel      : " << advec_cell_time/advec_cell_count*CloverCL::US_TO_SECONDS 
              << " seconds (host time)" << std::endl;
    std::cout << "Advec Mom kernel       : " << advec_mom_time/advec_mom_count*CloverCL::US_TO_SECONDS 
              << " seconds (host time)" << std::endl;
    std::cout << "Calc dt kernel         : " << calc_dt_time/calc_dt_count*CloverCL::US_TO_SECONDS 
              << " seconds (host time)" << std::endl;
    std::cout << "Comms kernel           : " << comms_buffers_time/std::max(1.0,comms_buffers_count)*CloverCL::US_TO_SECONDS 
              << " seconds (host time)" << std::endl;
    std::cout << "Field Summary kernel   : " << field_summ_time/field_summ_count*CloverCL::US_TO_SECONDS 
              << " seconds (host time)" << std::endl;
    std::cout << "Flux Calc kernel       : " << flux_calc_time/flux_calc_count*CloverCL::US_TO_SECONDS 
              << " seconds (host time)" << std::endl;
    //std::cout << "Generate Chunk kernel  : " << generate_chunk_time/std::max(1.0,generate_chunk_count)*CloverCL::US_TO_SECONDS 
    //          << " seconds (host time)" << std::endl;
    std::cout << "Ideal Gas kernel       : " << ideal_gas_time/ideal_gas_count*CloverCL::US_TO_SECONDS 
              << " seconds (host time)" << std::endl;
    //std::cout << "Initialise Chunk kernel: " << initialise_chunk_time/std::max(1.0,initialise_chunk_count)*CloverCL::US_TO_SECONDS 
    //          << " seconds (host time)" << std::endl;
    std::cout << "PdV kernel             : " << pdv_time/pdv_count*CloverCL::US_TO_SECONDS 
              << " seconds (host time)" << std::endl;
    std::cout << "Reset Field kernel     : " << reset_field_time/reset_field_count*CloverCL::US_TO_SECONDS 
              << " seconds (host time)" << std::endl;
    std::cout << "Revert kernel          : " << revert_time/revert_count*CloverCL::US_TO_SECONDS 
              << " seconds (host time)" << std::endl;
    std::cout << "Update Halo kernel     : " << udpate_halo_time/udpate_halo_count*CloverCL::US_TO_SECONDS 
              << " seconds (host time)" << std::endl;
    std::cout << "Viscosity kernel       : " << viscosity_time/viscosity_count*CloverCL::US_TO_SECONDS 
              << " seconds (host time)" << std::endl;
#endif
}

void CloverCL::zero_profiling_timers()
{
#if PROFILE_OCL_KERNELS
    accelerate_time = 0;
    advec_cell_time = 0;
    advec_mom_time = 0;
    calc_dt_time = 0;
    comms_buffers_time = 0;
    field_summ_time = 0;
    flux_calc_time = 0;
    generate_chunk_time = 0;
    ideal_gas_time = 0;
    initialise_chunk_time = 0;
    pdv_time = 0;
    reset_field_time = 0;
    revert_time = 0;
    udpate_halo_time = 0;
    viscosity_time = 0;
    
    accelerate_count = 0;
    advec_cell_count = 0;
    advec_mom_count = 0;
    calc_dt_count = 0;
    comms_buffers_count = 0;
    field_summ_count = 0;
    flux_calc_count = 0;
    generate_chunk_count = 0;
    ideal_gas_count = 0;
    initialise_chunk_count = 0;
    pdv_count = 0;
    reset_field_count = 0;
    revert_count = 0;
    udpate_halo_count = 0;
    viscosity_count = 0;
#endif
}
