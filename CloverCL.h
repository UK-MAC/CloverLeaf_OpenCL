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
 *  @brief CloverCL static class header file.
 *  @author Andrew Mallinson, David Beckingsale
 *  @details Contains common functionality required by all OCL kernels 
*/

#ifndef CLOVER_CL_H_
#define CLOVER_CL_H_

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <string>

/** 
 * @class CloverCL
 *
 * Class to wrap OpenCL functions, providing consistent access to devices,
 * contexts, kernels etc. throughout the code.
 */
class CloverCL { 
    public: 
        CloverCL(); 
        virtual ~CloverCL ();

        static int const fixed_wg_min_size_large_dim   = WG_SIZE_X; // x value passed in by preprocessor 
        static int const fixed_wg_min_size_small_dim   = WG_SIZE_Y; // y value passed in by preprocessor 
        static int xmax_plusfour_rounded;
        static int xmax_plusfive_rounded;
        static int ymax_plusfour_rounded;
        static int ymax_plusfive_rounded;

        static bool initialised;

        static cl::Platform platform;
        static cl::Context context;
        static cl::Device device;
        static cl::CommandQueue queue;
        static cl::CommandQueue outoforder_queue;
        static cl::Program program;

        static int const chunk_left   = 1;
        static int const chunk_right  = 2;
        static int const chunk_bottom = 3;
        static int const chunk_top    = 4;
        static int const external_face=-1;

        static int const field_density0   = 1;
        static int const field_density1   = 2;
        static int const field_energy0    = 3;
        static int const field_energy1    = 4;
        static int const field_pressure   = 5;
        static int const field_viscosity  = 6;
        static int const field_soundspeed = 7;
        static int const field_xvel0      = 8;
        static int const field_xvel1      = 9;
        static int const field_yvel0      =10;
        static int const field_yvel1      =11;
        static int const field_vol_flux_x =12;
        static int const field_vol_flux_y =13;
        static int const field_mass_flux_x=14;
        static int const field_mass_flux_y=15;
        static int const num_fields       =15;

        static const double NS_TO_SECONDS = 1e-9;
        static const double US_TO_SECONDS = 1e-6;

        static const int FIELD_DENSITY0   = 1;
        static const int FIELD_DENSITY1   = 2;
        static const int FIELD_ENERGY0    = 3;
        static const int FIELD_ENERGY1    = 4;
        static const int FIELD_PRESSURE   = 5;
        static const int FIELD_VISCOSITY  = 6;
        static const int FIELD_SOUNDSPEED = 7;
        static const int FIELD_XVEL0      = 8;
        static const int FIELD_XVEL1      = 9;
        static const int FIELD_YVEL0      =10;
        static const int FIELD_YVEL1      =11;
        static const int FIELD_VOL_FLUX_X =12;
        static const int FIELD_VOL_FLUX_Y =13;
        static const int FIELD_MASS_FLUX_X=14;
        static const int FIELD_MASS_FLUX_Y=15;

        static size_t prefer_wg_multiple;
        static size_t max_reduction_wg_size;
        static cl_uint device_procs;
        static size_t device_max_wg_size;
        static cl_ulong device_local_mem_size;
        static cl_device_type device_type;

        static int number_of_red_levels;
        static cl::Event last_event;

        static int mpi_rank; 
        static int xmax_c;
        static int ymax_c;

        static void init(
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
                double dtdiv_safe);

        static void determineWorkGroupSizeInfo();
        static void calculateKernelLaunchParams(int x_max, int y_max);

        static void allocateReductionInterBuffers();

        static void allocateLocalMemoryObjects();

        static void calculateReductionStructure( int xmax, int ymax);

        static void build_reduction_kernel_objects();

        static void printDeviceInformation();

        static void initPlatform(std::string name);

        static void initContext(std::string preferred_type);

        static void initDevice(int id);

        static void initCommandQueue();

        static void loadProgram(int xmin, int xmax,
                                int ymin, int ymax);

        static void createBuffers( int x_max, int y_max, int num_states);

        static void checkErr( cl_int err, std::string name);

        static void reportError( cl::Error err, std::string message);

        static void readVisualisationBuffers(
                int x_max,
                int y_max,
                double* vertexx,
                double* vertexy,
                double* density0,
                double* energy0,
                double* pressure,
                double* viscosity,
                double* xvel0,
                double* yvel0);

        static void readCommunicationBuffer(
                int* xmin,
                int* xmax,
                int* ymin,
                int* ymax,
                int* depth,
                int* xinc,
                int* yinc,
                int* field_name,
                double* buffer,
                int* which_edge);

        static void writeCommunicationBuffer(
                int* xmin,
                int* xmax,
                int* ymin,
                int* ymax,
                int* depth,
                int* xinc,
                int* yinc,
                int* field_name,
                double* buffer,
                int* which_edge);

        static void readAllCommunicationBuffers(
                int* x_max,
                int* y_max,
                double* denisty0,
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
                double* mass_flux_y);

        static void writeAllCommunicationBuffers(
                int* x_max,
                int* y_max,
                double* denisty0,
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
                double* mass_flux_y);

        static void enqueueKernel_nooffsets( cl::Kernel kernel, int num_x, int num_y);

        static void enqueueKernel( cl::Kernel kernel, int x_min, int x_max, int y_min, int y_max);

        static void enqueueKernel( cl::Kernel kernel, int min, int max);

        static void initialiseKernelArgs(
                int x_min,
                int x_max,
                int y_min,
                int y_max,
                double g_small,
                double g_big,
                double dtmin,
                double dtc_safe,
                double dtu_safe,
                double dtv_safe,
                double dtdiv_safe);

        static std::string errToString(cl_int err);

        static void read_back_all_ocl_buffers(double* density0, double* density1, double* energy0, double* energy1,
                                         double* pressure, double* viscosity, double* soundspeed,
                                         double* xvel0, double* xvel1, double* yvel0, double* yvel1,
                                         double* vol_flux_x, double* mass_flux_x,
                                         double* vol_flux_y, double* mass_flux_y,
                                         double* celldx, double* celldy, double* volume ); 

        static void write_back_all_ocl_buffers(double* density0, double* density1, double* energy0, double* energy1,
                                         double* pressure, double* viscosity, double* soundspeed,
                                         double* xvel0, double* xvel1, double* yvel0, double* yvel1,
                                         double* vol_flux_x, double* mass_flux_x,
                                         double* vol_flux_y, double* mass_flux_y,
                                         double* celldx, double* celldy, double* volume ); 

        static void dumpBinary();

        static cl::Buffer density0_buffer;
        static cl::Buffer density1_buffer;
        static cl::Buffer energy0_buffer;
        static cl::Buffer energy1_buffer;
        static cl::Buffer pressure_buffer;
        static cl::Buffer soundspeed_buffer;
        static cl::Buffer celldx_buffer;
        static cl::Buffer celldy_buffer;
        static cl::Buffer viscosity_buffer;
        static cl::Buffer xvel0_buffer;
        static cl::Buffer yvel0_buffer;
        static cl::Buffer xvel1_buffer;
        static cl::Buffer yvel1_buffer;
        static cl::Buffer vol_flux_x_buffer;
        static cl::Buffer vol_flux_y_buffer;
        static cl::Buffer mass_flux_x_buffer;
        static cl::Buffer mass_flux_y_buffer;

        static cl::Buffer volume_buffer;
        static cl::Buffer vertexdx_buffer;
        static cl::Buffer vertexx_buffer;
        static cl::Buffer vertexdy_buffer;
        static cl::Buffer vertexy_buffer;
        static cl::Buffer cellx_buffer;
        static cl::Buffer celly_buffer;
        static cl::Buffer xarea_buffer;
        static cl::Buffer yarea_buffer;

        static cl::Buffer dt_min_val_buffer;
        static cl::Buffer vol_sum_val_buffer;
        static cl::Buffer mass_sum_val_buffer;
        static cl::Buffer ie_sum_val_buffer;
        static cl::Buffer ke_sum_val_buffer;
        static cl::Buffer press_sum_val_buffer;

        static cl::Buffer state_density_buffer;
        static cl::Buffer state_energy_buffer;
        static cl::Buffer state_xvel_buffer;
        static cl::Buffer state_yvel_buffer;
        static cl::Buffer state_xmin_buffer;
        static cl::Buffer state_xmax_buffer;
        static cl::Buffer state_ymin_buffer;
        static cl::Buffer state_ymax_buffer;
        static cl::Buffer state_radius_buffer;
        static cl::Buffer state_geometry_buffer;

        static cl::Buffer top_send_buffer;
        static cl::Buffer top_recv_buffer;
        static cl::Buffer bottom_send_buffer;
        static cl::Buffer bottom_recv_buffer;
        static cl::Buffer left_send_buffer;
        static cl::Buffer left_recv_buffer;
        static cl::Buffer right_send_buffer;
        static cl::Buffer right_recv_buffer;

        static cl::Buffer cpu_min_red_buffer;
        static cl::Buffer cpu_vol_red_buffer;
        static cl::Buffer cpu_mass_red_buffer;
        static cl::Buffer cpu_ie_red_buffer;
        static cl::Buffer cpu_ke_red_buffer;
        static cl::Buffer cpu_press_red_buffer;

        static cl::Buffer work_array1_buffer;
        static cl::Buffer work_array2_buffer;
        static cl::Buffer work_array3_buffer;
        static cl::Buffer work_array4_buffer;
        static cl::Buffer work_array5_buffer;
        static cl::Buffer work_array6_buffer;
        static cl::Buffer work_array7_buffer;

        static cl::Kernel ideal_gas_predict_knl;
        static cl::Kernel ideal_gas_NO_predict_knl;
        static cl::Kernel viscosity_knl;
        static cl::Kernel flux_calc_knl;
        static cl::Kernel accelerate_knl;

        static cl::Kernel advec_cell_xdir_sec1_s1_knl;
        static cl::Kernel advec_cell_xdir_sec1_s2_knl;
        static cl::Kernel advec_cell_xdir_sec2_knl;
        static cl::Kernel advec_cell_xdir_sec3_knl;
        static cl::Kernel advec_cell_ydir_sec1_s1_knl;
        static cl::Kernel advec_cell_ydir_sec1_s2_knl;
        static cl::Kernel advec_cell_ydir_sec2_knl;
        static cl::Kernel advec_cell_ydir_sec3_knl;

        static cl::Kernel advec_mom_vol_knl;
        static cl::Kernel advec_mom_node_mass_pre_x_knl;
        static cl::Kernel advec_mom_node_x_knl;
        static cl::Kernel advec_mom_flux_x_vec1_knl;
        static cl::Kernel advec_mom_flux_x_vecnot1_knl;
        static cl::Kernel advec_mom_vel_x_knl;
        static cl::Kernel advec_mom_node_y_knl;
        static cl::Kernel advec_mom_node_mass_pre_y_knl;
        static cl::Kernel advec_mom_flux_y_vec1_knl;
        static cl::Kernel advec_mom_flux_y_vecnot1_knl;
        static cl::Kernel advec_mom_vel_y_knl;

        static cl::Kernel dt_calc_knl;

        static cl::Kernel minimum_red_knl;
        static cl::Kernel vol_sum_red_knl;
        static cl::Kernel mass_sum_red_knl;
        static cl::Kernel ie_sum_red_knl;
        static cl::Kernel ke_sum_red_knl;
        static cl::Kernel press_sum_red_knl;

        static cl::Kernel minimum_red_last_knl;
        static cl::Kernel vol_sum_red_last_knl;
        static cl::Kernel mass_sum_red_last_knl;
        static cl::Kernel ie_sum_red_last_knl;
        static cl::Kernel ke_sum_red_last_knl;
        static cl::Kernel press_sum_red_last_knl;

        static cl::Kernel pdv_correct_knl;
        static cl::Kernel pdv_predict_knl;
        static cl::Kernel reset_field_knl;
        static cl::Kernel revert_knl;

        static cl::Kernel generate_chunk_knl;
        static cl::Kernel initialise_chunk_cell_x_knl;
        static cl::Kernel initialise_chunk_cell_y_knl;
        static cl::Kernel initialise_chunk_vertex_x_knl;
        static cl::Kernel initialise_chunk_vertex_y_knl;
        static cl::Kernel initialise_chunk_volume_area_knl;
        static cl::Kernel field_summary_knl;

        static cl::Kernel update_halo_left_cell_knl;
        static cl::Kernel update_halo_right_cell_knl;
        static cl::Kernel update_halo_top_cell_knl;
        static cl::Kernel update_halo_bottom_cell_knl;
        static cl::Kernel update_halo_left_vel_knl;
        static cl::Kernel update_halo_right_vel_knl;
        static cl::Kernel update_halo_top_vel_knl;
        static cl::Kernel update_halo_bottom_vel_knl;
        static cl::Kernel update_halo_left_flux_x_knl;
        static cl::Kernel update_halo_right_flux_x_knl;
        static cl::Kernel update_halo_top_flux_x_knl;
        static cl::Kernel update_halo_bottom_flux_x_knl;
        static cl::Kernel update_halo_left_flux_y_knl;
        static cl::Kernel update_halo_right_flux_y_knl;
        static cl::Kernel update_halo_top_flux_y_knl;
        static cl::Kernel update_halo_bottom_flux_y_knl;

        static cl::Kernel read_top_buffer_knl;
        static cl::Kernel read_right_buffer_knl;
        static cl::Kernel read_bottom_buffer_knl;
        static cl::Kernel read_left_buffer_knl;

        static cl::Kernel write_top_buffer_knl;
        static cl::Kernel write_right_buffer_knl;
        static cl::Kernel write_bottom_buffer_knl;
        static cl::Kernel write_left_buffer_knl;

        static cl::Kernel minimum_red_cpu_knl;
        static cl::Kernel vol_sum_red_cpu_knl; 
        static cl::Kernel mass_sum_red_cpu_knl; 
        static cl::Kernel ie_sum_red_cpu_knl; 
        static cl::Kernel ke_sum_red_cpu_knl; 
        static cl::Kernel press_sum_red_cpu_knl; 

        static std::vector<cl::Kernel> min_reduction_kernels;
        static std::vector<cl::Kernel> vol_sum_reduction_kernels;
        static std::vector<cl::Kernel> mass_sum_reduction_kernels;
        static std::vector<cl::Kernel> ie_sum_reduction_kernels;
        static std::vector<cl::Kernel> ke_sum_reduction_kernels;
        static std::vector<cl::Kernel> press_sum_reduction_kernels;

        static std::vector<int> num_workitems_tolaunch;
        static std::vector<int> num_workitems_per_wg;
        static std::vector<int> local_mem_size;
        static std::vector<int> size_limits;
        static std::vector<int> buffer_sizes;
        static std::vector<bool> input_even;
        static std::vector<int> num_elements_per_wi;

        static std::vector<cl::Buffer> min_interBuffers;
        static std::vector<cl::Buffer> vol_interBuffers;
        static std::vector<cl::Buffer> mass_interBuffers;
        static std::vector<cl::Buffer> ie_interBuffers;
        static std::vector<cl::Buffer> ke_interBuffers;
        static std::vector<cl::Buffer> press_interBuffers;

        static std::vector<cl::LocalSpaceArg> min_local_memory_objects;
        static std::vector<cl::LocalSpaceArg> vol_local_memory_objects;
        static std::vector<cl::LocalSpaceArg> mass_local_memory_objects;
        static std::vector<cl::LocalSpaceArg> ie_local_memory_objects;
        static std::vector<cl::LocalSpaceArg> ke_local_memory_objects;
        static std::vector<cl::LocalSpaceArg> press_local_memory_objects;

    private:
        static std::vector<cl::Event> global_events;
};

#endif
