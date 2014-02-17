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
 *  @brief OCL host-side update halo kernel.
 *  @author Andrew Mallinson, David Beckingsale
 *  @details Launches the OCL device-side update halo kernel 
*/

#include "CloverCL.h"
#include "common_macs.h"

#include <sys/time.h>

#define ARRAY1D(i_index,i_lb) ((i_index)-(i_lb))

extern "C" void update_halo_kernel_ocl_(int *xmin, int *xmax,
                                        int *ymin, int *ymax,
                                        int *left, int *bottom,
                                        int *right, int *top,
                                        int *leftboundary, int *bottomboundary,
                                        int *rightboundary, int *topboundary,
                                        int *chunk_neighbours, int *fields,
                                        int *depth);

void update_halo_kernel_ocl_(int *xmin, int *xmax,
                             int *ymin, int *ymax,
                             int *left, int *bottom,
                             int *right, int *top,
                             int *leftboundary, int *bottomboundary,
                             int *rightboundary, int *topboundary,
                             int *chunk_neighbours, int *fields,
                             int *depth)
{
    cl_int err;

    std::vector<cl::Event> events2;

#if PROFILE_OCL_KERNELS
    timeval t_start;
    gettimeofday(&t_start, NULL);
#endif

    int uh_knl_launch_small_dim; 
    if (*depth == 2) {
        uh_knl_launch_small_dim = UH_SMALL_DIM_DEPTHTWO;
    }
    else {
        uh_knl_launch_small_dim = 1;
    }

    /* Perform the halo updates for the top and bottom faces in parallel */

    events2.push_back(CloverCL::last_event);
    CloverCL::outoforder_queue.enqueueWaitForEvents(events2);

    if ( fields[ARRAY1D(CloverCL::field_density0,1)] == 1 ) {
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_bottom,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_bottom_cell_knl.setArg(0, *depth);
                CloverCL::update_halo_bottom_cell_knl.setArg(1, CloverCL::density0_buffer);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_bottom_cell_knl,
                                         CloverCL::xmax_plusfour_rounded,*depth,
                                         CloverCL::fixed_wg_min_size_large_dim,uh_knl_launch_small_dim);

            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo density0 running bottom knl");
            }
        }
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_top,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_top_cell_knl.setArg(0, *depth);
                CloverCL::update_halo_top_cell_knl.setArg(1, CloverCL::density0_buffer);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_top_cell_knl,
                                         CloverCL::xmax_plusfour_rounded,*depth,
                                         CloverCL::fixed_wg_min_size_large_dim,uh_knl_launch_small_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo density0 running top knl");
            }
        }
    }

    if ( fields[ARRAY1D(CloverCL::field_density1,1)] == 1 ) {
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_bottom,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_bottom_cell_knl.setArg(0, *depth);
                CloverCL::update_halo_bottom_cell_knl.setArg(1, CloverCL::density1_buffer);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_bottom_cell_knl,
                                         CloverCL::xmax_plusfour_rounded,*depth,
                                         CloverCL::fixed_wg_min_size_large_dim,uh_knl_launch_small_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo density1 running bottom knl");
            }
        }
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_top,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_top_cell_knl.setArg(0, *depth);
                CloverCL::update_halo_top_cell_knl.setArg(1, CloverCL::density1_buffer);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_top_cell_knl,
                                         CloverCL::xmax_plusfour_rounded,*depth,
                                         CloverCL::fixed_wg_min_size_large_dim,uh_knl_launch_small_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo density1 running top knl");
            }
        }
    }


    if ( fields[ARRAY1D(CloverCL::field_energy0,1)] == 1 ) {
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_bottom,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_bottom_cell_knl.setArg(0, *depth);
                CloverCL::update_halo_bottom_cell_knl.setArg(1, CloverCL::energy0_buffer);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_bottom_cell_knl,
                                         CloverCL::xmax_plusfour_rounded,*depth,
                                         CloverCL::fixed_wg_min_size_large_dim,uh_knl_launch_small_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo energy0 running bottom knl");
            }
        }
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_top,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_top_cell_knl.setArg(0, *depth);
                CloverCL::update_halo_top_cell_knl.setArg(1, CloverCL::energy0_buffer);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_top_cell_knl,
                                         CloverCL::xmax_plusfour_rounded,*depth,
                                         CloverCL::fixed_wg_min_size_large_dim,uh_knl_launch_small_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo energy0 running top knl");
            }
        }
    }


    if ( fields[ARRAY1D(CloverCL::field_energy1,1)] == 1 ) {
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_bottom,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_bottom_cell_knl.setArg(0, *depth);
                CloverCL::update_halo_bottom_cell_knl.setArg(1, CloverCL::energy1_buffer);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_bottom_cell_knl,
                                         CloverCL::xmax_plusfour_rounded,*depth,
                                         CloverCL::fixed_wg_min_size_large_dim,uh_knl_launch_small_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo energy1 running bottom knl");
            }
        }
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_top,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_top_cell_knl.setArg(0, *depth);
                CloverCL::update_halo_top_cell_knl.setArg(1, CloverCL::energy1_buffer);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_top_cell_knl,
                                         CloverCL::xmax_plusfour_rounded,*depth,
                                         CloverCL::fixed_wg_min_size_large_dim,uh_knl_launch_small_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo energy1 running top knl");
            }
        }
    }


    if ( fields[ARRAY1D(CloverCL::field_pressure,1)] == 1 ) {
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_bottom,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_bottom_cell_knl.setArg(0, *depth);
                CloverCL::update_halo_bottom_cell_knl.setArg(1, CloverCL::pressure_buffer);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_bottom_cell_knl,
                                         CloverCL::xmax_plusfour_rounded,*depth,
                                         CloverCL::fixed_wg_min_size_large_dim,uh_knl_launch_small_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo pressure running bottom knl");
            }
        }
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_top,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_top_cell_knl.setArg(0, *depth);
                CloverCL::update_halo_top_cell_knl.setArg(1, CloverCL::pressure_buffer);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_top_cell_knl,
                                         CloverCL::xmax_plusfour_rounded,*depth,
                                         CloverCL::fixed_wg_min_size_large_dim,uh_knl_launch_small_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo pressure running top knl");
            }
        }
    }

    if ( fields[ARRAY1D(CloverCL::field_viscosity,1)] == 1 ) {
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_bottom,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_bottom_cell_knl.setArg(0, *depth);
                CloverCL::update_halo_bottom_cell_knl.setArg(1, CloverCL::viscosity_buffer);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_bottom_cell_knl,
                                         CloverCL::xmax_plusfour_rounded,*depth,
                                         CloverCL::fixed_wg_min_size_large_dim,uh_knl_launch_small_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo viscosity running bottom knl");
            }
        }
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_top,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_top_cell_knl.setArg(0, *depth);
                CloverCL::update_halo_top_cell_knl.setArg(1, CloverCL::viscosity_buffer);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_top_cell_knl,
                                         CloverCL::xmax_plusfour_rounded,*depth,
                                         CloverCL::fixed_wg_min_size_large_dim,uh_knl_launch_small_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo viscosity running top knl");
            }
        }
    }

    if ( fields[ARRAY1D(CloverCL::field_soundspeed,1)] == 1 ) {
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_bottom,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_bottom_cell_knl.setArg(0, *depth);
                CloverCL::update_halo_bottom_cell_knl.setArg(1, CloverCL::soundspeed_buffer);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_bottom_cell_knl,
                                         CloverCL::xmax_plusfour_rounded,*depth,
                                         CloverCL::fixed_wg_min_size_large_dim,uh_knl_launch_small_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo soundspeed running bottom knl");
            }
        }
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_top,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_top_cell_knl.setArg(0, *depth);
                CloverCL::update_halo_top_cell_knl.setArg(1, CloverCL::soundspeed_buffer);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_top_cell_knl,
                                         CloverCL::xmax_plusfour_rounded,*depth,
                                         CloverCL::fixed_wg_min_size_large_dim,uh_knl_launch_small_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo soundspeed running top knl");
            }
        }
    }

    if ( fields[ARRAY1D(CloverCL::field_xvel0,1)] == 1 ) {
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_bottom,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_bottom_vel_knl.setArg(0, *depth);
                CloverCL::update_halo_bottom_vel_knl.setArg(1, CloverCL::xvel0_buffer);
                CloverCL::update_halo_bottom_vel_knl.setArg(2, 1);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_bottom_vel_knl,
                                         CloverCL::xmax_plusfive_rounded,*depth,
                                         CloverCL::fixed_wg_min_size_large_dim,uh_knl_launch_small_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo xvel0 running bottom knl");
            }
        }
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_top,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_top_vel_knl.setArg(0, *depth);
                CloverCL::update_halo_top_vel_knl.setArg(1, CloverCL::xvel0_buffer);
                CloverCL::update_halo_top_vel_knl.setArg(2, 1);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_top_vel_knl,
                                         CloverCL::xmax_plusfive_rounded,*depth,
                                         CloverCL::fixed_wg_min_size_large_dim,uh_knl_launch_small_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo xvel0 running top knl");
            }
        }
    }

    if ( fields[ARRAY1D(CloverCL::field_xvel1,1)] == 1 ) {
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_bottom,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_bottom_vel_knl.setArg(0, *depth);
                CloverCL::update_halo_bottom_vel_knl.setArg(1, CloverCL::xvel1_buffer);
                CloverCL::update_halo_bottom_vel_knl.setArg(2, 1);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_bottom_vel_knl,
                                         CloverCL::xmax_plusfive_rounded,*depth,
                                         CloverCL::fixed_wg_min_size_large_dim,uh_knl_launch_small_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo ? running bottom knl");
            }
        }
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_top,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_top_vel_knl.setArg(0, *depth);
                CloverCL::update_halo_top_vel_knl.setArg(1, CloverCL::xvel1_buffer);
                CloverCL::update_halo_top_vel_knl.setArg(2, 1);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_top_vel_knl,
                                         CloverCL::xmax_plusfive_rounded,*depth,
                                         CloverCL::fixed_wg_min_size_large_dim,uh_knl_launch_small_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo ? running top knl");
            }
        }
    }

    if ( fields[ARRAY1D(CloverCL::field_yvel0,1)] == 1 ) {
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_bottom,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_bottom_vel_knl.setArg(0, *depth);
                CloverCL::update_halo_bottom_vel_knl.setArg(1, CloverCL::yvel0_buffer);
                CloverCL::update_halo_bottom_vel_knl.setArg(2, -1);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_bottom_vel_knl,
                                         CloverCL::xmax_plusfive_rounded,*depth,
                                         CloverCL::fixed_wg_min_size_large_dim,uh_knl_launch_small_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo yvel0 running bottom knl");
            }
        }
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_top,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_top_vel_knl.setArg(0, *depth);
                CloverCL::update_halo_top_vel_knl.setArg(1, CloverCL::yvel0_buffer);
                CloverCL::update_halo_top_vel_knl.setArg(2, -1);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_top_vel_knl,
                                         CloverCL::xmax_plusfive_rounded,*depth,
                                         CloverCL::fixed_wg_min_size_large_dim,uh_knl_launch_small_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo yvel0 running top knl");
            }
        }
    }

    if ( fields[ARRAY1D(CloverCL::field_yvel1,1)] == 1 ) {
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_bottom,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_bottom_vel_knl.setArg(0, *depth);
                CloverCL::update_halo_bottom_vel_knl.setArg(1, CloverCL::yvel1_buffer);
                CloverCL::update_halo_bottom_vel_knl.setArg(2, -1);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_bottom_vel_knl,
                                         CloverCL::xmax_plusfive_rounded,*depth,
                                         CloverCL::fixed_wg_min_size_large_dim,uh_knl_launch_small_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo yvel1 running bottom knl");
            }
        }
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_top,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_top_vel_knl.setArg(0, *depth);
                CloverCL::update_halo_top_vel_knl.setArg(1, CloverCL::yvel1_buffer);
                CloverCL::update_halo_top_vel_knl.setArg(2, -1);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_top_vel_knl,
                                         CloverCL::xmax_plusfive_rounded,*depth,
                                         CloverCL::fixed_wg_min_size_large_dim,uh_knl_launch_small_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo yvel1 running top knl");
            }
        }
    }

    if ( fields[ARRAY1D(CloverCL::field_vol_flux_x,1)] == 1 ) {
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_bottom,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_bottom_flux_x_knl.setArg(0, *depth);
                CloverCL::update_halo_bottom_flux_x_knl.setArg(1, CloverCL::vol_flux_x_buffer);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_bottom_flux_x_knl,
                                         CloverCL::xmax_plusfive_rounded,*depth,
                                         CloverCL::fixed_wg_min_size_large_dim,uh_knl_launch_small_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo vol_flux_x running bottom knl");
            }
        }
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_top,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_top_flux_x_knl.setArg(0, *depth);
                CloverCL::update_halo_top_flux_x_knl.setArg(1, CloverCL::vol_flux_x_buffer);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_top_flux_x_knl,
                                         CloverCL::xmax_plusfive_rounded,*depth,
                                         CloverCL::fixed_wg_min_size_large_dim,uh_knl_launch_small_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo vol_flux_x running top knl");
            }
        }
    }

    if ( fields[ARRAY1D(CloverCL::field_vol_flux_y,1)] == 1 ) {
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_bottom,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_bottom_flux_y_knl.setArg(0, *depth);
                CloverCL::update_halo_bottom_flux_y_knl.setArg(1, CloverCL::vol_flux_y_buffer);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_bottom_flux_y_knl,
                                         CloverCL::xmax_plusfour_rounded,*depth,
                                         CloverCL::fixed_wg_min_size_large_dim,uh_knl_launch_small_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo vol_flux_y running bottom knl");
            }
        }
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_top,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_top_flux_y_knl.setArg(0, *depth);
                CloverCL::update_halo_top_flux_y_knl.setArg(1, CloverCL::vol_flux_y_buffer);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_top_flux_y_knl,
                                         CloverCL::xmax_plusfour_rounded,*depth,
                                         CloverCL::fixed_wg_min_size_large_dim,uh_knl_launch_small_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo vol_flux_y running top knl");
            }
        }
    }

    if ( fields[ARRAY1D(CloverCL::field_mass_flux_x,1)] == 1 ) {
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_bottom,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_bottom_flux_x_knl.setArg(0, *depth);
                CloverCL::update_halo_bottom_flux_x_knl.setArg(1, CloverCL::mass_flux_x_buffer);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_bottom_flux_x_knl,
                                         CloverCL::xmax_plusfive_rounded,*depth,
                                         CloverCL::fixed_wg_min_size_large_dim,uh_knl_launch_small_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo mass_flux_x running bottom knl");
            }
        }
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_top,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_top_flux_x_knl.setArg(0, *depth);
                CloverCL::update_halo_top_flux_x_knl.setArg(1, CloverCL::mass_flux_x_buffer);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_top_flux_x_knl,
                                         CloverCL::xmax_plusfive_rounded,*depth,
                                         CloverCL::fixed_wg_min_size_large_dim,uh_knl_launch_small_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo mass_flux_x running top knl");
            }
        }
    }

    if ( fields[ARRAY1D(CloverCL::field_mass_flux_y,1)] == 1 ) {
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_bottom,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_bottom_flux_y_knl.setArg(0, *depth);
                CloverCL::update_halo_bottom_flux_y_knl.setArg(1, CloverCL::mass_flux_y_buffer);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_bottom_flux_y_knl,
                                         CloverCL::xmax_plusfour_rounded,*depth,
                                         CloverCL::fixed_wg_min_size_large_dim,uh_knl_launch_small_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo mass_flux_y running bottom knl");
            }
        }
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_top,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_top_flux_y_knl.setArg(0, *depth);
                CloverCL::update_halo_top_flux_y_knl.setArg(1, CloverCL::mass_flux_y_buffer);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_top_flux_y_knl,
                                         CloverCL::xmax_plusfour_rounded,*depth,
                                         CloverCL::fixed_wg_min_size_large_dim,uh_knl_launch_small_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo mass_flux_y running top knl");
            }
        }
    }


    /* Sync between top/bottom updates and left/right updates */
    CloverCL::outoforder_queue.enqueueBarrier();


    /* Perform left and right halo updates in parallel  */

    if ( fields[ARRAY1D(CloverCL::field_density0,1)] == 1 ) {
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_left,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_left_cell_knl.setArg(0, *depth);
                CloverCL::update_halo_left_cell_knl.setArg(1, CloverCL::density0_buffer);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_left_cell_knl,
                                         *depth,CloverCL::ymax_plusfour_rounded,
                                         uh_knl_launch_small_dim,CloverCL::fixed_wg_min_size_large_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo density0 running left knl");
            }
        }
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_right,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_right_cell_knl.setArg(0, *depth);
                CloverCL::update_halo_right_cell_knl.setArg(1, CloverCL::density0_buffer);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_right_cell_knl,
                                         *depth,CloverCL::ymax_plusfour_rounded,
                                         uh_knl_launch_small_dim,CloverCL::fixed_wg_min_size_large_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo density0 running right knl");
            }
        }
    }


    if ( fields[ARRAY1D(CloverCL::field_density1,1)] == 1 ) {
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_left,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_left_cell_knl.setArg(0, *depth);
                CloverCL::update_halo_left_cell_knl.setArg(1, CloverCL::density1_buffer);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_left_cell_knl,
                                         *depth,CloverCL::ymax_plusfour_rounded,
                                         uh_knl_launch_small_dim,CloverCL::fixed_wg_min_size_large_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo density1 running left knl");
            }
        }
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_right,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_right_cell_knl.setArg(0, *depth);
                CloverCL::update_halo_right_cell_knl.setArg(1, CloverCL::density1_buffer);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_right_cell_knl,
                                         *depth,CloverCL::ymax_plusfour_rounded,
                                         uh_knl_launch_small_dim,CloverCL::fixed_wg_min_size_large_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo density1 running right knl");
            }
        }
    }


    if ( fields[ARRAY1D(CloverCL::field_energy0,1)] == 1 ) {
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_left,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_left_cell_knl.setArg(0, *depth);
                CloverCL::update_halo_left_cell_knl.setArg(1, CloverCL::energy0_buffer);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_left_cell_knl,
                                         *depth,CloverCL::ymax_plusfour_rounded,
                                         uh_knl_launch_small_dim,CloverCL::fixed_wg_min_size_large_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo energy0 running left knl");
            }
        }
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_right,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_right_cell_knl.setArg(0, *depth);
                CloverCL::update_halo_right_cell_knl.setArg(1, CloverCL::energy0_buffer);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_right_cell_knl,
                                         *depth,CloverCL::ymax_plusfour_rounded,
                                         uh_knl_launch_small_dim,CloverCL::fixed_wg_min_size_large_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo energy0 running right knl");
            }
        }
    }


    if ( fields[ARRAY1D(CloverCL::field_energy1,1)] == 1 ) {
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_left,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_left_cell_knl.setArg(0, *depth);
                CloverCL::update_halo_left_cell_knl.setArg(1, CloverCL::energy1_buffer);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_left_cell_knl,
                                         *depth,CloverCL::ymax_plusfour_rounded,
                                         uh_knl_launch_small_dim,CloverCL::fixed_wg_min_size_large_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo energy1 running left knl");
            }
        }
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_right,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_right_cell_knl.setArg(0, *depth);
                CloverCL::update_halo_right_cell_knl.setArg(1, CloverCL::energy1_buffer);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_right_cell_knl,
                                         *depth,CloverCL::ymax_plusfour_rounded,
                                         uh_knl_launch_small_dim,CloverCL::fixed_wg_min_size_large_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo energy1 running right knl");
            }
        }
    }


    if ( fields[ARRAY1D(CloverCL::field_pressure,1)] == 1 ) {
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_left,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_left_cell_knl.setArg(0, *depth);
                CloverCL::update_halo_left_cell_knl.setArg(1, CloverCL::pressure_buffer);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_left_cell_knl,
                                         *depth,CloverCL::ymax_plusfour_rounded,
                                         uh_knl_launch_small_dim,CloverCL::fixed_wg_min_size_large_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo pressure running left knl");
            }
        }
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_right,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_right_cell_knl.setArg(0, *depth);
                CloverCL::update_halo_right_cell_knl.setArg(1, CloverCL::pressure_buffer);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_right_cell_knl,
                                         *depth,CloverCL::ymax_plusfour_rounded,
                                         uh_knl_launch_small_dim,CloverCL::fixed_wg_min_size_large_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo pressure running right knl");
            }
        }
    }

    if ( fields[ARRAY1D(CloverCL::field_viscosity,1)] == 1 ) {
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_left,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_left_cell_knl.setArg(0, *depth);
                CloverCL::update_halo_left_cell_knl.setArg(1, CloverCL::viscosity_buffer);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_left_cell_knl,
                                         *depth,CloverCL::ymax_plusfour_rounded,
                                         uh_knl_launch_small_dim,CloverCL::fixed_wg_min_size_large_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo viscosity running left knl");
            }
        }
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_right,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_right_cell_knl.setArg(0, *depth);
                CloverCL::update_halo_right_cell_knl.setArg(1, CloverCL::viscosity_buffer);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_right_cell_knl,
                                         *depth,CloverCL::ymax_plusfour_rounded,
                                         uh_knl_launch_small_dim,CloverCL::fixed_wg_min_size_large_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo viscosity running right knl");
            }
        }
    }

    if ( fields[ARRAY1D(CloverCL::field_soundspeed,1)] == 1 ) {
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_left,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_left_cell_knl.setArg(0, *depth);
                CloverCL::update_halo_left_cell_knl.setArg(1, CloverCL::soundspeed_buffer);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_left_cell_knl,
                                         *depth,CloverCL::ymax_plusfour_rounded,
                                         uh_knl_launch_small_dim,CloverCL::fixed_wg_min_size_large_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo soundspeed running left knl");
            }
        }
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_right,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_right_cell_knl.setArg(0, *depth);
                CloverCL::update_halo_right_cell_knl.setArg(1, CloverCL::soundspeed_buffer);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_right_cell_knl,
                                         *depth,CloverCL::ymax_plusfour_rounded,
                                         uh_knl_launch_small_dim,CloverCL::fixed_wg_min_size_large_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo soundspeed running right knl");
            }
        }
    }

    if ( fields[ARRAY1D(CloverCL::field_xvel0,1)] == 1 ) {
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_left,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_left_vel_knl.setArg(0, *depth);
                CloverCL::update_halo_left_vel_knl.setArg(1, CloverCL::xvel0_buffer);
                CloverCL::update_halo_left_vel_knl.setArg(2, -1);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_left_vel_knl,
                                         *depth,CloverCL::ymax_plusfive_rounded,
                                         uh_knl_launch_small_dim,CloverCL::fixed_wg_min_size_large_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo xvel0 running left knl");
            }
        }
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_right,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_right_vel_knl.setArg(0, *depth);
                CloverCL::update_halo_right_vel_knl.setArg(1, CloverCL::xvel0_buffer);
                CloverCL::update_halo_right_vel_knl.setArg(2, -1);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_right_vel_knl,
                                         *depth,CloverCL::ymax_plusfive_rounded,
                                         uh_knl_launch_small_dim,CloverCL::fixed_wg_min_size_large_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo xvel0 running right knl");
            }
        }
    }

    if ( fields[ARRAY1D(CloverCL::field_xvel1,1)] == 1 ) {
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_left,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_left_vel_knl.setArg(0, *depth);
                CloverCL::update_halo_left_vel_knl.setArg(1, CloverCL::xvel1_buffer);
                CloverCL::update_halo_left_vel_knl.setArg(2, -1);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_left_vel_knl,
                                         *depth,CloverCL::ymax_plusfive_rounded,
                                         uh_knl_launch_small_dim,CloverCL::fixed_wg_min_size_large_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo ? running left knl");
            }
        }
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_right,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_right_vel_knl.setArg(0, *depth);
                CloverCL::update_halo_right_vel_knl.setArg(1, CloverCL::xvel1_buffer);
                CloverCL::update_halo_right_vel_knl.setArg(2, -1);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_right_vel_knl,
                                         *depth,CloverCL::ymax_plusfive_rounded,
                                         uh_knl_launch_small_dim,CloverCL::fixed_wg_min_size_large_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo ? running right knl");
            }
        }
    }

    if ( fields[ARRAY1D(CloverCL::field_yvel0,1)] == 1 ) {
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_left,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_left_vel_knl.setArg(0, *depth);
                CloverCL::update_halo_left_vel_knl.setArg(1, CloverCL::yvel0_buffer);
                CloverCL::update_halo_left_vel_knl.setArg(2, 1);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_left_vel_knl,
                                         *depth,CloverCL::ymax_plusfive_rounded,
                                         uh_knl_launch_small_dim,CloverCL::fixed_wg_min_size_large_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo yvel0 running left knl");
            }
        }
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_right,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_right_vel_knl.setArg(0, *depth);
                CloverCL::update_halo_right_vel_knl.setArg(1, CloverCL::yvel0_buffer);
                CloverCL::update_halo_right_vel_knl.setArg(2, 1);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_right_vel_knl,
                                         *depth,CloverCL::ymax_plusfive_rounded,
                                         uh_knl_launch_small_dim,CloverCL::fixed_wg_min_size_large_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo yvel0 running right knl");
            }
        }
    }

    if ( fields[ARRAY1D(CloverCL::field_yvel1,1)] == 1 ) {
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_left,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_left_vel_knl.setArg(0, *depth);
                CloverCL::update_halo_left_vel_knl.setArg(1, CloverCL::yvel1_buffer);
                CloverCL::update_halo_left_vel_knl.setArg(2, 1);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_left_vel_knl,
                                         *depth,CloverCL::ymax_plusfive_rounded,
                                         uh_knl_launch_small_dim,CloverCL::fixed_wg_min_size_large_dim);

            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo yvel1 running left knl");
            }
        }
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_right,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_right_vel_knl.setArg(0, *depth);
                CloverCL::update_halo_right_vel_knl.setArg(1, CloverCL::yvel1_buffer);
                CloverCL::update_halo_right_vel_knl.setArg(2, 1);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_right_vel_knl,
                                         *depth,CloverCL::ymax_plusfive_rounded,
                                         uh_knl_launch_small_dim,CloverCL::fixed_wg_min_size_large_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo yvel1 running right knl");
            }
        }
    }

    if ( fields[ARRAY1D(CloverCL::field_vol_flux_x,1)] == 1 ) {
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_left,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_left_flux_x_knl.setArg(0, *depth);
                CloverCL::update_halo_left_flux_x_knl.setArg(1, CloverCL::vol_flux_x_buffer);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_left_flux_x_knl,
                                         *depth,CloverCL::ymax_plusfour_rounded,
                                         uh_knl_launch_small_dim,CloverCL::fixed_wg_min_size_large_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo vol_flux_x running left knl");
            }
        }
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_right,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_right_flux_x_knl.setArg(0, *depth);
                CloverCL::update_halo_right_flux_x_knl.setArg(1, CloverCL::vol_flux_x_buffer);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_right_flux_x_knl,
                                         *depth,CloverCL::ymax_plusfour_rounded,
                                         uh_knl_launch_small_dim,CloverCL::fixed_wg_min_size_large_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo vol_flux_x running right knl");
            }
        }
    }

    if ( fields[ARRAY1D(CloverCL::field_vol_flux_y,1)] == 1 ) {
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_left,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_left_flux_y_knl.setArg(0, *depth);
                CloverCL::update_halo_left_flux_y_knl.setArg(1, CloverCL::vol_flux_y_buffer);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_left_flux_y_knl,
                                         *depth,CloverCL::ymax_plusfive_rounded,
                                         uh_knl_launch_small_dim,CloverCL::fixed_wg_min_size_large_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo vol_flux_y running left knl");
            }
        }
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_right,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_right_flux_y_knl.setArg(0, *depth);
                CloverCL::update_halo_right_flux_y_knl.setArg(1, CloverCL::vol_flux_y_buffer);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_right_flux_y_knl,
                                         *depth,CloverCL::ymax_plusfive_rounded,
                                         uh_knl_launch_small_dim,CloverCL::fixed_wg_min_size_large_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo vol_flux_y running right knl");
            }
        }
    }

    if ( fields[ARRAY1D(CloverCL::field_mass_flux_x,1)] == 1 ) {
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_left,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_left_flux_x_knl.setArg(0, *depth);
                CloverCL::update_halo_left_flux_x_knl.setArg(1, CloverCL::mass_flux_x_buffer);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_left_flux_x_knl,
                                         *depth,CloverCL::ymax_plusfour_rounded,
                                         uh_knl_launch_small_dim,CloverCL::fixed_wg_min_size_large_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo mass_flux_x running left knl");
            }
        }
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_right,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_right_flux_x_knl.setArg(0, *depth);
                CloverCL::update_halo_right_flux_x_knl.setArg(1, CloverCL::mass_flux_x_buffer);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_right_flux_x_knl,
                                         *depth,CloverCL::ymax_plusfour_rounded,
                                         uh_knl_launch_small_dim,CloverCL::fixed_wg_min_size_large_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo mass_flux_x running right knl");
            }
        }
    }

    if ( fields[ARRAY1D(CloverCL::field_mass_flux_y,1)] == 1 ) {
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_left,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_left_flux_y_knl.setArg(0, *depth);
                CloverCL::update_halo_left_flux_y_knl.setArg(1, CloverCL::mass_flux_y_buffer);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_left_flux_y_knl,
                                         *depth,CloverCL::ymax_plusfive_rounded,
                                         uh_knl_launch_small_dim,CloverCL::fixed_wg_min_size_large_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo mass_flux_y running left knl");
            }
        }
        if (chunk_neighbours[ARRAY1D(CloverCL::chunk_right,1)] == CloverCL::external_face) {
            try {

                CloverCL::update_halo_right_flux_y_knl.setArg(0, *depth);
                CloverCL::update_halo_right_flux_y_knl.setArg(1, CloverCL::mass_flux_y_buffer);

                ENQUEUE_KERNEL_OOO_MACRO(CloverCL::update_halo_right_flux_y_knl,
                                         *depth,CloverCL::ymax_plusfive_rounded,
                                         uh_knl_launch_small_dim,CloverCL::fixed_wg_min_size_large_dim);
            } catch(cl::Error err) {
                CloverCL::reportError(err, "update halo mass_flux_y running right knl");
            }
        }
    }


    /*
     * Wait for all update left and right halo kernels to execute
     */
    CloverCL::outoforder_queue.finish();



    /* Generate profiling info */
#if PROFILE_OCL_KERNELS
    timeval t_end;

    gettimeofday(&t_end, NULL);

    CloverCL::udpate_halo_time += (t_end.tv_sec * 1.0E6 + t_end.tv_usec) - (t_start.tv_sec * 1.0E6 + t_start.tv_usec);
    CloverCL::udpate_halo_count++;
#endif

}
