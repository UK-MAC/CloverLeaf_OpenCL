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
 *  @brief OCL host-side advection momentum kernel.
 *  @author Andrew Mallinson, David Beckingsale
 *  @details Launches the OCL device-side advection momentum kernels 
*/

#include "CloverCL.h"

#include <sys/time.h>

extern "C" void advec_mom_kernel_ocl_(int *xmin, int *xmax,
                                      int *ymin, int *ymax,
                                      int *whch_vl, int *swp_nmbr,
                                      int *drctn, int *vctr);

void advec_mom_kernel_ocl_(int *xmin, int *xmax,
                           int *ymin, int *ymax,
                           int *whch_vl, int *swp_nmbr,
                           int *drctn, int *vctr)
{
#if PROFILE_OCL_KERNELS
    timeval t_start;
    gettimeofday(&t_start, NULL);
#endif

    try {
        int mom_sweep=*drctn+2*(*swp_nmbr-1);
        CloverCL::advec_mom_vol_knl.setArg(5, mom_sweep);

        if (*whch_vl== 1) {
            CloverCL::advec_mom_flux_x_vec1_knl.setArg(2, CloverCL::xvel1_buffer);

            CloverCL::advec_mom_flux_x_vecnot1_knl.setArg(2, CloverCL::xvel1_buffer);

            CloverCL::advec_mom_flux_y_vec1_knl.setArg(2, CloverCL::xvel1_buffer);

            CloverCL::advec_mom_flux_y_vecnot1_knl.setArg(2, CloverCL::xvel1_buffer);

            CloverCL::advec_mom_vel_x_knl.setArg(3, CloverCL::xvel1_buffer);

            CloverCL::advec_mom_vel_y_knl.setArg(3, CloverCL::xvel1_buffer);
        } else {

            CloverCL::advec_mom_flux_x_vec1_knl.setArg(2, CloverCL::yvel1_buffer);

            CloverCL::advec_mom_flux_x_vecnot1_knl.setArg(2, CloverCL::yvel1_buffer);

            CloverCL::advec_mom_flux_y_vec1_knl.setArg(2, CloverCL::yvel1_buffer);

            CloverCL::advec_mom_flux_y_vecnot1_knl.setArg(2, CloverCL::yvel1_buffer);

            CloverCL::advec_mom_vel_x_knl.setArg(3, CloverCL::yvel1_buffer);

            CloverCL::advec_mom_vel_y_knl.setArg(3, CloverCL::yvel1_buffer);
        }

    } catch (cl::Error err) {
        CloverCL::reportError(err, " advec_mom_knl setting arguments");
    }



    CloverCL::enqueueKernel_nooffsets_localwg( CloverCL::advec_mom_vol_knl, *xmax+4, *ymax+4, CloverCL::local_wg_x_advecmom_vol, CloverCL::local_wg_y_advecmom_vol);

    if (*drctn == 1) {

        CloverCL::enqueueKernel_nooffsets_localwg( CloverCL::advec_mom_node_x_knl, *xmax+4, *ymax+3, CloverCL::local_wg_x_advecmom_node_x, CloverCL::local_wg_y_advecmom_node_x);

        CloverCL::enqueueKernel_nooffsets_localwg( CloverCL::advec_mom_node_mass_pre_x_knl, *xmax+4, *ymax+3, CloverCL::local_wg_x_advecmom_node_mass_pre_x, CloverCL::local_wg_y_advecmom_node_mass_pre_x);

        if (*vctr==1) {
            CloverCL::enqueueKernel_nooffsets_localwg( CloverCL::advec_mom_flux_x_vec1_knl, *xmax+3, *ymax+3, CloverCL::local_wg_x_advecmom_flux_vec1_x, CloverCL::local_wg_y_advecmom_flux_vec1_x);
        } else {
            CloverCL::enqueueKernel_nooffsets_localwg( CloverCL::advec_mom_flux_x_vecnot1_knl, *xmax+3, *ymax+3, CloverCL::local_wg_x_advecmom_flux_notvec1_x, CloverCL::local_wg_y_advecmom_flux_notvec1_x);
        }

        CloverCL::enqueueKernel_nooffsets_localwg( CloverCL::advec_mom_vel_x_knl, *xmax+3, *ymax+3, CloverCL::local_wg_x_advecmom_vel_x, CloverCL::local_wg_y_advecmom_vel_x);

    } else {

        CloverCL::enqueueKernel_nooffsets_localwg( CloverCL::advec_mom_node_y_knl, *xmax+3, *ymax+4, CloverCL::local_wg_x_advecmom_node_y, CloverCL::local_wg_y_advecmom_node_y);

        CloverCL::enqueueKernel_nooffsets_localwg( CloverCL::advec_mom_node_mass_pre_y_knl, *xmax+3, *ymax+4, CloverCL::local_wg_x_advecmom_node_mass_pre_y, CloverCL::local_wg_y_advecmom_node_mass_pre_y);

        if (*vctr==1) {
            CloverCL::enqueueKernel_nooffsets_localwg( CloverCL::advec_mom_flux_y_vec1_knl, *xmax+3, *ymax+3, CloverCL::local_wg_x_advecmom_flux_vec1_y, CloverCL::local_wg_y_advecmom_flux_vec1_y);
        } else {
            CloverCL::enqueueKernel_nooffsets_localwg( CloverCL::advec_mom_flux_y_vecnot1_knl, *xmax+3, *ymax+3, CloverCL::local_wg_x_advecmom_flux_notvec1_y, CloverCL::local_wg_y_advecmom_flux_notvec1_y);
        }

        CloverCL::enqueueKernel_nooffsets_localwg( CloverCL::advec_mom_vel_y_knl, *xmax+3, *ymax+3, CloverCL::local_wg_x_advecmom_vel_y, CloverCL::local_wg_y_advecmom_vel_y);

    }


#if PROFILE_OCL_KERNELS
    timeval t_end;

    CloverCL::queue.finish();

    gettimeofday(&t_end, NULL);

    CloverCL::advec_mom_time += (t_end.tv_sec * 1.0E6 + t_end.tv_usec) - (t_start.tv_sec * 1.0E6 + t_start.tv_usec);
    CloverCL::advec_mom_count++;
#endif
}
