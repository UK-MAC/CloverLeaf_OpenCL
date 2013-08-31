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
 *  @brief OCL host-side field summary kernel.
 *  @author Andrew Mallinson, David Beckingsale
 *  @details Launches the OCL device-side field summary kernel 
*/

#include "CloverCL.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <iterator>
#include <cmath>

#include <sys/time.h>

extern "C" void field_summary_kernel_ocl_(
        int *xmin,
        int *xmax,
        int *ymin,
        int *ymax,
        double *vol,
        double *mass,
        double *ie,
        double *ke,
        double *press);

void field_summary_kernel_ocl_(
        int *xmin,
        int *xmax,
        int *ymin,
        int *ymax,
        double *vol,
        double *mass,
        double *ie,
        double *ke,
        double *press)
{   
    cl_int err;

    std::vector<cl::Event> events2;

#if PROFILE_OCL_KERNELS
    cl_ulong knl_start, knl_end;

    timeval t_start;
    gettimeofday(&t_start, NULL);
#endif


    /*
     * Run the field summary kernel
     */
    CloverCL::enqueueKernel_nooffsets(CloverCL::field_summary_knl, *xmax+2, *ymax+2);


    //Enqueue a wait for events to stop the reduction kernels execution before the 
    //main field summary kernel has executed     
    events2.push_back(CloverCL::last_event);

    CloverCL::outoforder_queue.enqueueWaitForEvents(events2);


    //Run the reduction kernels 
    try {

        for (int i=1; i<=CloverCL::number_of_red_levels; i++) {

            err = CloverCL::outoforder_queue.enqueueNDRangeKernel(CloverCL::vol_sum_reduction_kernels[i-1], cl::NullRange,
                                             cl::NDRange(CloverCL::num_workitems_tolaunch[i-1]),
                                             cl::NDRange(CloverCL::num_workitems_per_wg[i-1]),
                                             NULL, NULL); 
            err = CloverCL::outoforder_queue.enqueueNDRangeKernel(CloverCL::mass_sum_reduction_kernels[i-1], cl::NullRange,
                                             cl::NDRange(CloverCL::num_workitems_tolaunch[i-1]),
                                             cl::NDRange(CloverCL::num_workitems_per_wg[i-1]),
                                             NULL, NULL); 
            err = CloverCL::outoforder_queue.enqueueNDRangeKernel(CloverCL::ie_sum_reduction_kernels[i-1], cl::NullRange,
                                             cl::NDRange(CloverCL::num_workitems_tolaunch[i-1]),
                                             cl::NDRange(CloverCL::num_workitems_per_wg[i-1]),
                                             NULL, NULL); 
            err = CloverCL::outoforder_queue.enqueueNDRangeKernel(CloverCL::ke_sum_reduction_kernels[i-1], cl::NullRange,
                                             cl::NDRange(CloverCL::num_workitems_tolaunch[i-1]),
                                             cl::NDRange(CloverCL::num_workitems_per_wg[i-1]),
                                             NULL, NULL); 
            err = CloverCL::outoforder_queue.enqueueNDRangeKernel(CloverCL::press_sum_reduction_kernels[i-1], cl::NullRange,
                                             cl::NDRange(CloverCL::num_workitems_tolaunch[i-1]),
                                             cl::NDRange(CloverCL::num_workitems_per_wg[i-1]),
                                             NULL, NULL); 

            if (i < CloverCL::number_of_red_levels) { CloverCL::outoforder_queue.enqueueBarrier(); }
        }

        //required in order to force the execution of the above reduction kernels
        //without this overall runtime increases hugely, at least on Nvidia
        CloverCL::outoforder_queue.finish();

    } catch(cl::Error err) {
         std::cerr
             << "[CloverCL] ERROR: at sum reduction kernel launch in loop"
             << err.what()
             << "("
             << CloverCL::errToString(err.err())
             << ")"
             << std::endl;
    }


    /*
     * Read data back
     */
    try {

        CloverCL::outoforder_queue.enqueueReadBuffer(CloverCL::vol_sum_val_buffer, CL_FALSE, 0, 
                                                     sizeof(double), vol, NULL, NULL);

        CloverCL::outoforder_queue.enqueueReadBuffer(CloverCL::mass_sum_val_buffer, CL_FALSE, 0, 
                                                     sizeof(double), mass, NULL, NULL);

        CloverCL::outoforder_queue.enqueueReadBuffer(CloverCL::ie_sum_val_buffer, CL_FALSE, 0, 
                                                     sizeof(double), ie, NULL, NULL);

        CloverCL::outoforder_queue.enqueueReadBuffer(CloverCL::ke_sum_val_buffer, CL_FALSE, 0, 
                                                     sizeof(double), ke, NULL, NULL);

        CloverCL::outoforder_queue.enqueueReadBuffer(CloverCL::press_sum_val_buffer, CL_FALSE, 0, 
                                                     sizeof(double), press, NULL, NULL);

    } catch(cl::Error err) {
        CloverCL::reportError(err, "field_summary reading buffers");
    }


    CloverCL::outoforder_queue.finish();


#if PROFILE_OCL_KERNELS
    timeval t_end;
    gettimeofday(&t_end, NULL);

    std::cout << "[PROFILING]: field_summary OpenCL kernel took " 
              << (t_end.tv_usec - t_start.tv_usec)*CloverCL::US_TO_SECONDS 
              << " seconds (host time)" << std::endl;

#endif

}
