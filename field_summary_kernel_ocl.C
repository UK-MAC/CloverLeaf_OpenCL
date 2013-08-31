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


    events2.push_back(CloverCL::last_event);

    //Start of Reduction
    if (CloverCL::device_type == CL_DEVICE_TYPE_CPU) {

        //// Level 1 of CPU redcution
        //vol_sum_red_cpu_knl.setArg(0, CloverCL::vol_tmp_buffer); 
        //vol_sum_red_cpu_knl.setArg(1, CloverCL::cpu_vol_red_buffer); 
        //vol_sum_red_cpu_knl.setArg(2, CloverCL::num_elements_per_wi[0]); 
        //vol_sum_red_cpu_knl.setArg(3, CloverCL::size_limits[0]); 

        //err = outoforder_queue.enqueueNDRangeKernel(vol_sum_red_cpu_knl, cl::NDRange(1), 
        //                             cl::NDRange(CloverCL::num_workitems_tolaunch[0]),
        //                             cl::NDRange(CloverCL::num_workitems_per_wg[0]), 
        //                             &events2, &vol_reduction_event_array[0]);

        //mass_sum_red_cpu_knl.setArg(0, CloverCL::mass_tmp_buffer); 
        //mass_sum_red_cpu_knl.setArg(1, CloverCL::cpu_mass_red_buffer); 
        //mass_sum_red_cpu_knl.setArg(2, CloverCL::num_elements_per_wi[0]); 
        //mass_sum_red_cpu_knl.setArg(3, CloverCL::size_limits[0]); 

        //err = outoforder_queue.enqueueNDRangeKernel(mass_sum_red_cpu_knl, cl::NDRange(1), 
        //                             cl::NDRange(CloverCL::num_workitems_tolaunch[0]),
        //                             cl::NDRange(CloverCL::num_workitems_per_wg[0]), 
        //                             &events2, &mass_reduction_event_array[0]);

        //ie_sum_red_cpu_knl.setArg(0, CloverCL::ie_tmp_buffer); 
        //ie_sum_red_cpu_knl.setArg(1, CloverCL::cpu_ie_red_buffer); 
        //ie_sum_red_cpu_knl.setArg(2, CloverCL::num_elements_per_wi[0]); 
        //ie_sum_red_cpu_knl.setArg(3, CloverCL::size_limits[0]); 

        //err = outoforder_queue.enqueueNDRangeKernel(ie_sum_red_cpu_knl, cl::NDRange(1), 
        //                             cl::NDRange(CloverCL::num_workitems_tolaunch[0]),
        //                             cl::NDRange(CloverCL::num_workitems_per_wg[0]), 
        //                             &events2, &ie_reduction_event_array[0]);

        //ke_sum_red_cpu_knl.setArg(0, CloverCL::ke_tmp_buffer); 
        //ke_sum_red_cpu_knl.setArg(1, CloverCL::cpu_ke_red_buffer); 
        //ke_sum_red_cpu_knl.setArg(2, CloverCL::num_elements_per_wi[0]); 
        //ke_sum_red_cpu_knl.setArg(3, CloverCL::size_limits[0]); 

        //err = outoforder_queue.enqueueNDRangeKernel(ke_sum_red_cpu_knl, cl::NDRange(1), 
        //                             cl::NDRange(CloverCL::num_workitems_tolaunch[0]),
        //                             cl::NDRange(CloverCL::num_workitems_per_wg[0]), 
        //                             &events2, &ke_reduction_event_array[0]);

        //press_sum_red_cpu_knl.setArg(0, CloverCL::press_tmp_buffer); 
        //press_sum_red_cpu_knl.setArg(1, CloverCL::cpu_press_red_buffer); 
        //press_sum_red_cpu_knl.setArg(2, CloverCL::num_elements_per_wi[0]); 
        //press_sum_red_cpu_knl.setArg(3, CloverCL::size_limits[0]); 

        //err = outoforder_queue.enqueueNDRangeKernel(press_sum_red_cpu_knl, cl::NDRange(1), 
        //                             cl::NDRange(CloverCL::num_workitems_tolaunch[0]),
        //                             cl::NDRange(CloverCL::num_workitems_per_wg[0]), 
        //                             &events2, &press_reduction_event_array[0]);

        //outoforder_queue.enqueueBarrier();


        //// Level 2 of CPU reduction
        //vol_sum_red_cpu_knl.setArg(0, CloverCL::cpu_vol_red_buffer); 
        //vol_sum_red_cpu_knl.setArg(1, CloverCL::vol_sum_val_buffer); 
        //vol_sum_red_cpu_knl.setArg(2, CloverCL::num_elements_per_wi[1]); 
        //vol_sum_red_cpu_knl.setArg(3, CloverCL::size_limits[1]); 

        //err = outoforder_queue.enqueueNDRangeKernel(vol_sum_red_cpu_knl, cl::NDRange(1), 
        //                             cl::NDRange(CloverCL::num_workitems_tolaunch[1]),
        //                             cl::NDRange(CloverCL::num_workitems_per_wg[1]), 
        //                             NULL, &vol_reduction_event_array[1]);

        //mass_sum_red_cpu_knl.setArg(0, CloverCL::cpu_mass_red_buffer); 
        //mass_sum_red_cpu_knl.setArg(1, CloverCL::mass_sum_val_buffer); 
        //mass_sum_red_cpu_knl.setArg(2, CloverCL::num_elements_per_wi[1]); 
        //mass_sum_red_cpu_knl.setArg(3, CloverCL::size_limits[1]); 

        //err = outoforder_queue.enqueueNDRangeKernel(mass_sum_red_cpu_knl, cl::NDRange(1), 
        //                             cl::NDRange(CloverCL::num_workitems_tolaunch[1]),
        //                             cl::NDRange(CloverCL::num_workitems_per_wg[1]), 
        //                             NULL, &mass_reduction_event_array[1]);

        //ie_sum_red_cpu_knl.setArg(0, CloverCL::cpu_ie_red_buffer); 
        //ie_sum_red_cpu_knl.setArg(1, CloverCL::ie_sum_val_buffer); 
        //ie_sum_red_cpu_knl.setArg(2, CloverCL::num_elements_per_wi[1]); 
        //ie_sum_red_cpu_knl.setArg(3, CloverCL::size_limits[1]); 

        //err = outoforder_queue.enqueueNDRangeKernel(ie_sum_red_cpu_knl, cl::NDRange(1), 
        //                             cl::NDRange(CloverCL::num_workitems_tolaunch[1]),
        //                             cl::NDRange(CloverCL::num_workitems_per_wg[1]), 
        //                             NULL, &ie_reduction_event_array[1]);

        //ke_sum_red_cpu_knl.setArg(0, CloverCL::cpu_ke_red_buffer); 
        //ke_sum_red_cpu_knl.setArg(1, CloverCL::ke_sum_val_buffer); 
        //ke_sum_red_cpu_knl.setArg(2, CloverCL::num_elements_per_wi[1]); 
        //ke_sum_red_cpu_knl.setArg(3, CloverCL::size_limits[1]); 

        //err = outoforder_queue.enqueueNDRangeKernel(ke_sum_red_cpu_knl, cl::NDRange(1), 
        //                             cl::NDRange(CloverCL::num_workitems_tolaunch[1]),
        //                             cl::NDRange(CloverCL::num_workitems_per_wg[1]), 
        //                             NULL, &ke_reduction_event_array[1]);

        //press_sum_red_cpu_knl.setArg(0, CloverCL::cpu_press_red_buffer); 
        //press_sum_red_cpu_knl.setArg(1, CloverCL::press_sum_val_buffer); 
        //press_sum_red_cpu_knl.setArg(2, CloverCL::num_elements_per_wi[1]); 
        //press_sum_red_cpu_knl.setArg(3, CloverCL::size_limits[1]); 

        //err = outoforder_queue.enqueueNDRangeKernel(press_sum_red_cpu_knl, cl::NDRange(1), 
        //                             cl::NDRange(CloverCL::num_workitems_tolaunch[1]),
        //                             cl::NDRange(CloverCL::num_workitems_per_wg[1]), 
        //                             NULL, &press_reduction_event_array[1]);


        //outoforder_queue.enqueueBarrier();
    }
    else if (CloverCL::device_type == CL_DEVICE_TYPE_GPU) {

        CloverCL::outoforder_queue.enqueueWaitForEvents(events2);

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
    }
    else {
        std::cerr << "Error: in field summary device type not supported" << std::endl;
    }


    /*
     * Read data back
     */
    try {

        CloverCL::outoforder_queue.enqueueReadBuffer(
                CloverCL::vol_sum_val_buffer,
                CL_FALSE,
                0,
                sizeof(double),
                vol,
                NULL,
                NULL);

        CloverCL::outoforder_queue.enqueueReadBuffer(
                CloverCL::mass_sum_val_buffer,
                CL_FALSE,
                0,
                sizeof(double),
                mass,
                NULL,
                NULL);

        CloverCL::outoforder_queue.enqueueReadBuffer(
                CloverCL::ie_sum_val_buffer,
                CL_FALSE,
                0,
                sizeof(double),
                ie,
                NULL,
                NULL);

        CloverCL::outoforder_queue.enqueueReadBuffer(
                CloverCL::ke_sum_val_buffer,
                CL_FALSE,
                0,
                sizeof(double),
                ke,
                NULL,
                NULL);

        CloverCL::outoforder_queue.enqueueReadBuffer(
                CloverCL::press_sum_val_buffer,
                CL_FALSE,
                0,
                sizeof(double),
                press,
                NULL,
                NULL);

    } catch(cl::Error err) {
        CloverCL::reportError(err, "field_summary reading buffers");
    }


    CloverCL::outoforder_queue.finish();


#if PROFILE_OCL_KERNELS
    timeval t_end;
    gettimeofday(&t_end, NULL);

    std::cout << "[PROFILING]: field_summary OpenCL kernel took " << (t_end.tv_usec - t_start.tv_usec)*CloverCL::US_TO_SECONDS << " seconds (host time)" << std::endl;

#endif

}
