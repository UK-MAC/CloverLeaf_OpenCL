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
 *  @brief OCL host-side timestep calculation kernel.
 *  @author Andrew Mallinson, David Beckingsale
 *  @details Launches the OCL device-side timestep calculation kernel 
*/

#include "CloverCL.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <iterator>
#include <cmath>
#include <limits>

#include <sys/time.h>

#define ARRAY1D(i_index,i_lb) ((i_index)-(i_lb))
#define ARRAY2D(i_index,j_index,i_size,i_lb,j_lb) ((i_size)*(j_index-(j_lb))+(i_index)-(i_lb))

extern "C" void calc_dt_kernel_ocl_(
        int *xmin,
        int *xmax,
        int *ymin,
        int *ymax,
        double *dtmin,                 
        //double *cellx,                               
        //double *celly,                               
        //double *density0,                            
        //double *energy0,                             
        //double *pressure,                            
        //double *viscosity,                           
        //double *soundspeed,                          
        //double *xvel0,
        //double *yvel0,                         
        double *dt_min_val, 
        int *dtl_control,
        double *xl_pos,     
        double *yl_pos,     
        int *jldt,       
        int *kldt,       
        int *small);

void calc_dt_kernel_ocl_(
        int *xmin,
        int *xmax,
        int *ymin,
        int *ymax,
        double *dtmin,                 
        //double *cellx,                               
        //double *celly,                               
        //double *density0,                            
        //double *energy0,                             
        //double *pressure,                            
        //double *viscosity,                           
        //double *soundspeed,                          
        //double *xvel0,
        //double *yvel0,                         
        double *dt_min_val, 
        int *dtl_control,
        double *xl_pos,     
        double *yl_pos,     
        int *jldt,       
        int *kldt,       
        int *small)
{
    double jk_control=1.1;

    cl_int err;

#ifdef OCL_VERBOSE
    std::cout << "[CloverCL] -> in dt_calc_kernel_ocl.C..." << std::endl;
#endif

#if PROFILE_OCL_KERNELS
    timeval t_start;
    gettimeofday(&t_start, NULL);
#endif


    /*
     * Run the calc dt kernel
     */
    CloverCL::enqueueKernel_nooffsets(CloverCL::dt_calc_knl, *xmax+2, *ymax+2);



    // Run the reduction kernels 
    try {
    
        for (int i=1; i<=CloverCL::number_of_red_levels; i++) {

#ifdef OCL_VERBOSE
            std::cout << "Entering DT calc reduction level: " << i << std::endl; 
#endif

            err = CloverCL::queue.enqueueNDRangeKernel(CloverCL::min_reduction_kernels[i-1], cl::NullRange, 
                                                       cl::NDRange(CloverCL::num_workitems_tolaunch[i-1]),
        				                               cl::NDRange(CloverCL::num_workitems_per_wg[i-1]), 
        				                               NULL, NULL); 
        } 
        
        //clfinish required to force execution of the above reduction kernels 
        //as without this experience a large slowdown at least on Nvidia    
        CloverCL::queue.finish();
    
    } catch(cl::Error err) {
        CloverCL::reportError(err, "[CloverCL] ERROR: at min reduction kernel launch in loop");
    }

    /*
     * Read data back with a blocking read
     */
    try { 

        err = CloverCL::queue.enqueueReadBuffer(CloverCL::dt_min_val_buffer, CL_TRUE, 0, 
                                                sizeof(double), dt_min_val, NULL, NULL);

    } catch(cl::Error err) {
        CloverCL::reportError(err, "[CloverCL] ERROR: at dt_calc_knl read data back stage");
    }


    // Extract the mimimum timestep information
    *dtl_control = (int) (10.01*(jk_control - ((int) jk_control) ) );
    jk_control = jk_control-(jk_control- (int) jk_control);
    *jldt = ((int) jk_control) % *xmax;
    *kldt = (int) 1+(jk_control / *xmax);
    //*xl_pos = cellx[ARRAY1D(*jldt, *xmin-2)];
    //*yl_pos = celly[ARRAY1D(*kldt, *ymin-2)];


    if (*dt_min_val < *dtmin) { *small=1; }

    if (*small != 0) { 
        //try { 
        //    err = CloverCL::queue.enqueueReadBuffer(CloverCL::xvel0_buffer, CL_TRUE, 0, 
        //                                            (*xmax+5)*(*ymax+5)*sizeof(double), xvel0, NULL, NULL);

        //    err = CloverCL::queue.enqueueReadBuffer(CloverCL::yvel0_buffer, CL_TRUE, 0, 
        //                                            (*xmax+5)*(*ymax+5)*sizeof(double), yvel0, NULL, NULL);

        //    err = CloverCL::queue.enqueueReadBuffer(CloverCL::density0_buffer, CL_TRUE, 0, 
        //                                            (*xmax+4)*(*ymax+4)*sizeof(double), density0, NULL, NULL);

        //    err = CloverCL::queue.enqueueReadBuffer(CloverCL::energy0_buffer, CL_TRUE, 0, 
        //                                            (*xmax+4)*(*ymax+4)*sizeof(double), energy0, NULL, NULL);

        //    err = CloverCL::queue.enqueueReadBuffer(CloverCL::pressure_buffer, CL_TRUE, 0, 
        //                                            (*xmax+4)*(*ymax+4)*sizeof(double), pressure, NULL, NULL);

        //    err = CloverCL::queue.enqueueReadBuffer(CloverCL::soundspeed_buffer, CL_TRUE, 0, 
        //                                            (*xmax+4)*(*ymax+4)*sizeof(double), soundspeed, NULL, NULL);

        //} catch(cl::Error err) {
        //    CloverCL::reportError(err, "[CloverCL] ERROR: at dt_calc_knl read data back stage 2");
        //}

        std::cout << "Timestep information:" << std::endl;
        std::cout << "j, k                 : " << *jldt << "  " << *kldt << std::endl;
        //std::cout << "x, y                 : " << cellx[ARRAY1D(*jldt,*xmin-2)] << "  " << celly[ARRAY1D(*kldt,*ymin-2)] << std::endl;
        std::cout << "timestep : " << *dt_min_val << std::endl;
        std::cout << "dt_min : " << *dtmin << std::endl;
        //std::cout << "Cell velocities;" << std::endl;
        //std::cout << xvel0[ARRAY2D(*jldt  , *kldt  , *xmax+5, *xmin-2, *ymin-2)] << "  " << yvel0[ARRAY2D(*jldt  , *kldt   , *xmax+5, *xmin-2, *ymin-2)] << std::endl;
        //std::cout << xvel0[ARRAY2D(*jldt+1, *kldt  , *xmax+5, *xmin-2, *ymin-2)] << "  " << yvel0[ARRAY2D(*jldt+1, *kldt   , *xmax+5, *xmin-2, *ymin-2)] << std::endl;
        //std::cout << xvel0[ARRAY2D(*jldt+1, *kldt+1, *xmax+5, *xmin-2, *ymin-2)] << "  " << yvel0[ARRAY2D(*jldt+1, *kldt+1 , *xmax+5, *xmin-2, *ymin-2)] << std::endl;
        //std::cout << xvel0[ARRAY2D(*jldt  , *kldt+1, *xmax+5, *xmin-2, *ymin-2)] << "  " << yvel0[ARRAY2D(*jldt  , *kldt+1 , *xmax+5, *xmin-2, *ymin-2)] << std::endl;
        //std::cout << "density, energy, pressure, soundspeed " << std::endl;
        //std::cout << density0[ARRAY2D(*jldt, *kldt, *xmax+4, *xmin-2, *ymin-2)] << "  "
        //          << energy0[ARRAY2D(*jldt, *kldt, *xmax+4, *xmin-2, *ymin-2)] << "  "
        //          << pressure[ARRAY2D(*jldt, *kldt, *xmax+4, *xmin-2, *ymin-2)] << "  "
        //          << soundspeed[ARRAY2D(*jldt, *kldt, *xmax+4, *xmin-2, *ymin-2)] << std::endl;
    }



#if PROFILE_OCL_KERNELS
    timeval t_end;

    CloverCL::queue.finish();

    gettimeofday(&t_end, NULL);

    CloverCL::calc_dt_time += (t_end.tv_usec - t_start.tv_usec);
    CloverCL::calc_dt_count++;
#endif

#ifdef OCL_VERBOSE
    std::cout << "[CloverCL] -> leaving dt_calc_kernel_ocl.C..." << std::endl;
#endif

}
