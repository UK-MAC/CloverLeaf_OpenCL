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
 *  @brief OCL host-side advection cell kernel.
 *  @author Andrew Mallinson, David Beckingsale
 *  @details Launches the OCL device-side advection cell kernels 
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

extern "C" void advec_cell_kernel_ocl_(
        int *xmin,
        int *xmax,
        int *ymin,
        int *ymax,
        int *dir_dum,
        int *sweepnumber);

void advec_cell_kernel_ocl_(
        int *xmin,
        int *xmax,
        int *ymin,
        int *ymax,
        int *dir_dum,
        int *sweepnumber)
{
    int g_xdir=1;
    int g_ydir=2;


#if PROFILE_OCL_KERNELS
    timeval t_start;
    gettimeofday(&t_start, NULL);
#endif

#ifdef OCL_VERBOSE
    if (*dir_dum == g_xdir) {
        std::cout << "[CloverCL] -> in advec_cell_kernel_ocl.C... xdir" << std::endl;
    } else {
        std::cout << "[CloverCL] -> in advec_cell_kernel_ocl.C... ydir" << std::endl;
    }
#endif

    if (*dir_dum == g_xdir) {

        if (*sweepnumber == 1) {
            CloverCL::enqueueKernel_nooffsets( CloverCL::advec_cell_xdir_sec1_s1_knl, *xmax+4, *ymax+4);
        } else {
            CloverCL::enqueueKernel_nooffsets( CloverCL::advec_cell_xdir_sec1_s2_knl, *xmax+4, *ymax+4);
        }

        CloverCL::enqueueKernel_nooffsets( CloverCL::advec_cell_xdir_sec2_knl, *xmax+4, *ymax+2);

        CloverCL::enqueueKernel_nooffsets( CloverCL::advec_cell_xdir_sec3_knl, *xmax+2, *ymax+2);

    } else {
        if (*sweepnumber == 1) {
            CloverCL::enqueueKernel_nooffsets( CloverCL::advec_cell_ydir_sec1_s1_knl, *xmax+4, *ymax+4);
        } else {
            CloverCL::enqueueKernel_nooffsets( CloverCL::advec_cell_ydir_sec1_s2_knl, *xmax+4, *ymax+4);
        }

        CloverCL::enqueueKernel_nooffsets( CloverCL::advec_cell_ydir_sec2_knl, *xmax+2, *ymax+4);

        CloverCL::enqueueKernel_nooffsets( CloverCL::advec_cell_ydir_sec3_knl, *xmax+2, *ymax+2);
    }

#if PROFILE_OCL_KERNELS
    timeval t_end;

    CloverCL::queue.finish();

    gettimeofday(&t_end, NULL);

    CloverCL::advec_cell_time += (t_end.tv_usec - t_start.tv_usec);
    CloverCL::advec_cell_count++;
#endif

#ifdef OCL_VERBOSE
    if (*dir_dum == g_xdir) {
        std::cout << "[CloverCL] <- leaving advec_cell_kernel_ocl.C! in xdir" << std::endl;
    } else {
        std::cout << "[CloverCL] <- leaving advec_cell_kernel_ocl.C! in ydir" << std::endl;
    }
#endif
}

