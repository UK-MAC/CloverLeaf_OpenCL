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
 *  @brief OCL host-side flux kernel.
 *  @author Andrew Mallinson, David Beckingsale
 *  @details Launches the OCL device-side flux kernel 
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

extern "C" void flux_calc_kernel_ocl_(
     int *xmin,
     int *xmax,
     int *ymin,
     int *ymax,
     double *dt_dum);

void flux_calc_kernel_ocl_(
     int *xmin,
     int *xmax,
     int *ymin,
     int *ymax,
     double *dt_dum)
{
#if PROFILE_OCL_KERNELS
    timeval t_start;
    gettimeofday(&t_start, NULL);
#endif

#ifdef OCL_VERBOSE
   std::cout << "[CloverCL] -> in flux_calc_kernel_ocl.C..." << std::endl;
#endif

    try {
        CloverCL::flux_calc_knl.setArg(0, *dt_dum);

    } catch(cl::Error err) {
        CloverCL::reportError(err, "flux_calc setting args");
    }

    CloverCL::enqueueKernel_nooffsets( CloverCL::flux_calc_knl, *xmax+3, *ymax+3);

#if PROFILE_OCL_KERNELS
    timeval t_end;

    CloverCL::queue.finish();

    gettimeofday(&t_end, NULL);

    CloverCL::flux_calc_time += (t_end.tv_usec - t_start.tv_usec);
    CloverCL::flux_calc_count++;
#endif
}
