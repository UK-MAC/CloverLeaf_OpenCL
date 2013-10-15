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
 *  @brief OCL host-side accelerate kernel.
 *  @author Andrew Mallinson, David Beckingsale
 *  @details Launches the OCL device-side accelerate kernel 
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

extern "C" void accelerate_kernel_ocl_(
        int *xmin,
        int *xmax,
        int *ymin,
        int *ymax,
        double *dbyt); 

extern "C" void accelerate_kernel_ocl_(
        int *xmin,
        int *xmax,
        int *ymin,
        int *ymax,
        double *dbyt)
{
#if PROFILE_OCL_KERNELS
    timeval t_start;
    gettimeofday(&t_start, NULL);
#endif

    try {
        CloverCL::accelerate_knl.setArg(0, *dbyt);
    } catch(cl::Error err) {
        CloverCL::reportError(err, "accelerate_knl setting arguments");
    }

    CloverCL::enqueueKernel_nooffsets_localwg( CloverCL::accelerate_knl, *xmax+3, *ymax+3, CloverCL::local_wg_x_accelerate, CloverCL::local_wg_y_accelerate);

#if PROFILE_OCL_KERNELS
    timeval t_end;
    gettimeofday(&t_end, NULL);

    std::cout << "[PROFILING]: accelerate OpenCL kernel took "
        << (t_end.tv_usec - t_start.tv_usec)*CloverCL::US_TO_SECONDS
        << " seconds (host time)" << std::endl;
#endif
}
