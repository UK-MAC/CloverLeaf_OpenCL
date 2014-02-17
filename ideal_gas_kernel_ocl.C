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
 *  @brief OCL host-side ideal gas kernel.
 *  @author Andrew Mallinson, David Beckingsale
 *  @details Launches the OCL device-side ideal gas kernel 
*/

#include "CloverCL.h"

#include <sys/time.h>

extern "C" void ideal_gas_kernel_ocl_(int *xmin, int *xmax,
                                      int *ymin, int *ymax,
                                      int *prdct);


void ideal_gas_kernel_ocl_(int *xmin, int *xmax,
                           int *ymin, int *ymax,
                           int *prdct)
{
#if PROFILE_OCL_KERNELS
    timeval t_start;
    gettimeofday(&t_start, NULL);
#endif

    if ( *prdct == 0 ) {

        CloverCL::enqueueKernel_nooffsets_recordevent( CloverCL::ideal_gas_predict_knl, *xmax+2, *ymax+2);

    } else {

        CloverCL::enqueueKernel_nooffsets_recordevent( CloverCL::ideal_gas_NO_predict_knl, *xmax+2, *ymax+2);

    }

#if PROFILE_OCL_KERNELS
    timeval t_end;

    CloverCL::queue.finish();

    gettimeofday(&t_end, NULL);

    CloverCL::ideal_gas_time += (t_end.tv_sec * 1.0E6 + t_end.tv_usec) - (t_start.tv_sec * 1.0E6 + t_start.tv_usec);
    CloverCL::ideal_gas_count++;
#endif
}
