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
 *  @brief OCL host-side PdV kernel.
 *  @author Andrew Mallinson, David Beckingsale
 *  @details Launches the OCL device-side PdV kernel 
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

extern "C" void pdv_kernel_ocl_(
        int *prdct,
        int *xmin,
        int *xmax,
        int *ymin,
        int *ymax,
        double *dtbyt);

extern "C" void pdv_kernel_ocl_(
        int *prdct,
        int *xmin,
        int *xmax,
        int *ymin,
        int *ymax,
        double *dtbyt)
{
#if PROFILE_OCL_KERNELS
    timeval t_start;
    gettimeofday(&t_start, NULL);
#endif

    try {
        if( *prdct == 0) {
            CloverCL::pdv_correct_knl.setArg(0, *dtbyt);
            CloverCL::enqueueKernel_nooffsets_localwg( CloverCL::pdv_correct_knl, *xmax+2, *ymax+2, CloverCL::local_wg_x_pdv, CloverCL::local_wg_y_pdv);
        } else {
            CloverCL::pdv_predict_knl.setArg(0, *dtbyt);
            CloverCL::enqueueKernel_nooffsets_localwg( CloverCL::pdv_predict_knl, *xmax+2, *ymax+2, CloverCL::local_wg_x_pdv, CloverCL::local_wg_y_pdv);
        }
    } catch(cl::Error err) {
        CloverCL::reportError(err, "pdv_knl setting arguments");
    }

#if PROFILE_OCL_KERNELS
    timeval t_end;
    gettimeofday(&t_end, NULL);

    std::cout << "[PROFILING]: PdV OpenCL kernel took "
        << (t_end.tv_usec - t_start.tv_usec)*CloverCL::US_TO_SECONDS
        << " seconds (host time)" << std::endl;
#endif
}
