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
 *  @brief OCL host-side initialise chunk kernel.
 *  @author Andrew Mallinson, David Beckingsale
 *  @details Launches the OCL device-side initialise chunk kernel 
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

extern "C" void initialise_chunk_kernel_ocl_(
        int *x_min,
        int *x_max,
        int *y_min,
        int *y_max,
        double *xmin,
        double *ymin,
        double *dx,
        double *dy);


void initialise_chunk_kernel_ocl_(
        int *x_min,
        int *x_max,
        int *y_min,
        int *y_max,
        double *xmin,
        double *ymin,
        double *dx,
        double *dy)
{
#if PROFILE_OCL_KERNELS
    timeval t_start;
    gettimeofday(&t_start, NULL);
#endif

    try {
        CloverCL::initialise_chunk_cell_x_knl.setArg(0, *dx);
    } catch(cl::Error err) {
        CloverCL::reportError(err, "initialise chunk setting arguments cell x kernel");
    }


    try {
        CloverCL::initialise_chunk_cell_y_knl.setArg(0, *dy);
    } catch(cl::Error err) {
        CloverCL::reportError(err, "initialise chunk setting arguments cell y kernel");
    }

    try {
        CloverCL::initialise_chunk_vertex_x_knl.setArg(0, *xmin);
        CloverCL::initialise_chunk_vertex_x_knl.setArg(1, *dx);
    } catch(cl::Error err) {
        CloverCL::reportError(err, "initialise chunk setting arguments vertex x kernel");
    }

    try {
        CloverCL::initialise_chunk_vertex_y_knl.setArg(0, *ymin);
        CloverCL::initialise_chunk_vertex_y_knl.setArg(1, *dy);
    } catch(cl::Error err) {
        CloverCL::reportError(err, "initialise chunk setting arguments vertex y kernel");
    }

    try {
        CloverCL::initialise_chunk_volume_area_knl.setArg(0, *dx);
        CloverCL::initialise_chunk_volume_area_knl.setArg(1, *dy);
    } catch(cl::Error err) {
        CloverCL::reportError(err, "initialise chunk setting arguments volume area kernel");
    }

    CloverCL::enqueueKernel( CloverCL::initialise_chunk_vertex_x_knl, *x_min, *x_max+5);

    CloverCL::enqueueKernel( CloverCL::initialise_chunk_vertex_y_knl, *y_min, *y_max+5);

    CloverCL::enqueueKernel( CloverCL::initialise_chunk_cell_x_knl, *x_min, *x_max+4);

    CloverCL::enqueueKernel( CloverCL::initialise_chunk_cell_y_knl, *y_min, *y_max+4);

    CloverCL::enqueueKernel_nooffsets( CloverCL::initialise_chunk_volume_area_knl, *x_max+4, *y_max+4);

#if PROFILE_OCL_KERNELS
    timeval t_end;

    CloverCL::queue.finish();

    gettimeofday(&t_end, NULL);

    CloverCL::initialise_chunk_time += (t_end.tv_usec - t_start.tv_usec);
    CloverCL::initialise_chunk_count++;
#endif
}
