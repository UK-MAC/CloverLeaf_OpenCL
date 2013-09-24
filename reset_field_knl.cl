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
 *  @brief OCL device-side reset field kernels
 *  @author Andrew Mallinson, David Beckingsale, Wayne Gaudin
 *  @details Copies all of the final end of step filed data to the begining of
 *  step data, ready for the next timestep.
 */

#include "ocl_knls.h"

__kernel void reset_field_ocl_kernel(
    __global double * restrict density0,
    __global const double * restrict density1,
    __global double * restrict energy0,
    __global const double * restrict energy1,
    __global double * restrict xvel0,
    __global const double * restrict xvel1,
    __global double * restrict yvel0,
    __global const double * restrict yvel1)
{

    int k = get_global_id(1);
    int j = get_global_id(0);

    if ((j>=2) && (j<=XMAXPLUSONE) && (k>=2) && (k<=YMAXPLUSONE)) 
    {
	    density0[ARRAYXY(j,k,XMAXPLUSFOUR)] = density1[ARRAYXY(j,k,XMAXPLUSFOUR)];
		energy0[ARRAYXY(j,k,XMAXPLUSFOUR)] = energy1[ARRAYXY(j,k,XMAXPLUSFOUR)];
    }

    if ((j>=2) && (j<=XMAXPLUSTWO) && (k>=2) && (k<=YMAXPLUSTWO)) 
    {
        xvel0[ARRAYXY(j,k,XMAXPLUSFIVE)] = xvel1[ARRAYXY(j,k,XMAXPLUSFIVE)];
	    yvel0[ARRAYXY(j,k,XMAXPLUSFIVE)] = yvel1[ARRAYXY(j,k,XMAXPLUSFIVE)];
    }
}

