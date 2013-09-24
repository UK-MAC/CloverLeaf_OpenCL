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
 *  @brief OCL device-side revert kernels
 *  @author Andrew Mallinson, David Beckingsale, Wayne Gaudin
 *  @details Takes the half step field data used in the predictor and reverts
 *  it to the start of step data, ready for the corrector.
 *  Note that this does not seem necessary in this proxy-app but should be
 *  left in to remain relevant to the full method.
 */

#include "ocl_knls.h"

__kernel void revert_ocl_kernel(
    __global const double * restrict density0,
    __global double * restrict density1,
    __global const double * restrict energy0,
    __global double * restrict energy1)
{
    int  k = get_global_id(1);
    int  j = get_global_id(0);

    if ((j>=2) && (j<=XMAXPLUSONE) && (k>=2) && (k<=YMAXPLUSONE)) {

        density1[ARRAYXY(j,k,XMAXPLUSFOUR)] = density0[ARRAYXY(j,k,XMAXPLUSFOUR)];
        energy1[ARRAYXY(j,k,XMAXPLUSFOUR)] = energy0[ARRAYXY(j,k,XMAXPLUSFOUR)];

    }
}
