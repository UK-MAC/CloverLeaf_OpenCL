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
 *  @brief OCL device-side PdV kernels
 *  @author Andrew Mallinson, David Beckingsale, Wayne Gaudin
 *  @details Calculates the change in energy and density in a cell using the
 *  change on cell volume due to the velocity gradients in a cell. The time
 *  level of the velocity data depends on whether it is invoked as the
 *  predictor or corrector.
 */

#include "ocl_knls.h"

__kernel void pdv_correct_ocl_kernel(
        const double dt,
        __global const double * restrict xarea,
        __global const double * restrict yarea,
        __global const double * restrict volume,
        __global const double * restrict density0,
        __global double * restrict density1,
        __global const double * restrict energy0,
        __global double * restrict energy1,
        __global const double * restrict pressure,
        __global const double * restrict viscosity,
        __global const double * restrict xvel0,
        __global const double * restrict xvel1,
        __global const double * restrict yvel0,
        __global const double * restrict yvel1,
        __global double * restrict volume_change)
{
  double recip_volume,energy_change,min_cell_volume,right_flux,left_flux,top_flux,bottom_flux,total_flux;

  int j = get_global_id(0);
  int k = get_global_id(1);

  if((j>=2) && (j<=XMAXPLUSONE) && (k>=2) && (k<=YMAXPLUSONE)) {
        left_flux=  (xarea[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)])
                                   *(xvel0[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]
                                    +xvel0[ARRAYXY(j  ,k+1,XMAXPLUSFIVE)]
                                    +xvel0[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]
                                    +xvel0[ARRAYXY(j  ,k+1,XMAXPLUSFIVE)])
                                    *0.25*dt*0.5;
        right_flux= (xarea[ARRAYXY(j+1,k  ,XMAXPLUSFIVE)])
                                   *(xvel0[ARRAYXY(j+1,k  ,XMAXPLUSFIVE)]
                                    +xvel0[ARRAYXY(j+1,k+1,XMAXPLUSFIVE)]
                                    +xvel0[ARRAYXY(j+1,k  ,XMAXPLUSFIVE)]
                                    +xvel0[ARRAYXY(j+1,k+1,XMAXPLUSFIVE)])
                                    *0.25*dt*0.5;
        bottom_flux=(yarea[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)])
                                   *(yvel0[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]
                                    +yvel0[ARRAYXY(j+1,k  ,XMAXPLUSFIVE)]
                                    +yvel0[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]
                                    +yvel0[ARRAYXY(j+1,k  ,XMAXPLUSFIVE)])
                                    *0.25*dt*0.5;
        top_flux=   (yarea[ARRAYXY(j  ,k+1,XMAXPLUSFOUR)])
                                   *(yvel0[ARRAYXY(j  ,k+1,XMAXPLUSFIVE)]
                                    +yvel0[ARRAYXY(j+1,k+1,XMAXPLUSFIVE)]
                                    +yvel0[ARRAYXY(j  ,k+1,XMAXPLUSFIVE)]
                                    +yvel0[ARRAYXY(j+1,k+1,XMAXPLUSFIVE)])
                                    *0.25*dt*0.5;

        total_flux=right_flux-left_flux+top_flux-bottom_flux;

        volume_change[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]=volume[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)]
                                                         /(volume[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)]+total_flux);

        min_cell_volume=fmin(volume[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)]+right_flux-left_flux+top_flux-bottom_flux
                           ,fmin(volume[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)]+right_flux-left_flux
                           ,volume[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)]+top_flux-bottom_flux));

        recip_volume=1.0/volume[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)];

        energy_change=(pressure[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)]/density0[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)]
                     +viscosity[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)]/density0[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)])
                      *total_flux*recip_volume;

        energy1[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)]=energy0[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)]-energy_change;

        density1[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)]=density0[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)]
                                                           *volume_change[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)];

  }
}

__kernel void pdv_predict_ocl_kernel(
        const double dt,
        __global const double * restrict xarea,
        __global const double * restrict yarea,
        __global const double * restrict volume,
        __global const double * restrict density0,
        __global double * restrict density1,
        __global const double * restrict energy0,
        __global double * restrict energy1,
        __global const double * restrict pressure,
        __global const double * restrict viscosity,
        __global const double * restrict xvel0,
        __global const double * restrict xvel1,
        __global const double * restrict yvel0,
        __global const double * restrict yvel1,
        __global double * restrict volume_change)
{
  double recip_volume,energy_change,min_cell_volume,right_flux,left_flux,top_flux,bottom_flux,total_flux;

  int j = get_global_id(0);
  int k = get_global_id(1);

  if((j>=2) && (j<=XMAXPLUSONE) && (k>=2) && (k<=YMAXPLUSONE)) {
        left_flux=  (xarea[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)])
                                   *(xvel0[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]
                                    +xvel0[ARRAYXY(j  ,k+1,XMAXPLUSFIVE)]
                                    +xvel1[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]
                                    +xvel1[ARRAYXY(j  ,k+1,XMAXPLUSFIVE)])
                                    *0.25*dt;
        right_flux= (xarea[ARRAYXY(j+1,k  ,XMAXPLUSFIVE)])
                                   *(xvel0[ARRAYXY(j+1,k  ,XMAXPLUSFIVE)]
                                    +xvel0[ARRAYXY(j+1,k+1,XMAXPLUSFIVE)]
                                    +xvel1[ARRAYXY(j+1,k  ,XMAXPLUSFIVE)]
                                    +xvel1[ARRAYXY(j+1,k+1,XMAXPLUSFIVE)])
                                    *0.25*dt;
        bottom_flux=(yarea[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)])
                                   *(yvel0[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]
                                    +yvel0[ARRAYXY(j+1,k  ,XMAXPLUSFIVE)]
                                    +yvel1[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]
                                    +yvel1[ARRAYXY(j+1,k  ,XMAXPLUSFIVE)])
                                    *0.25*dt;
        top_flux=   (yarea[ARRAYXY(j  ,k+1,XMAXPLUSFOUR)])
                                   *(yvel0[ARRAYXY(j  ,k+1,XMAXPLUSFIVE)]
                                    +yvel0[ARRAYXY(j+1,k+1,XMAXPLUSFIVE)]
                                    +yvel1[ARRAYXY(j  ,k+1,XMAXPLUSFIVE)]
                                    +yvel1[ARRAYXY(j+1,k+1,XMAXPLUSFIVE)])
                                    *0.25*dt;

        total_flux=right_flux-left_flux+top_flux-bottom_flux;

        volume_change[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]=volume[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)]
                                                         /(volume[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)]+total_flux);

        min_cell_volume=fmin(volume[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)]+right_flux-left_flux+top_flux-bottom_flux
                             ,fmin(volume[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)]+right_flux-left_flux
                                  ,volume[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)]+top_flux-bottom_flux)
                            );

        recip_volume=1.0/volume[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)];

        energy_change=(pressure[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)]/density0[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)]
                     +viscosity[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)]/density0[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)])
                      *total_flux*recip_volume;

        energy1[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)]=energy0[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)]-energy_change;

        density1[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)]=density0[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)]
                                                           *volume_change[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)];

  }
}
