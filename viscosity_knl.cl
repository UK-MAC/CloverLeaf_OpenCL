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
 *  @brief OCL device-side viscosity kernels
 *  @author Andrew Mallinson, David Beckingsale, Wayne Gaudin
 *  @details  Calculates an artificial viscosity using the Wilkin's method to
 *  smooth out shock front and prevent oscillations around discontinuities.
 *  Only cells in compression will have a non-zero value.
 */

#include "ocl_knls.h"

__kernel void viscosity_ocl_kernel(
        __global const double * restrict celldx,
        __global const double * restrict celldy,
        __global const double * restrict density0,
        __global const double * restrict pressure,
        __global double * restrict viscosity,
        __global const double * restrict xvel0,
        __global const double * restrict yvel0)
{
    double ugrad,vgrad,grad2,pgradx,pgrady,pgradx2,pgrady2,grad,ygrad,pgrad,xgrad,div,strain2,limiter;

    int k = get_global_id(1);
    int j = get_global_id(0);

    if ((j>=2) && (j<=XMAXPLUSONE) && (k>=2) && (k<=YMAXPLUSONE)) {

          ugrad = (xvel0[ARRAYXY(j+1,k  ,XMAXPLUSFIVE)]
                  +xvel0[ARRAYXY(j+1,k+1,XMAXPLUSFIVE)])
                 -(xvel0[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]
                  +xvel0[ARRAYXY(j  ,k+1,XMAXPLUSFIVE)]);

          vgrad = (yvel0[ARRAYXY(j  ,k+1,XMAXPLUSFIVE)]
                  +yvel0[ARRAYXY(j+1,k+1,XMAXPLUSFIVE)])
                 -(yvel0[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]
                  +yvel0[ARRAYXY(j+1,k  ,XMAXPLUSFIVE)]);

          div = (celldx[j]*(ugrad) 
	            +celldy[k]*(vgrad));

          strain2=0.5*(xvel0[ARRAYXY(j  ,k+1,XMAXPLUSFIVE)]
                      +xvel0[ARRAYXY(j+1,k+1,XMAXPLUSFIVE)]
                      -xvel0[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]
                      -xvel0[ARRAYXY(j+1,k  ,XMAXPLUSFIVE)])/celldy[k]
                 +0.5*(yvel0[ARRAYXY(j+1,k  ,XMAXPLUSFIVE)]
                      +yvel0[ARRAYXY(j+1,k+1,XMAXPLUSFIVE)]
                      -yvel0[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]
                      -yvel0[ARRAYXY(j  ,k+1,XMAXPLUSFIVE)])/celldx[j];

          pgradx=(pressure[ARRAYXY(j+1,k  ,XMAXPLUSFOUR)]
                 -pressure[ARRAYXY(j-1,k  ,XMAXPLUSFOUR)])
                /(celldx[j]+celldx[j+1]);
          pgrady=(pressure[ARRAYXY(j  ,k+1,XMAXPLUSFOUR)]
                 -pressure[ARRAYXY(j  ,k-1,XMAXPLUSFOUR)])
                /(celldy[k]+celldy[k+1]);

          pgradx2 = pgradx*pgradx;
          pgrady2 = pgrady*pgrady;

          limiter = ((0.5*(ugrad)/celldx[j])
                      *pgradx2+(0.5*(vgrad)/celldy[k])*pgrady2+strain2*pgradx*pgrady)
                  /fmax(pgradx2+pgrady2,1.0e-16);

          pgradx = copysign(fmax(1.0e-16,fabs(pgradx)),pgradx);
          pgrady = copysign(fmax(1.0e-16,fabs(pgrady)),pgrady);
          pgrad = sqrt(pgradx*pgradx+pgrady*pgrady);
          xgrad = fabs(celldx[j]*pgrad/pgradx);
          ygrad = fabs(celldy[k]*pgrad/pgrady);
          grad  = fmin(xgrad,ygrad);
          grad2 = grad*grad;

          if(limiter > 0.0 || div >= 0.0){
              viscosity[ARRAYXY(j,k,XMAXPLUSFOUR)]=0.0;
          } else {
              viscosity[ARRAYXY(j,k,XMAXPLUSFOUR)]=2.0*density0[ARRAYXY(j,k,XMAXPLUSFOUR)]*grad2*limiter*limiter;
          }
    }
}
