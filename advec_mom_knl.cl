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
 *  @brief OCL device-side advection momentum kernels
 *  @author Andrew Mallinson, David Beckingsale, Wayne Gaudin
 *  @details Performs a second order advective remap on the vertex momentum
 *  using van-Leer limiting and directional splitting.
 *  Note that although pre_vol is only set and not used in the update, please
 *  leave it in the method.
 */

#include "ocl_knls.h"

__kernel void advec_mom_vol_ocl_kernel(
    __global const double * restrict volume,
    __global const double * restrict vol_flux_x,
    __global const double * restrict vol_flux_y,
    __global double * restrict pre_vol,
    __global double * restrict post_vol,
    const int mom_sweep)
{

    int k = get_global_id(1);
    int j = get_global_id(0);

    if ( (j<=XMAXPLUSTHREE) && (k<=YMAXPLUSTHREE) ) { 

        if(mom_sweep==1){
              post_vol[ARRAYXY(j,k,XMAXPLUSFIVE)]=volume[ARRAYXY(j,k,XMAXPLUSFOUR)]
                                                  +vol_flux_y[ARRAYXY(j,k+1,XMAXPLUSFOUR)]
                                                  -vol_flux_y[ARRAYXY(j,k  ,XMAXPLUSFOUR)];
              pre_vol[ARRAYXY(j,k,XMAXPLUSFIVE)]=post_vol[ARRAYXY(j,k,XMAXPLUSFIVE)]
                                                 +vol_flux_x[ARRAYXY(j+1,k,XMAXPLUSFIVE)]
                                                 -vol_flux_x[ARRAYXY(j  ,k,XMAXPLUSFIVE)];
        }
        else if(mom_sweep==2){
              post_vol[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]=volume[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)]
                                                                      +vol_flux_x[ARRAYXY(j+1,k  ,XMAXPLUSFIVE)]
                                                                      -vol_flux_x[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)];
              pre_vol[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]=post_vol[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]
                                                                     +vol_flux_y[ARRAYXY(j  ,k+1,XMAXPLUSFOUR)]
                                                                     -vol_flux_y[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)];
        }
        else if(mom_sweep==3){
              post_vol[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]=volume[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)];
              pre_vol[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]=post_vol[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]
                                                                +vol_flux_y[ARRAYXY(j  ,k+1,XMAXPLUSFOUR)]
                                                                -vol_flux_y[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)];
        }
        else if(mom_sweep==4){
              post_vol[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]=volume[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)];
              pre_vol[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]=post_vol[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]
                                                                +vol_flux_x[ARRAYXY(j+1,k  ,XMAXPLUSFIVE)]
                                                                -vol_flux_x[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)];
        }
    }
}

__kernel void advec_mom_node_ocl_kernel_x(
    __global const double * restrict mass_flux_x,
    __global double * restrict node_flux,
    __global const double * restrict density1,
    __global const double * restrict post_vol,
    __global double * restrict node_mass_post)
{
    int k = get_global_id(1);
    int j = get_global_id(0);

    if ( (j<=XMAXPLUSTHREE) && (k>=2) && (k<=YMAXPLUSTWO) ) {

        node_flux[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)] = 0.25*(mass_flux_x[ARRAYXY(j  ,k-1,XMAXPLUSFIVE)]
                                                        +mass_flux_x[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]
                                                        +mass_flux_x[ARRAYXY(j+1,k-1,XMAXPLUSFIVE)]
                                                        +mass_flux_x[ARRAYXY(j+1,k  ,XMAXPLUSFIVE)]);
    }

    if ( (j>=1) && (j<=XMAXPLUSTHREE) && (k>=2) && (k<=YMAXPLUSTWO) ) {

        node_mass_post[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]= 0.25*( density1[ARRAYXY(j  ,k-1,XMAXPLUSFOUR)]
                                                             *post_vol[ARRAYXY(j  ,k-1,XMAXPLUSFIVE)]
                                                             +density1[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)]
                                                             *post_vol[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]
                                                             +density1[ARRAYXY(j-1,k-1,XMAXPLUSFOUR)]
                                                             *post_vol[ARRAYXY(j-1,k-1,XMAXPLUSFIVE)]
                                                             +density1[ARRAYXY(j-1,k  ,XMAXPLUSFOUR)]
                                                             *post_vol[ARRAYXY(j-1,k  ,XMAXPLUSFIVE)]);
    }

}

__kernel void advec_mom_node_mass_pre_ocl_kernel_x(
    __global double * restrict node_mass_pre,
    __global const double * restrict node_mass_post,
    __global const double * restrict node_flux)
{
    int k = get_global_id(1);
    int j = get_global_id(0);

    if ( (j>=1) && (j<=XMAXPLUSTHREE) && (k>=2) && (k<=YMAXPLUSTWO) ) {

        node_mass_pre[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]=node_mass_post[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]
                                                     -node_flux[ARRAYXY(j-1,k  ,XMAXPLUSFIVE)]
								                     +node_flux[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)];
    }
}

__kernel void advec_mom_flux_ocl_kernel_x_vec1(
    __global const double * restrict node_flux,
    __global const double * restrict node_mass_pre,
    __global const double * restrict vel1,
    __global double * restrict advec_vel,
    __global double * restrict mom_flux,
    __global const double * restrict celldx)
{
    double sigma, sigma2, wind, wind2, width;
    double vdiffuw, vdiffdw, vdiffuw2, vdiffdw2, auw, adw, auw2, limiter, limiter2;

    int k = get_global_id(1);
    int j = get_global_id(0);

    if ( (j>=1) && (j<=XMAXPLUSTWO) && (k>=2) && (k<=YMAXPLUSTWO) ) {

          sigma=fabs(node_flux[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)])/(node_mass_pre[ARRAYXY(j+1,k, XMAXPLUSFIVE)]);
          sigma2=fabs(node_flux[ARRAYXY(j ,k  ,XMAXPLUSFIVE)])/(node_mass_pre[ARRAYXY(j  ,k, XMAXPLUSFIVE)]);
          width=celldx[j];
          vdiffuw=vel1[ARRAYXY(j+1,k, XMAXPLUSFIVE)]-vel1[ARRAYXY(j+2, k, XMAXPLUSFIVE)];
          vdiffdw=vel1[ARRAYXY(j  ,k, XMAXPLUSFIVE)]-vel1[ARRAYXY(j+1, k, XMAXPLUSFIVE)];
          vdiffuw2=vel1[ARRAYXY(j ,k, XMAXPLUSFIVE)]-vel1[ARRAYXY(j-1, k, XMAXPLUSFIVE)];
          vdiffdw2=-1*vdiffdw;
          auw=fabs(vdiffuw);
          adw=fabs(vdiffdw);
          auw2=fabs(vdiffuw2);
          wind=1.0;
          wind2=1.0;

          if(vdiffdw<=0.0) wind=-1.0;
          if(vdiffdw2<=0.0) wind2=-1.0;
          limiter=wind*fmin(width*((2.0-sigma)*adw/width+(1.0+sigma)*auw/celldx[j+1])/6.0,fmin(auw,adw));
          limiter2=wind2*fmin(width*((2.0-sigma2)*adw/width+(1.0+sigma2)*auw2/celldx[j-1])/6.0,fmin(auw2,adw));
          if(vdiffuw*vdiffdw<=0.0) limiter=0.0;
          if(vdiffuw2*vdiffdw2<=0.0) limiter2=0.0;
          if(node_flux[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]<0.0){
            advec_vel[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]=vel1[ARRAYXY(j+1,k  ,XMAXPLUSFIVE)]+(1.0-sigma)*limiter;
          }
          else{
            advec_vel[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]=vel1[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]+(1.0-sigma2)*limiter2;
          }
          mom_flux[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]=advec_vel[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]
                                                             *node_flux[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)];
    }
}

__kernel void advec_mom_flux_ocl_kernel_x_notvec1(
    __global const double * restrict node_flux,
    __global const double * restrict node_mass_pre,
    __global const double * restrict vel1,
    __global double * restrict advec_vel,
    __global double * restrict mom_flux,
    __global const double * restrict celldx)
{
    int upwind, donor, downwind, dif;
    double sigma, width, wind;
    double vdiffuw, vdiffdw, auw, adw, limiter;

    int k = get_global_id(1);
    int j = get_global_id(0);

    if ( (j>=1) && (j<=XMAXPLUSTWO) && (k>=2) && (k<=YMAXPLUSTWO) ) {

        if(node_flux[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]<0.0){
          upwind=j+2;
          donor=j+1;
          downwind=j;
          dif=donor;
        }
        else{
          upwind=j-1;
          donor=j;
          downwind=j+1;
          dif=upwind;
        }
        sigma=fabs(node_flux[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)])/(node_mass_pre[ARRAYXY(donor,k  ,XMAXPLUSFIVE)]);
        width=celldx[j];
        vdiffuw=vel1[ARRAYXY(donor,k  ,XMAXPLUSFIVE)]-vel1[ARRAYXY(upwind,k  ,XMAXPLUSFIVE)];
        vdiffdw=vel1[ARRAYXY(downwind,k  ,XMAXPLUSFIVE)]-vel1[ARRAYXY(donor,k  ,XMAXPLUSFIVE)];
        limiter=0.0;
        if(vdiffuw*vdiffdw>0.0){
          auw=fabs(vdiffuw);
          adw=fabs(vdiffdw);
          wind=1.0;
          if(vdiffdw<=0.0) wind=-1.0;
          limiter=wind*fmin(width*((2.0-sigma)*adw/width+(1.0+sigma)*auw/celldx[dif])/6.0,fmin(auw,adw));
        }
        advec_vel[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]=vel1[ARRAYXY(donor,k  ,XMAXPLUSFIVE)]+(1.0-sigma)*limiter;
        mom_flux[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]=advec_vel[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]
                                                           *node_flux[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)];
    }
}

__kernel void advec_mom_vel_ocl_kernel_x(
    __global const double * restrict node_mass_post,
    __global const double * restrict node_mass_pre,
    __global const double * restrict mom_flux,
    __global double * restrict vel1)
{
    int k = get_global_id(1);
    int j = get_global_id(0);

    if ( (j>=2) && (j<=XMAXPLUSTWO) && (k>=2) && (k<=YMAXPLUSTWO) ) {

        vel1[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]=(vel1[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]
                                                        *node_mass_pre[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]
                                                        +mom_flux[ARRAYXY(j-1,k  ,XMAXPLUSFIVE)]
                                                        -mom_flux[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)])
                                                        /node_mass_post[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)];
    }
}









__kernel void advec_mom_node_ocl_kernel_y(
    __global const double * restrict mass_flux_y,
    __global double * restrict node_flux,
    __global double * restrict node_mass_post,
    __global const double * restrict density1,
    __global const double * restrict post_vol)
{

    int k = get_global_id(1);
    int j = get_global_id(0);

    if ( (j>=2) && (j<=XMAXPLUSTWO) && (k<=YMAXPLUSTHREE) ) {

        node_flux[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]=0.25*(mass_flux_y[ARRAYXY(j-1,k  ,XMAXPLUSFOUR)]
                                                      +mass_flux_y[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)]
                                                      +mass_flux_y[ARRAYXY(j-1,k+1,XMAXPLUSFOUR)]
                                                      +mass_flux_y[ARRAYXY(j  ,k+1,XMAXPLUSFOUR)]);

    }

    if ( (j>=2) && (j<=XMAXPLUSTWO) && (k>=1) && (k<=YMAXPLUSTHREE) ) {

        node_mass_post[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]=0.25*(density1[ARRAYXY(j  ,k-1,XMAXPLUSFOUR)]
                                                           *post_vol[ARRAYXY(j  ,k-1,XMAXPLUSFIVE)]
                                                           +density1[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)]
                                                           *post_vol[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]
                                                           +density1[ARRAYXY(j-1,k-1,XMAXPLUSFOUR)]
                                                           *post_vol[ARRAYXY(j-1,k-1,XMAXPLUSFIVE)]
                                                           +density1[ARRAYXY(j-1,k  ,XMAXPLUSFOUR)]
                                                           *post_vol[ARRAYXY(j-1,k  ,XMAXPLUSFIVE)]);
    }
}

__kernel void advec_mom_node_mass_pre_ocl_kernel_y(
    __global double * restrict node_mass_pre,
    __global const double * restrict node_mass_post,
    __global const double * restrict node_flux)
{
    int k = get_global_id(1);
    int j = get_global_id(0);

    if ( (j>=2) && (j<=XMAXPLUSTWO) && (k>=1) && (k<=YMAXPLUSTHREE) ) {

        node_mass_pre[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)] = node_mass_post[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]
                                                       - node_flux[ARRAYXY(j  ,k-1,XMAXPLUSFIVE)]
                                                       + node_flux[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)];
    }
}

__kernel void advec_mom_flux_ocl_kernel_y_vec1(
    __global const double * restrict node_flux,
    __global const double * restrict node_mass_pre,
    __global const double * restrict vel1,
    __global double * restrict advec_vel,
    __global double * restrict mom_flux,
    __global const double * restrict celldy)
{
    double sigma, sigma2, width, wind, wind2;
    double vdiffuw, vdiffdw, vdiffuw2, vdiffdw2, auw, adw, auw2, limiter, limiter2;

    int k = get_global_id(1);
    int j = get_global_id(0);

    if ( (j>=2) && (j<=XMAXPLUSTWO) && (k>=1) && (k<=YMAXPLUSTWO) ) {

          sigma=fabs(node_flux[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)])/(node_mass_pre[ARRAYXY(j  ,k+1,XMAXPLUSFIVE)]);
          sigma2=fabs(node_flux[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)])/(node_mass_pre[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]);
          width=celldy[k];
          vdiffuw=vel1[ARRAYXY(j  ,k+1,XMAXPLUSFIVE)]-vel1[ARRAYXY(j  ,k+2,XMAXPLUSFIVE)];
          vdiffdw=vel1[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]-vel1[ARRAYXY(j  ,k+1,XMAXPLUSFIVE)];
          vdiffuw2=vel1[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]-vel1[ARRAYXY(j  ,k-1,XMAXPLUSFIVE)];
          vdiffdw2=-vdiffdw;
          auw=fabs(vdiffuw);
          adw=fabs(vdiffdw);
          auw2=fabs(vdiffuw2);
          wind=1.0;
          wind2=1.0;
          if(vdiffdw<=0.0) wind=-1.0;
          if(vdiffdw2<=0.0) wind2=-1.0;
          limiter=wind*fmin(width*((2.0-sigma)*adw/width+(1.0+sigma)*auw/celldy[k+1])/6.0,fmin(auw,adw));
          limiter2=wind2*fmin(width*((2.0-sigma2)*adw/width+(1.0+sigma2)*auw2/celldy[k-1])/6.0,fmin(auw2,adw));
          if(vdiffuw*vdiffdw<=0.0) limiter=0.0;
          if(vdiffuw2*vdiffdw2<=0.0) limiter2=0.0;
          if(node_flux[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]<0.0){
            advec_vel[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]=vel1[ARRAYXY(j  ,k+1,XMAXPLUSFIVE)]+(1.0-sigma)*limiter;
          }
          else{
            advec_vel[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]=vel1[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]+(1.0-sigma2)*limiter2;
          }
          mom_flux[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]=advec_vel[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]
                                                             *node_flux[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)];
    }
}

__kernel void advec_mom_flux_ocl_kernel_y_notvec1(
    __global const double * restrict node_flux,
    __global const double * restrict node_mass_pre,
    __global const double * restrict vel1,
    __global double * restrict advec_vel,
    __global double * restrict mom_flux,
    __global const double * restrict celldy)
{
    int upwind, donor, downwind, dif;
    double sigma, width, wind;
    double vdiffuw, vdiffdw, auw, adw, limiter;

    int k = get_global_id(1);
    int j = get_global_id(0);

    if ( (j>=2) && (j<=XMAXPLUSTWO) && (k>=1) &&  (k<=YMAXPLUSTWO) ) {

        if(node_flux[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]<0.0){
          upwind=k+2;
          donor=k+1;
          downwind=k;
          dif=donor;
        }
        else{
          upwind=k-1;
          donor=k;
          downwind=k+1;
          dif=upwind;
        }
        sigma=fabs(node_flux[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)])/(node_mass_pre[ARRAYXY(j  ,donor,XMAXPLUSFIVE)]);
        width=celldy[k];
        vdiffuw=vel1[ARRAYXY(j  ,donor,XMAXPLUSFIVE)]-vel1[ARRAYXY(j  ,upwind,XMAXPLUSFIVE)];
        vdiffdw=vel1[ARRAYXY(j  ,downwind ,XMAXPLUSFIVE)]-vel1[ARRAYXY(j  ,donor,XMAXPLUSFIVE)];
        limiter=0.0;
        if(vdiffuw*vdiffdw>0.0){
          auw=fabs(vdiffuw);
          adw=fabs(vdiffdw);
          wind=1.0;
          if(vdiffdw<=0.0) wind=-1.0;
          limiter=wind*fmin(width*((2.0-sigma)*adw/width+(1.0+sigma)*auw/celldy[dif])/6.0,fmin(auw,adw));
        }
        advec_vel[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]=vel1[ARRAYXY(j  ,donor,XMAXPLUSFIVE)]+(1.0-sigma)*limiter;
        mom_flux[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]=advec_vel[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]
                                                           *node_flux[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)];
    }
}

__kernel void advec_mom_vel_ocl_kernel_y(
    __global const double * restrict node_mass_post,
    __global const double * restrict node_mass_pre,
    __global const double * restrict mom_flux,
    __global double * restrict vel1)
{
    int k = get_global_id(1);
    int j = get_global_id(0);

    if ( (j>=2) && (j<=XMAXPLUSTWO) && (k>=2) && (k<=YMAXPLUSTWO) ) {

        vel1[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]=(vel1[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]
                                                        *node_mass_pre[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]
                                                        +mom_flux[ARRAYXY(j  ,k-1,XMAXPLUSFIVE)]
                                                        -mom_flux[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)])
                                                        /node_mass_post[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)];

    }
}
