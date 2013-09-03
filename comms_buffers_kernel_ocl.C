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

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <iterator>
#include <cmath>

#include <sys/time.h>

extern "C" void pack_comms_buffers_left_right_kernel_ocl_(int *left_neighbour, int *right_neighbour,
                                                          int *xinc, int *yinc,
                                                          int *depth, int *num_elements,
                                                          int *nameoffield,
                                                          double *host_left_snd_buffer, 
                                                          double *host_right_snd_buffer);

extern "C" void unpack_comms_buffers_left_right_kernel_ocl_(int *left_neighbour, int *right_neighbour,
                                                            int *xinc, int *yinc,
                                                            int *depth, int *num_elements,
                                                            int *nameoffield,
                                                            double *host_left_rcv_buffer, 
                                                            double *host_right_rcv_buffer); 

extern "C" void pack_comms_buffers_top_bottom_kernel_ocl_(int *top_neighbour, int *bottom_neighbour,
                                                          int *xinc, int *yinc,
                                                          int *depth, int *num_elements,
                                                          int *nameoffield,
                                                          double *host_top_snd_buffer, 
                                                          double *host_bottom_snd_buffer); 

extern "C" void unpack_comms_buffers_top_bottom_kernel_ocl_(int *top_neighbour, int *bottom_neighbour,
                                                            int *xinc, int *yinc,
                                                            int *depth, int *num_elements,
                                                            int *nameoffield,
                                                            double *host_top_rcv_buffer,
                                                            double *host_bottom_rcv_buffer); 



void pack_comms_buffers_left_right_kernel_ocl_(int *left_neighbour, int *right_neighbour,
                                               int *xinc, int *yinc,
                                               int *depth, int *num_elements,
                                               int *nameoffield,
                                               double *host_left_snd_buffer, 
                                               double *host_right_snd_buffer)
{
#if PROFILE_OCL_KERNELS
    timeval t_start;
    gettimeofday(&t_start, NULL);
#endif

    cl::Buffer *field_buffer; 
    int launch_height; 

    switch(*nameoffield) {
        case FIELD_DENSITY0: field_buffer = &CloverCL::density0_buffer; break;
        case FIELD_DENSITY1: field_buffer = &CloverCL::density1_buffer; break;
        case FIELD_ENERGY0: field_buffer = &CloverCL::energy0_buffer; break;
        case FIELD_ENERGY1: field_buffer = &CloverCL::energy1_buffer; break;
        case FIELD_PRESSURE: field_buffer = &CloverCL::pressure_buffer; break;
        case FIELD_VISCOSITY: field_buffer = &CloverCL::viscosity_buffer; break;
        case FIELD_SOUNDSPEED: field_buffer = &CloverCL::soundspeed_buffer; break;
        case FIELD_XVEL0: field_buffer = &CloverCL::xvel0_buffer; break;
        case FIELD_XVEL1: field_buffer = &CloverCL::xvel1_buffer; break;
        case FIELD_YVEL0: field_buffer = &CloverCL::yvel0_buffer; break;
        case FIELD_YVEL1: field_buffer = &CloverCL::yvel1_buffer; break;
        case FIELD_VOL_FLUX_X: field_buffer = &CloverCL::vol_flux_x_buffer; break;
        case FIELD_VOL_FLUX_Y: field_buffer = &CloverCL::vol_flux_y_buffer; break;
        case FIELD_MASS_FLUX_X: field_buffer = &CloverCL::mass_flux_x_buffer; break;
        case FIELD_MASS_FLUX_Y: field_buffer = &CloverCL::mass_flux_y_buffer; break;
    }

    if ( *yinc == 1 ) {
        launch_height = CloverCL::ymax_plusfive_rounded; 
    } 
    else {
        launch_height = CloverCL::ymax_plusfour_rounded; 
    } 

    // call clfinish on the inorder queue to ensure everything is finished
    CloverCL::queue.finish(); 

    // if left exchange enqueue on outoforder pack kernel for left buffer after setting args
    if ( *left_neighbour == CloverCL::external_face) {
        CloverCL::read_left_buffer_knl.setArg(0, *depth);
        CloverCL::read_left_buffer_knl.setArg(1, *xinc);
        CloverCL::read_left_buffer_knl.setArg(2, *yinc);
        CloverCL::read_left_buffer_knl.setArg(3, *field_buffer);
        CloverCL::read_left_buffer_knl.setArg(4, CloverCL::left_send_buffer);

        CloverCL::outoforder_queue.enqueueNDRangeKernel(CloverCL::read_left_buffer_knl, cl::NullRange,
                                                        cl::NDRange(*depth, launch_height), 
                                                        cl::NDRange(CloverCL::fixed_wg_min_size_small_dim, CloverCL::fixed_wg_min_size_large_dim),
                                                        NULL, NULL); 
    }
    

    // if right exchange enqueue on outoforder pack kernel for right buffer after setting args
    if ( *right_neighbour == CloverCL::external_face) {
        CloverCL::read_right_buffer_knl.setArg(0, *depth);
        CloverCL::read_right_buffer_knl.setArg(1, *xinc);
        CloverCL::read_right_buffer_knl.setArg(2, *yinc);
        CloverCL::read_right_buffer_knl.setArg(3, *field_buffer);
        CloverCL::read_right_buffer_knl.setArg(4, CloverCL::right_send_buffer);

        CloverCL::outoforder_queue.enqueueNDRangeKernel(CloverCL::read_right_buffer_knl, cl::NullRange,
                                                        cl::NDRange(*depth, launch_height), 
                                                        cl::NDRange(CloverCL::fixed_wg_min_size_small_dim, CloverCL::fixed_wg_min_size_large_dim),
                                                        NULL, NULL); 

    }

    // enqueue a barrier on the out of order queue
    if ( (*left_neighbour == CloverCL::external_face) || (*right_neighbour == CloverCL::external_face)) 
    {   CloverCL::outoforder_queue.enqueueBarrier(); }

    // if left exchange enqueue a buffer read back for the left send buffer
    if ( *left_neighbour == CloverCL::external_face) {

        CloverCL::outoforder_queue.enqueueReadBuffer(CloverCL::left_send_buffer, CL_FALSE, 0,
                                                     *num_elements*sizeof(double), host_left_snd_buffer, NULL, NULL);
    }

    // if right exchange enequeue a buffer read back for the right send buffer
    if ( *right_neighbour == CloverCL::external_face) {

        CloverCL::outoforder_queue.enqueueReadBuffer(CloverCL::right_send_buffer, CL_FALSE, 0, 
                                                     *num_elements*sizeof(double), host_right_snd_buffer, NULL, NULL);
    }

    //call clfinish on the out of order queue
    CloverCL::outoforder_queue.finish(); 


#if PROFILE_OCL_KERNELS
    timeval t_end;
    gettimeofday(&t_end, NULL);

    std::cout << "[PROFILING]: pack comms buffer left and right OpenCL kernel took "
        << (t_end.tv_usec - t_start.tv_usec)*CloverCL::US_TO_SECONDS
        << " seconds (host time)" << std::endl;
#endif
}





void unpack_comms_buffers_left_right_kernel_ocl_(int *left_neighbour, int *right_neighbour,
                                               int *xinc, int *yinc,
                                               int *depth, int *num_elements,
                                               int *nameoffield,
                                               double *host_left_rcv_buffer, 
                                               double *host_right_rcv_buffer)
{
#if PROFILE_OCL_KERNELS
    timeval t_start;
    gettimeofday(&t_start, NULL);
#endif

    cl::Buffer *field_buffer; 

    switch(*nameoffield) {
        case FIELD_DENSITY0: field_buffer = &CloverCL::density0_buffer; break;
        case FIELD_DENSITY1: field_buffer = &CloverCL::density1_buffer; break;
        case FIELD_ENERGY0: field_buffer = &CloverCL::energy0_buffer; break;
        case FIELD_ENERGY1: field_buffer = &CloverCL::energy1_buffer; break;
        case FIELD_PRESSURE: field_buffer = &CloverCL::pressure_buffer; break;
        case FIELD_VISCOSITY: field_buffer = &CloverCL::viscosity_buffer; break;
        case FIELD_SOUNDSPEED: field_buffer = &CloverCL::soundspeed_buffer; break;
        case FIELD_XVEL0: field_buffer = &CloverCL::xvel0_buffer; break;
        case FIELD_XVEL1: field_buffer = &CloverCL::xvel1_buffer; break;
        case FIELD_YVEL0: field_buffer = &CloverCL::yvel0_buffer; break;
        case FIELD_YVEL1: field_buffer = &CloverCL::yvel1_buffer; break;
        case FIELD_VOL_FLUX_X: field_buffer = &CloverCL::vol_flux_x_buffer; break;
        case FIELD_VOL_FLUX_Y: field_buffer = &CloverCL::vol_flux_y_buffer; break;
        case FIELD_MASS_FLUX_X: field_buffer = &CloverCL::mass_flux_x_buffer; break;
        case FIELD_MASS_FLUX_Y: field_buffer = &CloverCL::mass_flux_y_buffer; break;
    }

    if ( *yinc == 1 ) {
        launch_height = CloverCL::ymax_plusfive_rounded; 
    } 
    else {
        launch_height = CloverCL::ymax_plusfour_rounded; 
    } 

    //if left exchange then enqueue a biffer write to transfer the info from the host buffer to the card
    if ( *left_neighbour == CloverCL::external_face) {

        CloverCL::outoforder_queue.enqueueWriteBuffer(CloverCL::left_recv_buffer, CL_FALSE, 0,
                                                     *num_elements*sizeof(double), host_left_rcv_buffer, NULL, NULL);
    }

    //if right exchange then enqueue a biffer write to transfer the info from the host buffer to the card
    if ( *right_neighbour == CloverCL::external_face) {

        CloverCL::outoforder_queue.enqueueWriteBuffer(CloverCL::right_recv_buffer, CL_FALSE, 0,
                                                     *num_elements*sizeof(double), host_right_rcv_buffer, NULL, NULL);
    }


    // call clfinish on the outof order queue if either left or rigth exchange
    CloverCL::outoforder_queue.finish();

    // if left exhange enqueue and unpack left kernel on the outorder queue
    if ( *left_neighbour == CloverCL::external_face) {
        CloverCL::write_left_buffer_knl.setArg(0, *depth);
        CloverCL::write_left_buffer_knl.setArg(1, *xinc);
        CloverCL::write_left_buffer_knl.setArg(2, *yinc);
        CloverCL::write_left_buffer_knl.setArg(3, *field_buffer);
        CloverCL::write_left_buffer_knl.setArg(4, CloverCL::left_recv_buffer) 

        CloverCL::outoforder_queue.enqueueNDRangeKernel(CloverCL::write_left_buffer_knl, cl::NullRange,
                                                        cl::NDRange(*depth, launch_height), 
                                                        cl::NDRange(CloverCL::fixed_wg_min_size_small_dim, CloverCL::fixed_wg_min_size_large_dim),
                                                        NULL, NULL); 
    }

    // if right exchange enqueue the unpack right kernel on the outoforder queue
    if ( *right_neighbour == CloverCL::external_face) {
        CloverCL::write_right_buffer_knl.setArg(0, *depth);
        CloverCL::write_right_buffer_knl.setArg(1, *xinc);
        CloverCL::write_right_buffer_knl.setArg(2, *yinc);
        CloverCL::write_right_buffer_knl.setArg(3, *field_buffer);
        CloverCL::write_right_buffer_knl.setArg(4, CloverCL::right_recv_buffer);

        CloverCL::outoforder_queue.enqueueNDRangeKernel(CloverCL::write_right_buffer_knl, cl::NullRange,
                                                        cl::NDRange(*depth, launch_height), 
                                                        cl::NDRange(CloverCL::fixed_wg_min_size_small_dim, CloverCL::fixed_wg_min_size_large_dim),
                                                        NULL, NULL); 
    }

    // call clfinish on the outof order queue if either left or rigth exchange
    CloverCL::outoforder_queue.finish();


#if PROFILE_OCL_KERNELS
    timeval t_end;
    gettimeofday(&t_end, NULL);

    std::cout << "[PROFILING]: unpack comms buffer left and right OpenCL kernel took "
        << (t_end.tv_usec - t_start.tv_usec)*CloverCL::US_TO_SECONDS
        << " seconds (host time)" << std::endl;
#endif
}





void pack_comms_buffers_top_bottom_kernel_ocl_(int *top_neighbour, int *bottom_neighbour,
                                               int *xinc, int *yinc,
                                               int *depth, int *num_elements,
                                               int *nameoffield,
                                               double *host_top_snd_buffer, 
                                               double *host_bottom_snd_buffer)
{
#if PROFILE_OCL_KERNELS
    timeval t_start;
    gettimeofday(&t_start, NULL);
#endif

    cl::Buffer *field_buffer; 

    switch(*nameoffield) {
        case FIELD_DENSITY0: field_buffer = &CloverCL::density0_buffer; break;
        case FIELD_DENSITY1: field_buffer = &CloverCL::density1_buffer; break;
        case FIELD_ENERGY0: field_buffer = &CloverCL::energy0_buffer; break;
        case FIELD_ENERGY1: field_buffer = &CloverCL::energy1_buffer; break;
        case FIELD_PRESSURE: field_buffer = &CloverCL::pressure_buffer; break;
        case FIELD_VISCOSITY: field_buffer = &CloverCL::viscosity_buffer; break;
        case FIELD_SOUNDSPEED: field_buffer = &CloverCL::soundspeed_buffer; break;
        case FIELD_XVEL0: field_buffer = &CloverCL::xvel0_buffer; break;
        case FIELD_XVEL1: field_buffer = &CloverCL::xvel1_buffer; break;
        case FIELD_YVEL0: field_buffer = &CloverCL::yvel0_buffer; break;
        case FIELD_YVEL1: field_buffer = &CloverCL::yvel1_buffer; break;
        case FIELD_VOL_FLUX_X: field_buffer = &CloverCL::vol_flux_x_buffer; break;
        case FIELD_VOL_FLUX_Y: field_buffer = &CloverCL::vol_flux_y_buffer; break;
        case FIELD_MASS_FLUX_X: field_buffer = &CloverCL::mass_flux_x_buffer; break;
        case FIELD_MASS_FLUX_Y: field_buffer = &CloverCL::mass_flux_y_buffer; break;
    }

    if (*xinc == 1) {
        launch_width = CloverCL::xmax_plusfive_rounded;
    }
    else {
        launch_width = CloverCL::xmax_plusfour_rounded;
    }

    // if bottom exchange enqueue on outoforder pack kernel for bottom buffer after setting args
    if ( *bottom_neighbour == CloverCL::external_face ) {
        CloverCL::read_bottom_buffer_knl.setArg(0, *depth);
        CloverCL::read_bottom_buffer_knl.setArg(1, *xinc);
        CloverCL::read_bottom_buffer_knl.setArg(2, *yinc);
        CloverCL::read_bottom_buffer_knl.setArg(3, *field_buffer);
        CloverCL::read_bottom_buffer_knl.setArg(4, CloverCL::bottom_send_buffer);

        CloverCL::outoforder_queue.enqueueNDRangeKernel(CloverCL::read_bottom_buffer_knl, cl::NullRange,
                                                        cl::NDRange(launch_width, *depth),
                                                        cl::NDRange(CloverCL::fixed_wg_min_size_large_dim, CloverCL::fixed_wg_min_size_small_dim),
                                                        NULL, NULL); 
    }

    // if top exchange enqueue on outoforder pack kernel for top buffer after setting args
    if ( *top_neighbour == CloverCL::external_face ) {
        CloverCL::read_top_buffer_knl.setArg(0, *depth);
        CloverCL::read_top_buffer_knl.setArg(1, *xinc);
        CloverCL::read_top_buffer_knl.setArg(2, *yinc);
        CloverCL::read_top_buffer_knl.setArg(3, *field_buffer);
        CloverCL::read_top_buffer_knl.setArg(4, CloverCL::top_send_buffer);

        CloverCL::outoforder_queue.enqueueNDRangeKernel(CloverCL::read_top_buffer_knl, cl::NullRange,
                                                        cl::NDRange(launch_width, *depth),
                                                        cl::NDRange(CloverCL::fixed_wg_min_size_large_dim, CloverCL::fixed_wg_min_size_small_dim),
                                                        NULL, NULL); 
    }

    // enqueue a barrier on the out of order queue
    if ( (*top_neighbour == CloverCL::external_face) || (*bottom_neighbour == CloverCL::external_face))
    {   CloverCL::outoforder_queue.enqueueBarrier(); }

    // if bottom exchange enqueue a buffer read back for the bottom send buffer
    if ( *bottom_neighbour == CloverCL::external_face ) {

        CloverCL::outoforder_queue.enqueueReadBuffer(CloverCL::bottom_send_buffer, CL_FALSE, 0, 
                                                     *num_elements*sizeof(double), host_bottom_send_buffer);
    }

    // if top exchange enequeue a buffer read back for the top send buffer
    if ( *top_neighbour == CloverCL::external_face ) {

        CloverCL::outoforder_queue.enqueueReadBuffer(CloverCL::top_send_buffer, CL_FALSE, 0, 
                                                     *num_elements*sizeof(double), host_top_send_buffer);
    }

    //call clfinish on the out of order queue
    CloverCL::outoforder_queue.finish(); 


#if PROFILE_OCL_KERNELS
    timeval t_end;
    gettimeofday(&t_end, NULL);

    std::cout << "[PROFILING]: pack comms buffer left and right OpenCL kernel took "
        << (t_end.tv_usec - t_start.tv_usec)*CloverCL::US_TO_SECONDS
        << " seconds (host time)" << std::endl;
#endif
}





void unpack_comms_buffers_top_bottom_kernel_ocl_(int *top_neighbour, int *bottom_neighbour,
                                                 int *xinc, int *yinc,
                                                 int *depth, int *num_elements,
                                                 int *nameoffield,
                                                 double *host_top_rcv_buffer,
                                                 double *host_bottom_rcv_buffer)
{
#if PROFILE_OCL_KERNELS
    timeval t_start;
    gettimeofday(&t_start, NULL);
#endif

    cl::Buffer *field_buffer;

    //if bottom exchange then enqueue a buffer write to transfer the data to the card 
    if ( *bottom_neighbour == CloverCL::external_face) {

        CloverCL::outoforder_queue.enqueueWriteBuffer(CloverCL::bottom_recv_buffer, CL_FALSE, 0,
                                                     *num_elements*sizeof(double), host_bottom_rcv_buffer, NULL, NULL);
    }

    // if top exchage then enqueue a buffer write to transfer the data to  the card 
    if ( *top_neighbour == CloverCL::external_face) {

        CloverCL::outoforder_queue.enqueueWriteBuffer(CloverCL::top_recv_buffer, CL_FALSE, 0,
                                                     *num_elements*sizeof(double), host_top_rcv_buffer, NULL, NULL);
    }

    switch(*nameoffield) {
        case FIELD_DENSITY0: field_buffer = &CloverCL::density0_buffer; break;
        case FIELD_DENSITY1: field_buffer = &CloverCL::density1_buffer; break;
        case FIELD_ENERGY0: field_buffer = &CloverCL::energy0_buffer; break;
        case FIELD_ENERGY1: field_buffer = &CloverCL::energy1_buffer; break;
        case FIELD_PRESSURE: field_buffer = &CloverCL::pressure_buffer; break;
        case FIELD_VISCOSITY: field_buffer = &CloverCL::viscosity_buffer; break;
        case FIELD_SOUNDSPEED: field_buffer = &CloverCL::soundspeed_buffer; break;
        case FIELD_XVEL0: field_buffer = &CloverCL::xvel0_buffer; break;
        case FIELD_XVEL1: field_buffer = &CloverCL::xvel1_buffer; break;
        case FIELD_YVEL0: field_buffer = &CloverCL::yvel0_buffer; break;
        case FIELD_YVEL1: field_buffer = &CloverCL::yvel1_buffer; break;
        case FIELD_VOL_FLUX_X: field_buffer = &CloverCL::vol_flux_x_buffer; break;
        case FIELD_VOL_FLUX_Y: field_buffer = &CloverCL::vol_flux_y_buffer; break;
        case FIELD_MASS_FLUX_X: field_buffer = &CloverCL::mass_flux_x_buffer; break;
        case FIELD_MASS_FLUX_Y: field_buffer = &CloverCL::mass_flux_y_buffer; break;
    }

    if (*xinc == 1) {
        launch_width = CloverCL::xmax_plusfive_rounded;
    }
    else {
        launch_width = CloverCL::xmax_plusfour_rounded;
    }

    // call clfinish to sync while the above ops finsih, could if test this possibly 
    CloverCL::outoforder_queue.finish();


    // if bottom exhange enqueue and unpack bottom kernel on the outorder queue
    if ( *bottom_neighbour == CloverCL::external_face) {
        CloverCL::write_bottom_buffer_knl.setArg(0, *depth);
        CloverCL::write_bottom_buffer_knl.setArg(1, *xinc);
        CloverCL::write_bottom_buffer_knl.setArg(2, *yinc);
        CloverCL::write_bottom_buffer_knl.setArg(3, *field_buffer);
        CloverCL::write_bottom_buffer_knl.setArg(4, CloverCL::bottom_recv_buffer);

        CloverCL::outoforder_queue.enqueueNDRangeKernel(CloverCL::write_bottom_buffer_knl, cl::NullRange,
                                                        cl::NDRange(launch_width, *depth),
                                                        cl::NDRange(CloverCL::fixed_wg_min_size_large_dim, CloverCL::fixed_wg_min_size_small_dim),
                                                        NULL, NULL);
    }


    // if top exchange enqueue the unpack top kernel on the outoforder queue
    if ( *top_neighbour == CloverCL::external_face) {
        CloverCL::write_top_buffer_knl.setArg(0, *depth);
        CloverCL::write_top_buffer_knl.setArg(1, *xinc);
        CloverCL::write_top_buffer_knl.setArg(2, *yinc);
        CloverCL::write_top_buffer_knl.setArg(3, *field_buffer);
        CloverCL::write_top_buffer_knl.setArg(4, CloverCL::top_recv_buffer);

        CloverCL::outoforder_queue.enqueueNDRangeKernel(CloverCL::write_top_buffer_knl, cl::NullRange,
                                                        cl::NDRange(launch_width, *depth),
                                                        cl::NDRange(CloverCL::fixed_wg_min_size_large_dim, CloverCL::fixed_wg_min_size_small_dim),
                                                        NULL, NULL);
    }

    // call clfinish on the outof order queue if either top or bottom exchange
    // could possibly just enqueue a barrier and let the update halo do its thing = future opts 
    CloverCL::outoforder_queue.finish();


#if PROFILE_OCL_KERNELS
    timeval t_end;
    gettimeofday(&t_end, NULL);

    std::cout << "[PROFILING]: pack comms buffer left and right OpenCL kernel took "
        << (t_end.tv_usec - t_start.tv_usec)*CloverCL::US_TO_SECONDS
        << " seconds (host time)" << std::endl;
#endif
}
