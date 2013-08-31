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
 *  @brief OCL host-side generate chunk kernel.
 *  @author Andrew Mallinson, David Beckingsale
 *  @details Launches the OCL device-side generate chunk kernel 
*/

#include "CloverCL.h"

#include <iostream>

#include <sys/time.h>

extern "C" void generate_chunk_kernel_ocl_(
        int *xmin,
        int *xmax,
        int *ymin,
        int *ymax,
        int *nm_stes,
        double *state_density,
        double *state_energy,
        double *state_xvel,
        double *state_yvel,
        double *state_xmin,
        double *state_xmax,
        double *state_ymin,
        double *state_ymax,
        double *state_radius,
        int *state_geometry,
        int *g_rect,
        int *g_circ);

void generate_chunk_kernel_ocl_(
        int *xmin,
        int *xmax,
        int *ymin,
        int *ymax,
        int *nm_stes,
        double *state_density,
        double *state_energy,
        double *state_xvel,
        double *state_yvel,
        double *state_xmin,
        double *state_xmax,
        double *state_ymin,
        double *state_ymax,
        double *state_radius,
        int *state_geometry,
        int *g_rect,
        int *g_circ)
{
#if PROFILE_OCL_KERNELS
    timeval t_start;
    gettimeofday(&t_start, NULL);
#endif

    cl::Event event1, event2, event3, event4, event5, event6, event7, event8, event9, event10;
    std::vector<cl::Event> events;

#ifdef OCL_VERBOSE
    std::cout << "Num states = " << *nm_stes << std::endl;
#endif

    try {
        CloverCL::generate_chunk_knl.setArg(8, *nm_stes);
        CloverCL::generate_chunk_knl.setArg(19, *g_rect);
        CloverCL::generate_chunk_knl.setArg(20, *g_circ);
    } catch (cl::Error err) {
        CloverCL::reportError(err, " generate_chunk_knl setting arguments");
    }

    try {
        CloverCL::queue.enqueueWriteBuffer(
                CloverCL::state_density_buffer,
                CL_FALSE,
                0,
                *nm_stes*sizeof(double),
                state_density,
                NULL,
                &event1);

        CloverCL::queue.enqueueWriteBuffer(
                CloverCL::state_energy_buffer,
                CL_FALSE,
                0,
                *nm_stes*sizeof(double),
                state_energy,
                NULL,
                &event2);

        CloverCL::queue.enqueueWriteBuffer(
                CloverCL::state_xvel_buffer,
                CL_FALSE,
                0,
                *nm_stes*sizeof(double),
                state_xvel,
                NULL,
                &event3);

        CloverCL::queue.enqueueWriteBuffer(
                CloverCL::state_yvel_buffer,
                CL_FALSE,
                0,
                *nm_stes*sizeof(double),
                state_yvel,
                NULL,
                &event4);

        CloverCL::queue.enqueueWriteBuffer(
                CloverCL::state_xmin_buffer,
                CL_FALSE,
                0,
                *nm_stes*sizeof(double),
                state_xmin,
                NULL,
                &event5);

        CloverCL::queue.enqueueWriteBuffer(
                CloverCL::state_xmax_buffer,
                CL_FALSE,
                0,
                *nm_stes*sizeof(double),
                state_xmax,
                NULL,
                &event6);

        CloverCL::queue.enqueueWriteBuffer(
                CloverCL::state_ymin_buffer,
                CL_FALSE,
                0,
                *nm_stes*sizeof(double),
                state_ymin,
                NULL,
                &event7);

        CloverCL::queue.enqueueWriteBuffer(
                CloverCL::state_ymax_buffer,
                CL_FALSE,
                0,
                *nm_stes*sizeof(double),
                state_ymax,
                NULL,
                &event8);

        CloverCL::queue.enqueueWriteBuffer(
                CloverCL::state_radius_buffer,
                CL_FALSE,
                0,
                *nm_stes*sizeof(double),
                state_radius,
                NULL,
                &event9);

        CloverCL::queue.enqueueWriteBuffer(
                CloverCL::state_geometry_buffer,
                CL_FALSE,
                0,
                *nm_stes*sizeof(int),
                state_geometry,
                NULL,
                &event10);
    } catch (cl::Error err) {
        CloverCL::reportError(err, "generate_chunk_knl filling buffers");
    }

    events.push_back(event1);
    events.push_back(event2);
    events.push_back(event3);
    events.push_back(event4);
    events.push_back(event5);
    events.push_back(event6);
    events.push_back(event7);
    events.push_back(event8);
    events.push_back(event9);
    events.push_back(event10);

    cl::Event::waitForEvents(events);
    events.clear();

    CloverCL::enqueueKernel( CloverCL::generate_chunk_knl, *xmin, *xmax+4, *ymin, *ymax+4);

#if PROFILE_OCL_KERNELS
    timeval t_end;
    gettimeofday(&t_end, NULL);

    std::cout << "[PROFILING]: generate_chunk OpenCL kernel took "
        << (t_end.tv_usec - t_start.tv_usec)*CloverCL::US_TO_SECONDS
        << " seconds (host time)" << std::endl;
#endif
}
