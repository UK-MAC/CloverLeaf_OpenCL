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
 *  @brief OCL host-side read OCL buffers kernel.
 *  @author Andrew Mallinson, David Beckingsale
 *  @details Launches the OCL device-side read OCL buffers kernels 
*/

#include "CloverCL.h"

extern "C" void ocl_read_back_all_buffers_(double* density0, double* density1, double* energy0, double* energy1,
                                          double* pressure, double* viscosity, double* soundspeed, 
                                          double* xvel0, double* xvel1, double* yvel0, double* yvel1,
                                          double* vol_flux_x, double* mass_flux_x, 
                                          double* vol_flux_y, double* mass_flux_y,
                                          double* celldx, double* celldy, double* volume );

extern "C" void ocl_write_back_all_buffers_(double* density0, double* density1, double* energy0, double* energy1,
                                            double* pressure, double* viscosity, double* soundspeed, 
                                            double* xvel0, double* xvel1, double* yvel0, double* yvel1,
                                            double* vol_flux_x, double* mass_flux_x, 
                                            double* vol_flux_y, double* mass_flux_y,
                                            double* celldx, double* celldy, double* volume );

extern "C" void ocl_read_vis_buffers_(int* x_max, int* y_max, double* vertexx, double* vertexy, 
                                      double* density0, double* energy0, double* pressure, 
                                      double* viscosity, double* xvel0, double* yvel0);

extern "C" void ocl_read_all_comm_buffers_(int* x_max, int* y_max, double* denisty0,
                                           double* density1, double* energy0, double* energy1, 
                                           double* pressure, double* viscosity, double* soundspeed, 
                                           double* xvel0, double* xvel1, double* yvel0, double* yvel1, 
                                           double* vol_flux_x, double* vol_flux_y, double* mass_flux_x,
                                           double* mass_flux_y);

extern "C" void ocl_write_all_comm_buffers_(
        int* x_max,
        int* y_max,
        double* denisty0,
        double* density1,
        double* energy0,
        double* energy1,
        double* pressure,
        double* viscosity,
        double* soundspeed,
        double* xvel0,
        double* xvel1,
        double* yvel0,
        double* yvel1,
        double* vol_flux_x,
        double* vol_flux_y,
        double* mass_flux_x,
        double* mass_flux_y);

extern "C" void ocl_read_comm_buffer_(
        int* xmin,
        int* xmax,
        int* ymin,
        int* ymax,
        int* depth,
        int* xinc,
        int* yinc,
        int* field_name,
        double* buffer,
        int* which_edge);

extern "C" void ocl_write_comm_buffer_(
        int* xmin,
        int* xmax,
        int* ymin,
        int* ymax,
        int* depth,
        int* xinc,
        int* yinc,
        int* field_name,
        double* buffer,
        int* which_edge);

void ocl_read_back_all_buffers_(double* density0, double* density1, double* energy0, double* energy1,
                                          double* pressure, double* viscosity, double* soundspeed, 
                                          double* xvel0, double* xvel1, double* yvel0, double* yvel1,
                                          double* vol_flux_x, double* mass_flux_x, 
                                          double* vol_flux_y, double* mass_flux_y,
                                          double* celldx, double* celldy, double* volume  )
{
    CloverCL::read_back_all_ocl_buffers(density0, density1, energy0, energy1, pressure, viscosity, soundspeed, 
                                        xvel0, xvel1, yvel0, yvel1, vol_flux_x, mass_flux_x, vol_flux_y, mass_flux_y,
                                        celldx, celldy, volume);
}

void ocl_write_back_all_buffers_(double* density0, double* density1, double* energy0, double* energy1,
                                            double* pressure, double* viscosity, double* soundspeed, 
                                            double* xvel0, double* xvel1, double* yvel0, double* yvel1,
                                            double* vol_flux_x, double* mass_flux_x, 
                                            double* vol_flux_y, double* mass_flux_y,
                                            double* celldx, double* celldy, double* volume  )
{
    CloverCL::write_back_all_ocl_buffers(density0, density1, energy0, energy1, pressure, viscosity, soundspeed, 
                                         xvel0, xvel1, yvel0, yvel1, vol_flux_x, mass_flux_x, vol_flux_y, mass_flux_y,
                                         celldx, celldy, volume);
}

void ocl_read_vis_buffers_(
        int* x_max,
        int* y_max,
        double* vertexx,
        double* vertexy,
        double* density0,
        double* energy0,
        double* pressure,
        double* viscosity,
        double* xvel0,
        double* yvel0)
{
    CloverCL::readVisualisationBuffers(
            *x_max,
            *y_max,
            vertexx,
            vertexy,
            density0,
            energy0,
            pressure,
            viscosity,
            xvel0,
            yvel0);
}

void ocl_read_comm_buffer_(
        int* xmin,
        int* xmax,
        int* ymin,
        int* ymax,
        int* depth,
        int* xinc,
        int* yinc,
        int* field_name,
        double* buffer,
        int* which_edge)
{
    CloverCL::readCommunicationBuffer(
        xmin,
        xmax,
        ymin,
        ymax,
        depth,
        xinc,
        yinc,
        field_name,
        buffer,
        which_edge);
}

void ocl_write_comm_buffer_(
        int* xmin,
        int* xmax,
        int* ymin,
        int* ymax,
        int* depth,
        int* xinc,
        int* yinc,
        int* field_name,
        double* buffer,
        int* which_edge)
{
    CloverCL::writeCommunicationBuffer(
        xmin,
        xmax,
        ymin,
        ymax,
        depth,
        xinc,
        yinc,
        field_name,
        buffer,
        which_edge);
}

void ocl_read_all_comm_buffers_(
        int* x_max,
        int* y_max,
        double* denisty0,
        double* density1,
        double* energy0,
        double* energy1,
        double* pressure,
        double* viscosity,
        double* soundspeed,
        double* xvel0,
        double* xvel1,
        double* yvel0,
        double* yvel1,
        double* vol_flux_x,
        double* vol_flux_y,
        double* mass_flux_x,
        double* mass_flux_y)
{
    CloverCL::readAllCommunicationBuffers(
        x_max,
        y_max,
        denisty0,
        density1,
        energy0,
        energy1,
        pressure,
        viscosity,
        soundspeed,
        xvel0,
        xvel1,
        yvel0,
        yvel1,
        vol_flux_x,
        vol_flux_y,
        mass_flux_x,
        mass_flux_y);
}

void ocl_write_all_comm_buffers_(
        int* x_max,
        int* y_max,
        double* denisty0,
        double* density1,
        double* energy0,
        double* energy1,
        double* pressure,
        double* viscosity,
        double* soundspeed,
        double* xvel0,
        double* xvel1,
        double* yvel0,
        double* yvel1,
        double* vol_flux_x,
        double* vol_flux_y,
        double* mass_flux_x,
        double* mass_flux_y)
{
    CloverCL::writeAllCommunicationBuffers(
        x_max,
        y_max,
        denisty0,
        density1,
        energy0,
        energy1,
        pressure,
        viscosity,
        soundspeed,
        xvel0,
        xvel1,
        yvel0,
        yvel1,
        vol_flux_x,
        vol_flux_y,
        mass_flux_x,
        mass_flux_y);
}
