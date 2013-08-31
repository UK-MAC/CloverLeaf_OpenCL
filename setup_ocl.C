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
 *  @brief OCL setup functions.
 *  @author Andrew Mallinson, David Beckingsale
 *  @details Sets up the OCL environment for CloverLeaf 
*/

#include "CloverCL.h"
#include <iostream>
#include <algorithm>

extern "C" void setup_opencl_(
        char* platform_name,
        char* platform_type,
        int* xmin,
        int* xmax,
        int* ymin,
        int* ymax,
        int* num_states,
        double* g_small,
        double* g_big,
        double* dtmin,
        double* dtc_safe,
        double* dtu_safe,
        double* dtv_safe,
        double* dtdiv_safe);

void setup_opencl_(
        char* platform_name,
        char* platform_type,
        int* xmin,
        int* xmax,
        int* ymin,
        int* ymax,
        int* num_states,
        double* g_small,
        double* g_big,
        double* dtmin,
        double* dtc_safe,
        double* dtu_safe,
        double* dtv_safe,
        double* dtdiv_safe)
{

    std::string platform = platform_name;
    std::string type = platform_type;

    /*
     * Trim the strings...
     */
    std::remove(platform.begin(), platform.end(), ' ');
    std::remove(type.begin(), type.end(), ' ');

#ifdef OCL_VERBOSE
    std::cout << "Strings are " << platform_name << ", and " << platform_type << std::endl;
#endif

    if(platform == "NULL") {
        std::cerr << "[ERROR] No OpenCL_vendor specified, using Intel..." << std::cout;
        platform = "Intel";
    }

    if(type == "NULL") {
        std::cerr << "[ERROR] No OpenCL_type specified, using CPU..." << std::cout;
        type = "CPU";
    }

    CloverCL::init( platform, type, *xmin, *xmax, *ymin, *ymax, *num_states,
            *g_small, *g_big, *dtmin, *dtc_safe, *dtu_safe, *dtv_safe, *dtdiv_safe);
}
