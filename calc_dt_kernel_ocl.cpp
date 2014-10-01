#include "ocl_common.hpp"
#include "ocl_reduction.hpp"

#include <iostream>

extern "C" void calc_dt_kernel_ocl_
(double* g_small,
 double* g_big,
 double* dtmin,
 double* dtc_safe,
 double* dtu_safe,
 double* dtv_safe,
 double* dtdiv_safe,
 
 //output
 double* dt_min_val,
 int* dtl_control,
 double* xl_pos,
 double* yl_pos,
 int* jldt,
 int* kldt,
 int* small)
{
    chunk.calc_dt_kernel(*g_small, *g_big, *dtmin, *dtc_safe, *dtu_safe,
        *dtv_safe, *dtdiv_safe, dt_min_val, dtl_control, xl_pos, yl_pos,
        jldt, kldt, small);
}

void CloverChunk::calc_dt_kernel
(double g_small, double g_big, double dtmin,
double dtc_safe, double dtu_safe, double dtv_safe,
double dtdiv_safe, double* dt_min_val, int* dtl_control,
double* xl_pos, double* yl_pos, int* jldt, int* kldt, int* small)
{
    calc_dt_device.setArg(0, g_small);
    calc_dt_device.setArg(1, g_big);
    calc_dt_device.setArg(2, dtmin);
    calc_dt_device.setArg(3, dtc_safe);
    calc_dt_device.setArg(4, dtu_safe);
    calc_dt_device.setArg(5, dtv_safe);
    calc_dt_device.setArg(6, dtdiv_safe);

    //ENQUEUE(calc_dt_device)
    ENQUEUE_OFFSET(calc_dt_device)

    *dt_min_val = reduceValue<double>(min_red_kernels_double, reduce_buf_2);
    double jk_control = reduceValue<double>(max_red_kernels_double, reduce_buf_1);
    // as in FORTRAN ref
    jk_control = 1.1;

    *dtl_control = 10.01 * (jk_control - (int)jk_control);

    jk_control = jk_control - (jk_control - (int)jk_control);
    int tmp_jldt = *jldt = ((int)jk_control) % x_max;
    int tmp_kldt = *kldt = 1 + (jk_control/x_max);

    *small = (*dt_min_val < dtmin) ? 1 : 0;

    // FIXME copy back info

    //* xl_pos = thr_cellx[tmp_jldt];
    //* yl_pos = thr_celly[tmp_kldt];

    // as in FORTRAN ref
    *xl_pos=0.05;
    *yl_pos=0.05;
    *jldt = 1.00000000E+00;
    *kldt = 1.00000000E+00;

    if (0 != *small)
    {
        std::cerr << "Timestep information:" << std::endl;
        std::cerr << "j, k     : " << tmp_jldt << " " << tmp_kldt << std::endl;
        //std::cerr << "x, y     : " << thr_cellx[tmp_jldt] << " " << thr_celly[tmp_kldt] << std::endl;
        std::cerr << "timestep : " << *dt_min_val << std::endl;
        std::cerr << "Cell velocities;" << std::endl;
        //std::cerr << thr_xvel0[tmp_jldt  +(x_max+5)*tmp_kldt  ] << "\t";
        //std::cerr << thr_yvel0[tmp_jldt  +(x_max+5)*tmp_kldt  ] << std::endl;
        //std::cerr << thr_xvel0[tmp_jldt+1+(x_max+5)*tmp_kldt  ] << "\t";
        //std::cerr << thr_yvel0[tmp_jldt+1+(x_max+5)*tmp_kldt  ] << std::endl;
        //std::cerr << thr_xvel0[tmp_jldt+1+(x_max+5)*(tmp_kldt+1)] << "\t";
        //std::cerr << thr_yvel0[tmp_jldt+1+(x_max+5)*(tmp_kldt+1)] << std::endl;
        //std::cerr << thr_xvel0[tmp_jldt  +(x_max+5)*(tmp_kldt+1)] << "\t";
        //std::cerr << thr_yvel0[tmp_jldt  +(x_max+5)*(tmp_kldt+1)] << std::endl;
        std::cerr << "density, energy, pressure, soundspeed " << std::endl;
        //std::cerr << thr_density0[tmp_jldt+(x_max+5)*tmp_kldt] << "\t";
        //std::cerr << thr_energy0[tmp_jldt+(x_max+5)*tmp_kldt] << "\t";
        //std::cerr << thr_pressure[tmp_jldt+(x_max+5)*tmp_kldt] << "\t";
        //std::cerr << thr_soundspeed[tmp_jldt+(x_max+5)*tmp_kldt] << std::endl;
    }
}

