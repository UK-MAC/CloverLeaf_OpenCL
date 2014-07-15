__kernel void calc_dt
(const double g_small,
 const double g_big,
 const double dtmin,
 const double dtc_safe,
 const double dtu_safe,
 const double dtv_safe,
 const double dtw_safe,
 const double dtdiv_safe,

 __global const double * __restrict const xarea,
 __global const double * __restrict const yarea,
 __global const double * __restrict const zarea,
 __global const double * __restrict const celldx,
 __global const double * __restrict const celldy,
 __global const double * __restrict const celldz,
 __global const double * __restrict const volume,
 __global const double * __restrict const density0,
 __global const double * __restrict const viscosity,
 __global const double * __restrict const soundspeed,
 __global const double * __restrict const xvel0,
 __global const double * __restrict const yvel0,
 __global const double * __restrict const zvel0,

 __global double * __restrict const jk_ctrl_out,
 __global double * __restrict const dt_min_out)
{
    __kernel_indexes;

    double dsx, dsy,dsz;
    double cc;
    double dtct;
    double div;
    double dv1;
    double dv2;
    double dtut;
    double dtvt;
    double dtwt;
    double dtdivt;

    //reduced
    double dt_min_val = g_big;
    double jk_control = 0.0;

#if defined(NO_KERNEL_REDUCTIONS)
    dt_min_out[gid] = dt_min_val;
    jk_ctrl_out[gid] = jk_control;
#else
    __local double dt_min_shared[BLOCK_SZ];
    __local double jk_ctrl_shared[BLOCK_SZ];
    dt_min_shared[lid] = dt_min_val;
    jk_ctrl_shared[lid] = jk_control;
#endif

    if(row >= (y_min + 1) && row <= (y_max + 1)
    && column >= (x_min + 1) && column <= (x_max + 1)
    && slice >= (z_min + 1) && slice <= (z_max + 1))
    {
        dsx = celldx[column];
        dsy = celldy[row];
        dsz = celldz[slice];

        cc = soundspeed[THARR3D(0, 0, 0,0,0)] * soundspeed[THARR3D(0, 0, 0,0,0)];
        cc += 2.0 * viscosity[THARR3D(0, 0, 0,0,0)] / density0[THARR3D(0, 0, 0,0,0)];
        cc = MAX(SQRT(cc), g_small);

        dtct = dtc_safe * MIN(dsx, MIN(dsy,dsz))/cc;

        div = 0.0;

        //x
        dv1=(xvel0[THARR3D(0  ,0  ,0,1,1  )]+xvel0[THARR3D(0  ,1,0,1,1  )]+xvel0[THARR3D(0  ,0  ,1,1,1)]+xvel0[THARR3D(0  ,1,1,1,1)])*xarea[THARR3D(0  ,0  ,0,1,0  )];

        dv2=(xvel0(1,0  ,0,1,1  )+xvel0(1,1,0,1,1  )+xvel0(1,0  ,1,1,1)+xvel0(1,1,1,1,1))*xarea(1,0  ,0,1,0  );

        div += dv2 - dv1;

        dtut = dtu_safe * 2.0 * volume[THARR3D(0, 0, 0,0,0)]
            / MAX(g_small*volume[THARR3D(0, 0, 0,0,0)],
            MAX(fabs(dv1), fabs(dv2)));

        //y
        dv1=(yvel0[THARR3D(0  ,0  ,0,1,1  )]+yvel0[THARR3D(1,0  ,0,1,1  )]+yvel0[THARR3D(0  ,0  ,1,1,1)]+yvel0[THARR3D(1,0  ,1,1,1)])*yarea[THARR3D(0  ,0  ,0,0,1  )];

        dv2=(yvel0[THARR3D(0  ,1,0,1,1  )]+yvel0[THARR3D(1,1,0,1,1  )]+yvel0[THARR3D(0  ,1,1,1,1)]+yvel0[THARR3D(1,1,1,1,1)])*yarea[THARR3D(0  ,1,0,0,1  )];

        div += dv2 - dv1;

        dtvt = dtv_safe * 2.0 * volume[THARR3D(0, 0, 0,0,0)]
            / MAX(g_small*volume[THARR3D(0, 0, 0,0,0)],
            MAX(fabs(dv1), fabs(dv2)));

        //z
        dv1=(zvel0[THARR3D(0  ,0  ,0,1,1  )]+zvel0[THARR3D(1,0  ,0,1,1  )]+zvel0[THARR3D(0  ,1  ,0,1,1)]+zvel0[THARR3D(1,1  ,0,1,1)])*zarea[THARR3D(0  ,0  ,0,0,0  )];

        dv2=(zvel0[THARR3D(0  ,0,1,1,1  )]+zvel0[THARR3D(1,0,1,1,1  )]+zvel0[THARR3D(0  ,1,1,1,1)]+zvel0[THARR3D(1,1,1,1,1)])*zarea[THARR3D(0  ,0,1,0,1  )];

        div += dv2 - dv1;

        dtwt = dtw_safe * 2.0 * volume[THARR3D(0, 0, 0,0,0)]
            / MAX(g_small*volume[THARR3D(0, 0, 0,0,0)],
            MAX(fabs(dv1), fabs(dv2)));


        //
        div /= (2.0 * volume[THARR3D(0, 0, 0,0,0)]);

        dtdivt = (div < (-g_small)) ? dtdiv_safe * (-1.0/div) : g_big;

#if defined(NO_KERNEL_REDUCTIONS)
        dt_min_out[gid] = MIN(dtdivt, MIN(dtvt, MIN(dtct, MIN(dtut,dtwt))));
//THIS NEEDS FIXING
        jk_ctrl_out[gid] = (column + (x_max * (row - 1))) + 0.4;
#else
        dt_min_shared[lid] = MIN(dtdivt, MIN(dtvt, MIN(dtct, MIN(dtut,dtwt))));
//THIS NEEDS FIXING
        jk_ctrl_shared[lid] = (column + (x_max * (row - 1))) + 0.4;
#endif
    }

#if !defined(NO_KERNEL_REDUCTIONS)
    REDUCTION(dt_min_shared, dt_min_out, MIN)
    REDUCTION(jk_ctrl_shared, jk_ctrl_out, MAX)
#endif
}
