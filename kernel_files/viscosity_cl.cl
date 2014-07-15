__kernel void viscosity
(__global const double * __restrict const celldx,
 __global const double * __restrict const celldy,
 __global const double * __restrict const celldz,
 __global const double * __restrict const density0,
 __global const double * __restrict const pressure,
 __global double * __restrict const viscosity,
 __global const double * __restrict const xvel0,
 __global const double * __restrict const yvel0,
 __global const double * __restrict const zvel0)
{
    __kernel_indexes;

    double ugrad, vgrad,wgrad, grad2, pgradx, pgrady,pgradz, pgradx2, pgrady2,pgradz2,
        grad, ygrad, pgrad, xgrad,zgrad, div, strain2, limiter;

    if(/*row >= (y_min + 1) && */ row <= (y_max + 1)
    && /*column >= (x_min + 1) && */ column <= (x_max + 1)
    && /*slice >= (z_min + 1) && */ slice <= (z_max + 1))
    {
        ugrad=0.5*((xvel0[THARR3D(1,0 ,0,1,1 )]+xvel0[THARR3D(1,1,0,1,1 )]+xvel0[THARR3D(1,0 ,1,1,1)]+xvel0[THARR3D(1,1,1,1,1)])
		-(xvel0[THARR3D(0 ,0 ,0,1,1 )]+xvel0[THARR3D(0 ,1,0,1,1 )]+xvel0[THARR3D(0 ,0 ,1,1,1)]+xvel0[THARR3D(0 ,1,1,1,1)]));

        vgrad=0.5*((yvel0[THARR3D(0 ,1,0,1,1 )]+yvel0[THARR3D(1,1,0,1,1 )]+yvel0[THARR3D(0 ,1,1,1,1)]+yvel0[THARR3D(1,1,1,1,1)])
		-(yvel0[THARR3D(0 ,0 ,0,1,1 )]+yvel0[THARR3D(1,0 ,0,1,1 )]+yvel0[THARR3D(0 ,0 ,1,1,1)]+yvel0[THARR3D(1,0 ,1,1,1)]));

        wgrad=0.5*((zvel0[THARR3D(0 ,0 ,1,1,1)]+zvel0[THARR3D(1,1,1,1,1)]+zvel0[THARR3D(0 ,0 ,1,1,1)]+zvel0[THARR3D(1,1,1,1,1)])
		-(zvel0[THARR3D(0 ,0,0,1,1 )]+zvel0[THARR3D(1,0 ,0,1,1 )]+zvel0[THARR3D(0 ,1,0,1,1 )]+zvel0[THARR3D(1,1,0,1,1 )]));

        div = (celldx[column]*(ugrad)+  celldy[row]*(vgrad))+ celldz[slice]*(wgrad);

//Double check that (celldy[row]*celldz[slice])==yarea, etc

        strain2 = 0.5*(xvel0[THARR3D(0, 1,0,1,1 )] + xvel0[THARR3D(1,1,1,1,1)]-xvel0[THARR3D(0 ,0 ,0,1,1)]-xvel0[THARR3D(1,0 ,0,1,1 )])/(celldy[row]*celldz[slice])
		+ 0.5*(yvel0[THARR3D(1,0 ,0,1,1 )] + yvel0[THARR3D(1,1,1,1,1)]-yvel0[THARR3D(0 ,0 ,0,1,1)]-yvel0[THARR3D(0 ,1,0,1,1 )])/(celldx[column]*celldz[slice])
		+ 0.5*(zvel0[THARR3D(0 ,0 ,1,1,1)] + zvel0[THARR3D(1,1,1,1,1)]-zvel0[THARR3D(0 ,0 ,0,1,1)]-zvel0[THARR3D(0 ,0 ,1,1,1)])/(celldy[row]*celldx[column]);


        pgradx = (pressure[THARR3D(1, 0, 0,0,0)] - pressure[THARR3D(-1, 0, 0,0,0)])
               / (celldx[column] + celldx[column + 1]);
        pgrady = (pressure[THARR3D(0, 1, 0,0,0)] - pressure[THARR3D(0, -1, 0,0,0)])
               / (celldy[row] + celldy[row + 1]);
        pgradz = (pressure(0,0,1,0,0)-pressure(0,0,-1,0,0))/(celldz[slice] + celldz[slice + 1]);

        pgradx2 = pgradx*pgradx;
        pgrady2 = pgrady*pgrady;
        pgradz2 = pgradz*pgradz;

        limiter = ((0.5*(ugrad)/celldx(column))*pgradx2
		+ (0.5*(vgrad)/celldy(row))*pgrady2
		+(0.5*(wgrad)/celldz(slice))*pgradz2
		+strain2*pgradx*pgrady*pgradz)
		/ MAX(pgradx2+pgrady2+pgradz2,1.0e-16);


        pgradx = SIGN(MAX(1.0e-16, fabs(pgradx)), pgradx);
        pgrady = SIGN(MAX(1.0e-16, fabs(pgrady)), pgrady);
        pgradz = SIGN(MAX(1.0e-16, fabs(pgradz)), pgradz);
        pgrad = SQRT(pgradx*pgradx+pgrady*pgrady+pgradz*pgradz);
        xgrad = fabs(celldx[column] * pgrad / pgradx);
        ygrad = fabs(celldy[row] * pgrad / pgrady);
        zgrad = fabs(celldz[slice]*pgrad / pgradz);

        grad = MIN(xgrad, MIN(ygrad,zgrad));
        grad2 = grad * grad;

        if(limiter > 0 || div >= 0.0)
        {
            viscosity[THARR3D(0,0,0,0,0)] = 0.0;
        }
        else
        {
            viscosity[THARR3D(0,0,0,0,0)] = 2.0 * density0[THARR3D(0,0,0,0,0)] * grad2 * (limiter * limiter);
        }
    }
}
