#include "./kernel_files/macros_cl.cl"
__kernel void viscosity
(__global const double * __restrict const celldx,
 __global const double * __restrict const celldy,
 __global const double * __restrict const celldz,
 __global const double * __restrict const density0,
 __global const double * __restrict const pressure,
 __global double * __restrict const viscosity,
 __global const double * __restrict const xvel0,
 __global const double * __restrict const yvel0,
 __global const double * __restrict const zvel0,
 __global const double * __restrict const xarea,
 __global const double * __restrict const yarea,
 __global const double * __restrict const zarea)
{
    __kernel_indexes;

    if(/*row >= (y_min + 1) && */ row <= (y_max + 1)
    && /*column >= (x_min + 1) && */ column <= (x_max + 1)
    && /*slice >= (z_min + 1) && */ slice <= (z_max + 1))
    {
        #define GRADVARS(uvw, xyz) \
        const double uvw##gradx1 = (xyz##vel0[THARR3D(0, 0, 0, 1, 1)]+xyz##vel0[THARR3D(0, 1, 0, 1, 1)]+xyz##vel0[THARR3D(0, 0, 1, 1, 1)]+xyz##vel0[THARR3D(0, 1, 1, 1, 1)]);   \
		const double uvw##gradx2 = (xyz##vel0[THARR3D(1, 0, 0, 1, 1)]+xyz##vel0[THARR3D(1, 1, 0, 1, 1)]+xyz##vel0[THARR3D(1, 0, 1, 1, 1)]+xyz##vel0[THARR3D(1, 1, 1, 1, 1)]);   \
        const double uvw##grady1 = (xyz##vel0[THARR3D(0, 0, 0, 1, 1)]+xyz##vel0[THARR3D(1, 0, 0, 1, 1)]+xyz##vel0[THARR3D(0, 0, 1, 1, 1)]+xyz##vel0[THARR3D(1, 0, 1, 1, 1)]);   \
		const double uvw##grady2 = (xyz##vel0[THARR3D(0, 1, 0, 1, 1)]+xyz##vel0[THARR3D(1, 1, 0, 1, 1)]+xyz##vel0[THARR3D(0, 1, 1, 1, 1)]+xyz##vel0[THARR3D(1, 1, 1, 1, 1)]);   \
        const double uvw##gradz1 = (xyz##vel0[THARR3D(0, 0, 0, 1, 1)]+xyz##vel0[THARR3D(1, 0, 0, 1, 1)]+xyz##vel0[THARR3D(0, 1, 0, 1, 1)]+xyz##vel0[THARR3D(1, 1, 0, 1, 1)]);   \
		const double uvw##gradz2 = (xyz##vel0[THARR3D(0, 0, 1, 1, 1)]+xyz##vel0[THARR3D(1, 0, 1, 1, 1)]+xyz##vel0[THARR3D(0, 1, 1, 1, 1)]+xyz##vel0[THARR3D(1, 1, 1, 1, 1)]);

        GRADVARS(u, x)

XEON_PHI_LOCAL_MEM_BARRIER;

        GRADVARS(v, y)

XEON_PHI_LOCAL_MEM_BARRIER;

        GRADVARS(w, z)

XEON_PHI_LOCAL_MEM_BARRIER;

        const double div = xarea[THARR3D(0, 0, 0, 1, 0)]*(ugradx2 - ugradx1) +
                           yarea[THARR3D(0, 0, 0, 0, 1)]*(vgrady2 - vgrady1) +
                           zarea[THARR3D(0, 0, 0, 0, 0)]*(wgradz2 - wgradz1);

        const double xx = 0.25*(ugradx2 - ugradx1)/celldx[column];
        const double yy = 0.25*(vgrady2 - vgrady1)/celldy[row];
        const double zz = 0.25*(wgradz2 - wgradz1)/celldz[slice];

        const double xy = 0.25*(ugrady2 - ugrady1)/celldy[row]   + 0.25*(vgradx2 - vgradx1)/celldx[column];
        const double xz = 0.25*(ugradz2 - ugradz1)/celldz[slice] + 0.25*(wgradx2 - wgradx1)/celldx[column];
        const double yz = 0.25*(vgradz2 - vgradz1)/celldz[slice] + 0.25*(wgrady2 - wgrady1)/celldy[row];

        double pgradx = (pressure[THARR3D(1, 0, 0, 0, 0)] - pressure[THARR3D(-1, 0, 0, 0, 0)]) / (celldx[column] + celldx[column + 1]);
        double pgrady = (pressure[THARR3D(0, 1, 0, 0, 0)] - pressure[THARR3D(0, -1, 0, 0, 0)]) / (celldy[row]    + celldy[row + 1]);
        double pgradz = (pressure[THARR3D(0, 0, 1, 0, 0)] - pressure[THARR3D(0, 0, -1, 0, 0)]) / (celldz[slice]  + celldz[slice + 1]);

        const double pgradx2 = pgradx*pgradx;
        const double pgrady2 = pgrady*pgrady;
        const double pgradz2 = pgradz*pgradz;

        const double limiter = (xx*pgradx2 + yy*pgrady2 + zz*pgradz2 +
            xy*pgradx*pgrady + xz*pgradx*pgradz + yz*pgrady*pgradz) /
            MAX(pgradx2+pgrady2+pgradz2, 1.0e-16);

        if(limiter > 0 || div >= 0.0)
        {
            viscosity[THARR3D(0, 0, 0, 0, 0)] = 0.0;
        }
        else
        {
            pgradx = SIGN(MAX(1.0e-16, fabs(pgradx)), pgradx);
            pgrady = SIGN(MAX(1.0e-16, fabs(pgrady)), pgrady);
            pgradz = SIGN(MAX(1.0e-16, fabs(pgradz)), pgradz);
            const double pgrad = SQRT(pgradx*pgradx + pgrady*pgrady + pgradz*pgradz);
            const double xgrad = fabs(celldx[column]*pgrad/pgradx);
            const double ygrad = fabs(celldy[row]   *pgrad/pgrady);
            const double zgrad = fabs(celldz[slice] *pgrad/pgradz);

            const double grad = MIN(xgrad, MIN(ygrad, zgrad));
            const double grad2 = grad*grad;

            viscosity[THARR3D(0, 0, 0, 0, 0)] = 2.0*density0[THARR3D(0, 0, 0, 0, 0)]*grad2*(limiter*limiter);
        }
    }
}
