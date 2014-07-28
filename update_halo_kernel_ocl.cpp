#include "ocl_common.hpp"
extern CloverChunk chunk;

// types of array data

const static cell_info_t CELL(    0, 0, 0, 1, 1, 1, 0, 0,0, CELL_DATA);
const static cell_info_t VERTEX_X(1, 1, 1,-1, 1, 1, 0, 0, 0, VERTEX_DATA);
const static cell_info_t VERTEX_Y(1, 1, 1, 1,-1, 1, 0, 0, 0, VERTEX_DATA);
const static cell_info_t VERTEX_Z(1, 1, 1, 1, 1,-1, 0, 0, 0, VERTEX_DATA);
const static cell_info_t X_FACE(  1, 0, 0,-1, 1, 1, 1, 0, 0, X_FACE_DATA);
const static cell_info_t Y_FACE(  0, 1, 0, 1,-1, 1, 0, 1, 0, Y_FACE_DATA);
const static cell_info_t Z_FACE(  0, 0, 1, 1, 1,-1, 0, 0, 1, Z_FACE_DATA);

extern "C" void update_halo_kernel_ocl_
(const int* chunk_neighbours,
const int* fields,
const int* depth)
{
    chunk.update_halo_kernel(fields, *depth, chunk_neighbours);
}

void CloverChunk::update_array
(cl::Buffer& cur_array,
const cell_info_t& array_type,
const int* chunk_neighbours,
int depth)
{
    // could do clenqueuecopybufferrect, but it's blocking and would be slow

    // could do offset launch for updating bottom/right, but dont to keep parity with cuda
    #define CHECK_LAUNCH(face, dir) \
    if(chunk_neighbours[CHUNK_ ## face - 1] == EXTERNAL_FACE)\
    {\
        update_halo_##face##_device.setArg(0, array_type.x_extra); \
        update_halo_##face##_device.setArg(1, array_type.y_extra); \
        update_halo_##face##_device.setArg(2, array_type.z_extra); \
        update_halo_##face##_device.setArg(3, array_type.x_invert); \
        update_halo_##face##_device.setArg(4, array_type.y_invert); \
        update_halo_##face##_device.setArg(5, array_type.z_invert); \
        update_halo_##face##_device.setArg(6, array_type.x_face); \
        update_halo_##face##_device.setArg(7, array_type.y_face); \
        update_halo_##face##_device.setArg(8, array_type.z_face); \
        update_halo_##face##_device.setArg(9, array_type.grid_type); \
        update_halo_##face##_device.setArg(10, depth); \
        update_halo_##face##_device.setArg(11, cur_array); \
        CloverChunk::enqueueKernel(update_halo_##face##_device, \
            __LINE__, __FILE__, \
            cl::NullRange, \
            update_##dir##_global_size[depth-1], \
            update_##dir##_local_size[depth-1]); \
    }

    CHECK_LAUNCH(left, lr)
    CHECK_LAUNCH(right, lr)
    CHECK_LAUNCH(top, ud)
    CHECK_LAUNCH(bottom, ud)
    CHECK_LAUNCH(back, fb)
    CHECK_LAUNCH(front, fb)
}

void CloverChunk::update_halo_kernel
(const int* fields,
const int depth,
const int* chunk_neighbours)
{
double test;
#define HALO_UPDATE_RESIDENT(arr, type) \
if(fields[FIELD_ ## arr - 1] == 1) \
{ \
update_array(arr, type, chunk_neighbours, depth); \
}

    HALO_UPDATE_RESIDENT(density0, CELL);
    HALO_UPDATE_RESIDENT(density1, CELL);
    HALO_UPDATE_RESIDENT(energy0, CELL);
    HALO_UPDATE_RESIDENT(energy1, CELL);
    HALO_UPDATE_RESIDENT(pressure, CELL);
    HALO_UPDATE_RESIDENT(viscosity, CELL);

    HALO_UPDATE_RESIDENT(xvel0, VERTEX_X);
    HALO_UPDATE_RESIDENT(xvel1, VERTEX_X);

    HALO_UPDATE_RESIDENT(yvel0, VERTEX_Y);
    HALO_UPDATE_RESIDENT(yvel1, VERTEX_Y);

    HALO_UPDATE_RESIDENT(zvel0, VERTEX_Z);
    HALO_UPDATE_RESIDENT(zvel1, VERTEX_Z);

    HALO_UPDATE_RESIDENT(vol_flux_x, X_FACE);
    HALO_UPDATE_RESIDENT(mass_flux_x, X_FACE);

    HALO_UPDATE_RESIDENT(vol_flux_y, Y_FACE);
    HALO_UPDATE_RESIDENT(mass_flux_y, Y_FACE);

    HALO_UPDATE_RESIDENT(vol_flux_z, Z_FACE);
    HALO_UPDATE_RESIDENT(mass_flux_z, Z_FACE);
}
