#include "ocl_common.hpp"
extern CloverChunk chunk;

extern "C" void generate_chunk_kernel_ocl_
(const int* number_of_states,

const double* state_density,
const double* state_energy,
const double* state_xvel,
const double* state_yvel,
const double* state_zvel,
const double* state_xmin,
const double* state_xmax,
const double* state_ymin,
const double* state_ymax,
const double* state_zmin,
const double* state_zmax,
const double* state_radius,
const int* state_geometry,

const int* g_rect,
const int* g_circ,
const int* g_point)
{
    chunk.generate_chunk_kernel(
        * number_of_states, state_density, state_energy, state_xvel,
        state_yvel,state_zvel, state_xmin, state_xmax, state_ymin, state_ymax,state_zmin, state_zmax,
        state_radius, state_geometry, * g_rect, * g_circ, *g_point);
}

void CloverChunk::generate_chunk_kernel
(const int number_of_states,
const double* state_density, const double* state_energy,
const double* state_xvel, const double* state_yvel,const double* state_zvel,
const double* state_xmin, const double* state_xmax,
const double* state_ymin, const double* state_ymax,
const double* state_zmin, const double* state_zmax,
const double* state_radius, const int* state_geometry,
const int g_rect, const int g_circ, const int g_point)
{
#define TEMP_ALLOC(arr) \
cl::Buffer tmp_state_##arr; \
try \
{ \
tmp_state_##arr = cl::Buffer(context, \
CL_MEM_READ_ONLY, \
number_of_states*sizeof(*state_##arr)); \
queue.enqueueWriteBuffer(tmp_state_##arr, \
CL_TRUE, \
0, \
number_of_states*sizeof(*state_##arr), \
state_##arr); \
} \
catch (cl::Error e) \
{ \
DIE("Error in creating %s buffer %d\n", #arr, e.err()); \
}

    TEMP_ALLOC(density);
    TEMP_ALLOC(energy);
    TEMP_ALLOC(xvel);
    TEMP_ALLOC(yvel);
    TEMP_ALLOC(zvel);
    TEMP_ALLOC(xmin);
    TEMP_ALLOC(xmax);
    TEMP_ALLOC(ymin);
    TEMP_ALLOC(ymax);
    TEMP_ALLOC(zmin);
    TEMP_ALLOC(zmax);
    TEMP_ALLOC(radius);
    TEMP_ALLOC(geometry);

#undef TEMP_ALLOC

    generate_chunk_init_device.setArg(4, tmp_state_density);
    generate_chunk_init_device.setArg(5, tmp_state_energy);
    generate_chunk_init_device.setArg(6, tmp_state_xvel);
    generate_chunk_init_device.setArg(7, tmp_state_yvel);
    generate_chunk_init_device.setArg(8, tmp_state_zvel);

    ENQUEUE(generate_chunk_init_device);
    //ENQUEUE_OFFSET(generate_chunk_init_device);

    generate_chunk_device.setArg(9, tmp_state_density);
    generate_chunk_device.setArg(10, tmp_state_energy);
    generate_chunk_device.setArg(11, tmp_state_xvel);
    generate_chunk_device.setArg(12, tmp_state_yvel);
    generate_chunk_device.setArg(13, tmp_state_zvel);
    generate_chunk_device.setArg(14, tmp_state_xmin);
    generate_chunk_device.setArg(15, tmp_state_xmax);
    generate_chunk_device.setArg(16, tmp_state_ymin);
    generate_chunk_device.setArg(17, tmp_state_ymax);
    generate_chunk_device.setArg(18, tmp_state_zmin);
    generate_chunk_device.setArg(19, tmp_state_zmax);
    generate_chunk_device.setArg(20, tmp_state_radius);
    generate_chunk_device.setArg(21, tmp_state_geometry);

    generate_chunk_device.setArg(22, g_rect);
    generate_chunk_device.setArg(23, g_circ);
    generate_chunk_device.setArg(24, g_point);

    for (int state = 1; state < number_of_states; state++)
    {
        generate_chunk_device.setArg(25, state);

        ENQUEUE(generate_chunk_device);
        //ENQUEUE_OFFSET(generate_chunk_device);
    }
}
