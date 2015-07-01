#include <string.h>
#include <stdlib.h>
#include "solver_dens_step.h"
#include "solver_vel_step.h"
extern "C" void solver_dens_step(const buffer_t *densbuf, const buffer_t *Dens0buf, const buffer_t *uBuf, const buffer_t *vBuf, const buffer_t *f34);

static int N;
buffer_t out;

static buffer_t floatbuf(int x_size, int y_size, int z_size, int w_size, uint8_t* data)
{
	buffer_t buf;
	
    buf.elem_size = sizeof(float) / 8;
    size_t size = 1;
    if (x_size) size *= x_size;
    if (y_size) size *= y_size;
    if (z_size) size *= z_size;
    if (w_size) size *= w_size;
    else buf.host = data;
    buf.host_dirty = true;
    buf.dev_dirty = false;
    buf.extent[0] = x_size;
    buf.extent[1] = y_size;
    buf.extent[2] = z_size;
    buf.extent[3] = w_size;
    buf.stride[0] = 1;
    buf.stride[1] = x_size;
    buf.stride[2] = x_size*y_size;
    buf.stride[3] = x_size*y_size*z_size;
    buf.min[0] = 0;
    buf.min[1] = 0;
    buf.min[2] = 0;
    buf.min[3] = 0;

	return buf;
}

void step( float* _u, float* _v, float* _u0, float* _v0,
           float* _dens, float* _dens0 )
{
    buffer_t dens = floatbuf(N+2, N+2, 0, 0, (uint8_t*)_dens);
    buffer_t dens0 = floatbuf(N+2, N+2, 0, 0, (uint8_t*)_dens0);
    buffer_t u = floatbuf(N+2, N+2, 0, 0, (uint8_t*)_u);
    buffer_t v = floatbuf(N+2, N+2, 0, 0, (uint8_t*)_v);
    buffer_t u0 = floatbuf(N+2, N+2, 0, 0, (uint8_t*)_u0);
    buffer_t v0 = floatbuf(N+2, N+2, 0, 0, (uint8_t*)_v0);

    out.extent[2] = 4;
    solver_vel_step ( &u, &v, &u0, &v0, &out );

    size_t sz = sizeof(float)*(N+2)*(N+2);
    size_t stride = out.stride[2];
    memcpy(  u.host, out.host+0*stride, sz);
    memcpy(  v.host, out.host+1*stride, sz);
    memcpy( u0.host, out.host+2*stride, sz);
    memcpy( v0.host, out.host+3*stride, sz);

    out.extent[2] = 0;
    solver_dens_step ( &dens, &dens0, &u, &v, &out );
    
    memcpy(dens.host, out.host, sizeof(float)*(N+2)*(N+2));
}

void hlinit( int N_, float visc_, float diff_, float dt_ )
{
	N = N_;

    out = floatbuf(N+2, N+2, 4, 0, (uint8_t*)malloc((N+2)*(N+2)*4*sizeof(float)));

	return;
}
