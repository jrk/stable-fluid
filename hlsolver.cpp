#include <Halide.h>
#include <string.h>

using namespace Halide;

Var x("x"), y("y");

static int N;
static float dt, diff, visc;
static ImageParam _dens, _dens0, _u, _v, _u0, _v0;
static Func _dens_step, _vel_step;

Func add_source_func( int N, Func in, Func s, float dt )
{
    Func f;
    Var x("x"), y("y");

    f(x,y) = in(x,y) + dt*s(x,y);

    return f;
}

Func set_bnd_func ( int N, int b, Func in )
{
    Func f;

    Expr clampX = clamp(x, 1, N);
    Expr clampY = clamp(y, 1, N);
    Expr interior = in(clampX, clampY);

    b = 0; // NOTE: currently forcing everything not to impose boundaries - simplifies codegen

    if (b == 1) {
        f(x,y) = select(x < 1 || x > N,
            #if 1
                        -in(clampX, y),
                        in(x,y));
            #else
                        -interior,
                        interior);
            #endif
    } else if (b == 2) {
        f(x,y) = select(y < 1 || y > N,
            #if 1
                        -in(x, clampY),
                        in(x,y));
            #else
                        -interior,
                        interior);
            #endif
    } else {
        // f(x,y) = interior;
        f = in;
    }

    return f;
}

Func lin_solve_func ( int N, int b, Func in, Func x0, Expr a, Expr c, int num_steps=10 )
{
    Expr cx = x;
    Expr cy = y;
    cx = clamp(x, 1, N);
    cy = clamp(y, 1, N);
    #if 1
    Func* step = new Func[num_steps+1]; // TODO: leaks - don't care for now
    step[0] = in;
    Var xi("xi"), yi("yi");

    for ( int k=1 ; k<=num_steps ; k++ ) {
        Func f;
        f(x,y) = (x0(cx,cy) + a*(step[k-1](cx-1,cy)
                                +step[k-1](cx+1,cy)
                                +step[k-1](cx,cy-1)
                                +step[k-1](cx,cy+1)))/c;
        step[k] = set_bnd_func ( N, b, f );
        if ((k-(num_steps))%2 == 0) {
            step[k].tile(x, y, xi, yi, 16, 16);
            step[k].compute_root().store_root().parallel(y);//.vectorize(xi, 8);

            for (int i = k-1; (i-(num_steps))%2 != 0 && i > 0; i--) {
                step[i].compute_at(step[k], x).store_at(step[k], x);
            }
        }
    }

    return step[num_steps];
    #else // reduction
    Func solve;
    Var k("k");
    RDom rk(1, num_steps);
    solve(x,y,k) = in(x,y);
    #if 0
    solve(x,y,rk) = (x0(cx,cy) + a*(solve(cx-1,cy,rk-1)
                                   +solve(cx+1,cy,rk-1)
                                   +solve(cx,cy-1,rk-1)
                                   +solve(cx,cy+1,rk-1)))/c;
    #else
    solve(x,y,rk) = in(x,y);
    #endif
    Func solved;
    solved(x,y) = solve(x,y,num_steps-1);
    return solved;
    #endif
}

Func diffuse_func( int b, Func dens, Func dens0, Expr diff )
{
    Expr a = dt*diff*N*N;
    return lin_solve_func(N, b, dens, dens0, a, 1.0f+4.0f*a);
}

Func advect_func ( int b, Func d0, Func u, Func v )
{
    Func advected;

    Expr dt0 = dt*N;

    Expr xx = clamp(x-dt0*u(x,y), 0.5f, N+0.5f);
    Expr yy = clamp(y-dt0*v(x,y), 0.5f, N+0.5f);
    Expr i0=cast<int>(xx);
    Expr j0=cast<int>(yy);
    Expr i1=i0+1;
    Expr j1=j0+1;
    Expr s1 = xx-i0;
    Expr t1 = yy-j0;
    Expr s0 = 1-s1;
    Expr t0 = 1-t1;
    advected(x,y) = s0*(t0*d0(i0,j0)+t1*d0(i0,j1))+
                    s1*(t0*d0(i1,j0)+t1*d0(i1,j1));

    return set_bnd_func(N, b, advected);
}

Func project_func ( Func u, Func v )
{
    Func div, zero;
    div(x,y) = -0.5f*(u(x+1,y)-u(x-1,y)+v(x,y+1)-v(x,y-1))/N;
    zero(x,y) = 0.f;

    Func p = lin_solve_func(N, 0, zero, set_bnd_func(N, 0, div), 1.0f, 4.0f);
    p.compute_root().store_root();

    Func uu, vv;

    uu(x,y) = u(x,y) - 0.5f*N*(p(x+1,y)-p(x-1,y));
    vv(x,y) = v(x,y) - 0.5f*N*(p(x,y+1)-p(x,y-1));

    Func res;
    Var uv("uv");
    res(x,y,uv) = select(uv == 0,
                    set_bnd_func(N, 1, uu)(x,y),
                    select(uv == 1,
                        set_bnd_func(N, 2, vv)(x,y),
                        select(uv == 2,
                            p(x,y),
                            div(x,y))));
    res.unroll(uv, 4);
    return res;
}

Func dens_step_func ( Func dens, Func dens0, Func u, Func v )
{
    Func src_added = add_source_func(N, dens, dens0, dt);
    Func diffused = diffuse_func(0, dens0, src_added, diff);
    return advect_func(0, diffused, u, v);
}

Func vel_step_func( Func u, Func v, Func u0, Func v0 )
{
    Func uu = add_source_func(N, u, u0, dt);
    Func vv = add_source_func(N, v, v0, dt);
    uu.compute_root().store_root();
    vv.compute_root().store_root();

    Func diffU = diffuse_func(1, u0, uu, visc);
    Func diffV = diffuse_func(2, v0, vv, visc);
    diffU.compute_root().store_root();
    diffV.compute_root().store_root();

    Func projected = project_func(diffU, diffV);
    projected.compute_root().store_root();

    Func au, au0, av, av0;
    au0(x,y) = projected(x,y,0);
    av0(x,y) = projected(x,y,1);
    au(x,y)  = projected(x,y,2);
    av(x,y)  = projected(x,y,3);

    Func adU = advect_func(1, au0, au0, av0);
    Func adV = advect_func(2, av0, au0, av0);
    adU.compute_root().store_root();
    adV.compute_root().store_root();

    return project_func(adU, adV);
}

void vel_step ( float * u, float * v, float * u0, float * v0 )
{
    _u.set(Buffer(Float(32), N+2, N+2, 0, 0, (uint8_t*)u));
    _v.set(Buffer(Float(32), N+2, N+2, 0, 0, (uint8_t*)v));
    _u0.set(Buffer(Float(32), N+2, N+2, 0, 0, (uint8_t*)u0));
    _v0.set(Buffer(Float(32), N+2, N+2, 0, 0, (uint8_t*)v0));

    Image<float> res = _vel_step.realize(N+2, N+2, 4);
    size_t sz = sizeof(float)*res.width()*res.height();
    size_t stride = res.stride(2);
    memcpy(  u, res.data()+0*stride, sz);
    memcpy(  v, res.data()+1*stride, sz);
    memcpy( u0, res.data()+2*stride, sz);
    memcpy( v0, res.data()+3*stride, sz);
}

void dens_step ( float * dens, float * dens0, float * u, float * v )
{
    _dens.set(Buffer(Float(32), N+2, N+2, 0, 0, (uint8_t*)dens));
    _dens0.set(Buffer(Float(32), N+2, N+2, 0, 0, (uint8_t*)dens0));
    _u.set(Buffer(Float(32), N+2, N+2, 0, 0, (uint8_t*)u));
    _v.set(Buffer(Float(32), N+2, N+2, 0, 0, (uint8_t*)v));

    Image<float> res = _dens_step.realize(N+2,N+2);
    memcpy(dens, res.data(), sizeof(float)*res.width()*res.height());
}

void step( float* u, float* v, float* u_prev, float* v_prev,
           float* dens, float* dens_prev )
{
    vel_step ( u, v, u_prev, v_prev );
    dens_step ( dens, dens_prev, u, v );
}

// TODO: switch to static compile?
void hlinit( int N_, float visc_, float diff_, float dt_ )
{
    N = N_;
    visc = visc_;
    diff = diff_;
    dt = dt_;

    _dens = ImageParam(Float(32), 2, "densbuf");
    _dens0= ImageParam(Float(32), 2, "Dens0buf");
    _u = ImageParam(Float(32), 2, "uBuf");
    _v = ImageParam(Float(32), 2, "vBuf");
    _u0 = ImageParam(Float(32), 2, "u0Buf");
    _v0 = ImageParam(Float(32), 2, "v0Buf");

    Func dens("dens"), dens0("dens0"), u("U"), v("V"), u0("U0"), v0("V0");
    Expr cx = clamp(x, 0, N+1);
    Expr cy = clamp(y, 0, N+1);
    dens(x,y) = _dens(cx, cy);
    dens0(x,y)=_dens0(cx, cy);
    u(x,y)  = _u(cx, cy);
    v(x,y)  = _v(cx, cy);
    u0(x,y) =_u0(cx, cy);
    v0(x,y) =_v0(cx, cy);

    fprintf(stderr, "Compile dens_step...");
    _dens_step = dens_step_func(dens, dens0, u, v);
	#ifndef HL_STATIC_COMPILE
    _dens_step.compile_jit();
	#endif
    fprintf(stderr, "done\n");

    fprintf(stderr, "Compile vel_step...");
    _vel_step = vel_step_func(u, v, u0, v0);
	#ifndef HL_STATIC_COMPILE
    _vel_step.compile_jit();
	#endif
    fprintf(stderr, "done\n");
}

void hlstep( int N, float* u, float* v, float* u_prev, float* v_prev,
             float* dens, float* dens_prev )
{

}

#ifdef HL_STATIC_COMPILE
int main ( int argc, char ** argv )
{
	int N;
	float dt, diff, visc, force, source;
	
	if ( argc != 1 && argc != 7 ) {
		fprintf ( stderr, "usage : %s N dt diff visc force source\n", argv[0] );
		fprintf ( stderr, "where:\n" );\
		fprintf ( stderr, "\t N      : grid resolution\n" );
		fprintf ( stderr, "\t dt     : time step\n" );
		fprintf ( stderr, "\t diff   : diffusion rate of the density\n" );
		fprintf ( stderr, "\t visc   : viscosity of the fluid\n" );
		fprintf ( stderr, "\t force  : scales the mouse movement that generate a force\n" );
		fprintf ( stderr, "\t source : amount of density that will be deposited\n" );
		exit ( 1 );
	}

	if ( argc == 1 ) {
		N = 64;
		dt = 0.1f;
		diff = 0.0f;
		visc = 0.0f;
		force = 5.0f;
		source = 100.0f;
		fprintf ( stderr, "Using defaults : N=%d dt=%g diff=%g visc=%g force = %g source=%g\n",
			N, dt, diff, visc, force, source );
	} else {
		N = atoi(argv[1]);
		dt = atof(argv[2]);
		diff = atof(argv[3]);
		visc = atof(argv[4]);
		force = atof(argv[5]);
		source = atof(argv[6]);
	}
	
	hlinit(N, visc, diff, dt);
	_dens_step.compile_to_file("solver_dens_step", _dens, _dens0, _u, _v);
	_vel_step.compile_to_file("solver_vel_step", _u, _v, _u0, _v0);
}
#endif