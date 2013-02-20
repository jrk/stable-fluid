#include <Halide.h>
#include <string.h>

using namespace Halide;

#define IX(i,j) ((i)+(N+2)*(j))
#define SWAP(x0,x) {float * tmp=x0;x0=x;x=tmp;}
#define FOR_EACH_CELL for ( i=1 ; i<=N ; i++ ) { for ( j=1 ; j<=N ; j++ ) {
#define END_FOR }}

Func add_source_func( int N, Func in, Func s, float dt )
{
    Func f("add_source");
    Var x("x"), y("y");

    f(x,y) = in(x,y) + dt*s(x,y);

    return f;
}

static int N;
static float dt, diff, visc;
static Param<float> _diff, _visc, _a, _c;
static ImageParam _x, _s, _x0, _u, _v;
static Func _add_source;
static Func _set_bnd[3];
static Func _lin_solve[3];
static Func _project;
static Func _dens_step;

void add_source ( int N, float * x, float * s, float dt  )
{
    #if 1
    _x.set(Buffer(Float(32), N+2, N+2, 0, 0, (uint8_t*)x));
    _s.set(Buffer(Float(32), N+2, N+2, 0, 0, (uint8_t*)s));

    Image<float> res = _add_source.realize(N+2,N+2);
    memcpy(x, res.data(), sizeof(float)*res.width()*res.height());

    #else
    int i, size=(N+2)*(N+2);
    for ( i=0 ; i<size ; i++ ) x[i] += dt*s[i];
    #endif
}

Func set_bnd_func ( int N, int b, Func in )
{
    Func f;
    Var x("x"), y("y");

    Expr clampX = clamp(x, 1, N);
    Expr clampY = clamp(y, 1, N);
    Expr interior = in(clampX, clampY);

    if (b == 1) {
        f(x,y) = select(x < 1 || x > N,
                        -interior,
                        interior);
    } else if (b == 2) {
        f(x,y) = select(y < 1 || y > N,
                        -interior,
                        interior);
    } else {
        f(x,y) = interior;
    }

    return f;
}

void set_bnd ( int N, int b, float * x )
{
    #if 1
    _x.set(Buffer(Float(32), N+2, N+2, 0, 0, (uint8_t*)x));

    Image<float> res = _set_bnd[b].realize(N+2,N+2);
    memcpy(x, res.data(), sizeof(float)*res.width()*res.height());
    #else
    int i;

    for ( i=1 ; i<=N ; i++ ) {
        x[IX(0  ,i)] = b==1 ? -x[IX(1,i)] : x[IX(1,i)];
        x[IX(N+1,i)] = b==1 ? -x[IX(N,i)] : x[IX(N,i)];
        x[IX(i,0  )] = b==2 ? -x[IX(i,1)] : x[IX(i,1)];
        x[IX(i,N+1)] = b==2 ? -x[IX(i,N)] : x[IX(i,N)];
    }
    x[IX(0  ,0  )] = 0.5f*(x[IX(1,0  )]+x[IX(0  ,1)]);
    x[IX(0  ,N+1)] = 0.5f*(x[IX(1,N+1)]+x[IX(0  ,N)]);
    x[IX(N+1,0  )] = 0.5f*(x[IX(N,0  )]+x[IX(N+1,1)]);
    x[IX(N+1,N+1)] = 0.5f*(x[IX(N,N+1)]+x[IX(N+1,N)]);
    #endif
}

Func lin_solve_func ( int N, int b, Func in, Func x0, Expr a, Expr c, int num_steps=20 )
{
    Var x("x"),y("y");
    Expr cx = clamp(x, 1, N);
    Expr cy = clamp(y, 1, N);
    Func prevStep = in;

    for ( int k=0 ; k<num_steps ; k++ ) {
        Func f;
        f(x,y) = (x0(cx,cy) + a*(prevStep(cx-1,cy)
                                +prevStep(cx+1,cy)
                                +prevStep(cx,cy-1)
                                +prevStep(cx,cy+1)))/c;
        prevStep = set_bnd_func ( N, b, f );
        if ((k-(num_steps-1))%1 == 0) {
            prevStep.compute_root().store_root().parallel(y).vectorize(x, 8);
        }
    }

    return prevStep;
}

void lin_solve ( int b, float * x, float * x0, float a, float c )
{
    #if 1
    _a.set(a);
    _c.set(c);
    _x.set(Buffer(Float(32), N+2, N+2, 0, 0, (uint8_t*)x));
    _x0.set(Buffer(Float(32), N+2, N+2, 0, 0, (uint8_t*)x0));

    Image<float> res = _lin_solve[b].realize(N+2,N+2);
    memcpy(x, res.data(), sizeof(float)*res.width()*res.height());
    #else
    int i, j, k;

    for ( k=0 ; k<20 ; k++ ) {
        FOR_EACH_CELL
            x[IX(i,j)] = (x0[IX(i,j)] + a*(x[IX(i-1,j)]+x[IX(i+1,j)]+x[IX(i,j-1)]+x[IX(i,j+1)]))/c;
        END_FOR
        set_bnd ( N, b, x );
    }
    #endif
}

Func diffuse_func( int b, Func x, Func x0, Expr diff )
{
    Expr a = dt*diff*N*N;
    return lin_solve_func(N, b, x, x0, a, 1.0f+4.0f*a);
}

static Func _diffuse[3];
void diffuse ( int b, float * x, float * x0, float diff )
{
    #if 1
    _diff.set(diff);
    _x.set(Buffer(Float(32), N+2, N+2, 0, 0, (uint8_t*)x));
    _x0.set(Buffer(Float(32), N+2, N+2, 0, 0, (uint8_t*)x0));

    Image<float> res = _diffuse[b].realize(N+2,N+2);
    memcpy(x, res.data(), sizeof(float)*res.width()*res.height());
    #else
    float a=dt*diff*N*N;
    // Param<Float(32)> diff;
    // Func solver = lin_solve_func(N, b, inf, x0f, a, 1+4*a);
    lin_solve ( b, x, x0, a, 1+4*a );
    #endif
}

Func advect_func ( int b, Func d0, Func u, Func v )
{
    Func advected;
    Var x("x"), y("y");

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

static Func _advect[3];
void advect ( int b, float * d, float * d0, float * u, float * v )
{
    #if 1
    _x0.set(Buffer(Float(32), N+2, N+2, 0, 0, (uint8_t*)d0));
    _u.set(Buffer(Float(32), N+2, N+2, 0, 0, (uint8_t*)u));
    _v.set(Buffer(Float(32), N+2, N+2, 0, 0, (uint8_t*)v));

    Buffer res(Float(32), N+2, N+2, 0, 0, (uint8_t*)d);
    _advect[b].realize(res);
    // Image<float> res = _advect[b].realize(N+2,N+2);
    // memcpy(d, res.data(), sizeof(float)*res.width()*res.height());
    #else
    int i, j, i0, j0, i1, j1;
    float x, y, s0, t0, s1, t1, dt0;

    dt0 = dt*N;
    FOR_EACH_CELL
        x = i-dt0*u[IX(i,j)]; y = j-dt0*v[IX(i,j)];
        if (x<0.5f) x=0.5f; if (x>N+0.5f) x=N+0.5f; i0=(int)x; i1=i0+1;
        if (y<0.5f) y=0.5f; if (y>N+0.5f) y=N+0.5f; j0=(int)y; j1=j0+1;
        s1 = x-i0; s0 = 1-s1; t1 = y-j0; t0 = 1-t1;
        d[IX(i,j)] = s0*(t0*d0[IX(i0,j0)]+t1*d0[IX(i0,j1)])+
                     s1*(t0*d0[IX(i1,j0)]+t1*d0[IX(i1,j1)]);
    END_FOR
    set_bnd ( N, b, d );
    #endif
}

Func project_func ( Func u, Func v )
{
    Var x("x"), y("y");
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
    // res.unroll(uv, 4);
    return res;
}

void project ( float * u, float * v, float * p, float * div )
{
    #if 1
    _u.set(Buffer(Float(32), N+2, N+2, 0, 0, (uint8_t*)u));
    _v.set(Buffer(Float(32), N+2, N+2, 0, 0, (uint8_t*)v));

    Image<float> res = _project.realize(N+2, N+2, 4);
    size_t sz = sizeof(float)*res.width()*res.height();
    size_t stride = res.stride(2);
    memcpy(  u, res.data()+0*stride, sz);
    memcpy(  v, res.data()+1*stride, sz);
    memcpy(  p, res.data()+2*stride, sz);
    memcpy(div, res.data()+3*stride, sz);
    #else
    int i, j;

    FOR_EACH_CELL
        div[IX(i,j)] = -0.5f*(u[IX(i+1,j)]-u[IX(i-1,j)]+v[IX(i,j+1)]-v[IX(i,j-1)])/N;
        p[IX(i,j)] = 0;
    END_FOR 
    set_bnd ( N, 0, div ); set_bnd ( N, 0, p );

    lin_solve ( 0, p, div, 1, 4 );

    FOR_EACH_CELL
        u[IX(i,j)] -= 0.5f*N*(p[IX(i+1,j)]-p[IX(i-1,j)]);
        v[IX(i,j)] -= 0.5f*N*(p[IX(i,j+1)]-p[IX(i,j-1)]);
    END_FOR
    set_bnd ( N, 1, u ); set_bnd ( N, 2, v );
    #endif
}

Func dens_step_func ( Func x, Func x0, Func u, Func v )
{
    Func src_added = add_source_func(N, x, x0, dt);
    Func diffused = diffuse_func(0, x0, src_added, diff);
    return advect_func(0, diffused, u, v);
}

void dens_step ( float * x, float * x0, float * u, float * v )
{
    #if 1
    _x.set(Buffer(Float(32), N+2, N+2, 0, 0, (uint8_t*)x));
    _x0.set(Buffer(Float(32), N+2, N+2, 0, 0, (uint8_t*)x0));
    _u.set(Buffer(Float(32), N+2, N+2, 0, 0, (uint8_t*)u));
    _v.set(Buffer(Float(32), N+2, N+2, 0, 0, (uint8_t*)v));

    Image<float> res = _dens_step.realize(N+2,N+2);
    memcpy(x, res.data(), sizeof(float)*res.width()*res.height());
    #else
    add_source ( N, x, x0, dt );
    SWAP ( x0, x ); diffuse ( 0, x, x0, diff );
    SWAP ( x0, x ); advect ( 0, x, x0, u, v );
    #endif
}

void vel_step ( float * u, float * v, float * u0, float * v0 )
{
    add_source ( N, u, u0, dt ); add_source ( N, v, v0, dt );
    SWAP ( u0, u ); diffuse ( 1, u, u0, visc );
    SWAP ( v0, v ); diffuse ( 2, v, v0, visc );
    project ( u, v, u0, v0 );
    SWAP ( u0, u ); SWAP ( v0, v );
    advect ( 1, u, u0, u0, v0 ); advect ( 2, v, v0, u0, v0 );
    project ( u, v, u0, v0 );
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

    _x = ImageParam(Float(32), 2, "Xbuf");
    _x0= ImageParam(Float(32), 2, "X0buf");
    _s = ImageParam(Float(32), 2, "Sbuf");
    _u = ImageParam(Float(32), 2, "Ubuf");
    _v = ImageParam(Float(32), 2, "Vbuf");

    Var x("x"),y("y");
    Func in("X"), in0("X0"), s("S"), u("U"), v("V");
    in(x,y) = _x(clamp(x, 0, N+1), clamp(y, 0, N+1));
    in0(x,y)=_x0(clamp(x, 0, N+1), clamp(y, 0, N+1));
    s(x,y)  = _s(clamp(x, 0, N+1), clamp(y, 0, N+1));
    u(x,y)  = _u(clamp(x, 0, N+1), clamp(y, 0, N+1));
    v(x,y)  = _v(clamp(x, 0, N+1), clamp(y, 0, N+1));

    _add_source = add_source_func(N, in, s, dt);
    _add_source.compile_jit();

    Expr a = dt*N*N*_diff;
    for (int b = 0; b < 3; b++) {
        _set_bnd[b] = set_bnd_func(N, b, in);
        _set_bnd[b].compile_jit();

        _lin_solve[b] = lin_solve_func(N, b, in, in0, _a, _c);
        _lin_solve[b].compile_jit();

        _advect[b] = advect_func(b, in0, u, v);
        _advect[b].compile_jit();

        _diffuse[b] = diffuse_func(b, in, in0, _diff);
        _diffuse[b].compile_jit();
    }

    _project = project_func(u, v);
    _project.compile_jit();

    _dens_step = dens_step_func(in, in0, u, v);
}

void hlstep( int N, float* u, float* v, float* u_prev, float* v_prev,
             float* dens, float* dens_prev )
{

}
