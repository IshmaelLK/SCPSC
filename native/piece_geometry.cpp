// Piece-preserving clipping, payload distance brackets and coverage witnesses.
// The old single-plane clipping implementation is included unchanged so that
// the exact same rectangle intersections are used. No external geometry library.
#include "source_clip.cpp"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace piece_geometry {
using Vec = std::array<long double,3>;
inline Vec add(Vec a,const Vec& b) { for(int k=0;k<3;++k)a[k]+=b[k];return a; }
inline Vec sub(Vec a,const Vec& b) { for(int k=0;k<3;++k)a[k]-=b[k];return a; }
inline Vec mul(Vec a,long double t) { for(auto& x:a)x*=t;return a; }
inline long double dot(const Vec& a,const Vec& b) {return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
inline long double norm(const Vec& a) {return std::sqrt(std::max(0.L,dot(a,a)));}
inline Vec read(const double* p) {return {p[0],p[1],p[2]};}

// Solve an at-most 3x3 positive definite Gram system. Rank-degenerate faces
// are skipped; their lower-dimensional subfaces are enumerated separately.
inline bool solve(long double A[3][3],long double b[3],int n,long double x[3]) {
    long double scale=0.;for(int i=0;i<n;++i)for(int j=0;j<n;++j)scale=std::max(scale,std::abs(A[i][j]));
    if(!(scale>0.))return false;
    for(int k=0;k<n;++k) {
        int p=k;for(int i=k+1;i<n;++i)if(std::abs(A[i][k])>std::abs(A[p][k]))p=i;
        if(std::abs(A[p][k])<=128.L*std::numeric_limits<long double>::epsilon()*scale)return false;
        for(int j=0;j<n;++j)std::swap(A[k][j],A[p][j]);std::swap(b[k],b[p]);
        for(int i=k+1;i<n;++i) {
            const long double f=A[i][k]/A[k][k];
            for(int j=k;j<n;++j)A[i][j]-=f*A[k][j];b[i]-=f*b[k];
        }
    }
    for(int i=n-1;i>=0;--i) {long double v=b[i];for(int j=i+1;j<n;++j)v-=A[i][j]*x[j];x[i]=v/A[i][i];}
    return true;
}

// Projection onto a small simplex using explicit feasible convex witnesses.
// Feasibility does not rely on a solver flag or on a rounded affine projection.
// At most four positive-weight vertices survive a projection; the next
// support point can temporarily make a fifth. Keep these tiny sets on the stack.
using Simplex = std::array<Vec,5>;
inline Vec closest(Simplex& simplex, int& n) {
    long double best=std::numeric_limits<long double>::infinity();Vec point{};
    std::array<int,4> active;
    int active_size=0;
    for(unsigned mask=1;mask<(1u<<n);++mask) {
        int ids[4],m=0;for(int j=0;j<n;++j)if(mask&(1u<<j)){if(m==4){m=5;break;}ids[m++]=j;}
        if(m>4)continue;
        long double w[4]={1.,0.,0.,0.};
        if(m==2) {
            // The one-dimensional Gram system has no elimination steps.
            const Vec edge=sub(simplex[ids[1]],simplex[ids[0]]);
            const long double gram=dot(edge,edge);
            if(!(gram>0.) || gram<=128.L*std::numeric_limits<long double>::epsilon()*gram)continue;
            w[1]=-dot(edge,simplex[ids[0]])/gram;
            w[0]-=w[1];
        } else if(m>2) {
            Vec e[3];for(int j=1;j<m;++j)e[j-1]=sub(simplex[ids[j]],simplex[ids[0]]);
            long double G[3][3]{},rhs[3]{},x[3]{};
            for(int i=0;i<m-1;++i){rhs[i]=-dot(e[i],simplex[ids[0]]);for(int j=0;j<m-1;++j)G[i][j]=dot(e[i],e[j]);}
            if(!solve(G,rhs,m-1,x))continue;
            for(int j=1;j<m;++j){w[j]=x[j-1];w[0]-=w[j];}
        }
        bool ok=true;long double sum=0.;
        for(int j=0;j<m;++j){if(w[j]<-1e-13L)ok=false;w[j]=std::max(0.L,w[j]);sum+=w[j];}
        if(!ok || !(sum>0.))continue;
        Vec p{};for(int j=0;j<m;++j)p=add(p,mul(simplex[ids[j]],w[j]/sum));
        const long double d=dot(p,p);
        if(d<best) {
            best=d;point=p;active_size=0;
            for(int j=0;j<m;++j)if(w[j]>0.)active[active_size++]=ids[j];
        }
    }
    if(!std::isfinite(best))throw std::runtime_error("invalid simplex");
    // Indices increase, so compacting left never overwrites an unread point.
    for(int j=0;j<active_size;++j)simplex[j]=simplex[active[j]];
    n=active_size;
    return point;
}

struct Bracket { Vec n{0.,0.,-1.}; long double lower=0.,upper=0.; int iterations=0,status=0; };

// dist(segment, polygon + downward ray). The finite floor is below both the
// segment and ALL top vertices. No Euclidean closest pair needs a lower point:
// raising such an obstacle point decreases its distance to every segment point.
// Every upper bound is an explicit feasible Minkowski-difference witness;
// every lower bound uses a valid downward unit normal against the unbounded set.
inline Bracket distance(const double* vertices,int count,const double* seeds,double tolerance,int max_iter) {
    const Vec s0=read(seeds),s1=read(seeds+3);long double floor=std::min(s0[2],s1[2]);
    long double scale=1.;for(int k=0;k<6;++k)scale=std::max(scale,std::abs((long double)seeds[k]));
    for(int i=0;i<count;++i){floor=std::min(floor,(long double)vertices[3*i+2]);for(int k=0;k<3;++k)scale=std::max(scale,std::abs((long double)vertices[3*i+k]));}
    floor-=1.;
    const long double guard=256.L*std::numeric_limits<double>::epsilon()*scale;
    auto support=[&](const Vec& d) {
        Vec a{};long double best=-std::numeric_limits<long double>::infinity();
        for(int i=0;i<count;++i) {
            Vec v=read(vertices+3*i);if(d[2]<0.)v[2]=floor;
            const long double value=dot(d,v);if(value>best){best=value;a=v;}
        }
        return sub(a,dot(d,s0)<=dot(d,s1)?s0:s1);
    };
    auto lower=[&](Vec n) {
        if(n[2]>0.)n[2]=0.;long double len=norm(n);if(!(len>0.))n={0.,0.,-1.};else n=mul(n,1./len);
        long double b=std::numeric_limits<long double>::infinity();
        for(int i=0;i<count;++i)b=std::min(b,dot(n,read(vertices+3*i)));
        return std::make_pair(std::max(0.L,b-std::max(dot(n,s0),dot(n,s1))-guard),n);
    };
    Simplex simplex;int n=1;
    Vec p=simplex[0]=support(Vec{0.,0.,1.});Bracket out;
    out.upper=norm(p)+guard;
    auto h=lower(Vec{0.,0.,-1.});out.lower=h.first;out.n=h.second;
    for(int it=0;it<max_iter;++it) {
        out.iterations=it+1;const long double length=norm(p);
        out.upper=std::min(out.upper,length+guard);
        if(length>guard) {auto q=lower(p);if(q.first>out.lower){out.lower=q.first;out.n=q.second;}}
        if(out.upper-out.lower<=tolerance){out.status=1;return out;}
        if(length<=guard){out.status=1;return out;}
        const Vec next=support(mul(p,-1.));
        bool repeated=false;for(int j=0;j<n;++j)if(norm(sub(next,simplex[j]))<=guard*0.01L){repeated=true;break;}
        if(repeated){out.status=2;return out;}
        simplex[n++]=next;p=closest(simplex,n);
    }
    out.status=3;return out; // A valid, possibly loose bracket; never false free.
}
} // namespace piece_geometry

namespace {
// Both source entry points use the original per-triangle clipping operations,
// including the inside shortcut and the order of the four rectangle planes.
int append_source_piece(Polygon& p, const double* lo, const double* hi,
        std::int64_t source_id, double* out, std::int64_t cap,
        std::int64_t* offsets, std::int64_t* source_ids,
        std::int64_t& used, std::int64_t& pieces) {
    Polygon q{};
    bool inside=true;
    for(int j=0;j<3;++j)inside=inside && p[j][0]>=lo[0] && p[j][0]<=hi[0] && p[j][1]>=lo[1] && p[j][1]<=hi[1];
    int count=3;
    const std::array<std::array<double,3>,4> planes{{{1.,0.,hi[0]},{-1.,0.,-lo[0]},{0.,1.,hi[1]},{0.,-1.,-lo[1]}}};
    if(!inside)for(const auto& plane:planes) {if(!count)break;count=clip_polygon(p,count,q,plane[0],plane[1],plane[2]);if(count<0)return -4;p=q;}
    if(!count)return 0;
    if(used+count>cap)return -2;
    for(int j=0;j<count;++j)for(int k=0;k<3;++k)out[3*(used+j)+k]=p[j][k];
    used+=count;source_ids[pieces]=source_id;offsets[++pieces]=used;
    return 0;
}
} // namespace

// Offsets are length triangle_count+1; output capacity is in vertices.
// Preserve every nonempty fragment, including line/point boundary intersections.
SCPSC_EXPORT std::int64_t scpsc_clip_pieces(const double* tris,std::int64_t n,
        const double* lo,const double* hi,double* out,std::int64_t cap,
        std::int64_t* offsets,std::int64_t* source_ids) noexcept {
    if(n<0 || cap<0 || !lo || !hi || !offsets || !(lo[0]<hi[0]) || !(lo[1]<hi[1]) || (n && (!tris || !out || !source_ids)))return -1;
    try {
        std::int64_t used=0,pieces=0;offsets[0]=0;
        for(std::int64_t i=0;i<n;++i) {
            Polygon p{};for(int j=0;j<3;++j)for(int k=0;k<3;++k){const double x=tris[9*i+3*j+k];if(!std::isfinite(x))return -1;p[j][k]=x;}
            const int status=append_source_piece(p,lo,hi,i,out,cap,offsets,source_ids,used,pieces);
            if(status)return status;
        }
        return pieces;
    }catch(...){return -3;}
}

// Fuse scpsc_raster_window with scpsc_clip_pieces without a triangle array.
// Node spacing need not be uniform. As before, IDs enumerate all SW-SE-NE
// triangles in row-major order, followed by all SW-NE-NW triangles.
// -1: invalid argument, -2: nonfinite node in the selected raster cells,
// -3: exception, -4: clipping exceeds seven vertices, -5: output too small.
SCPSC_EXPORT std::int64_t scpsc_clip_raster_pieces(
        const double* x, const double* y, const double* height,
        std::int64_t columns, std::int64_t rows, std::int64_t i0,
        std::int64_t i1, std::int64_t j0, std::int64_t j1,
        const double* lo, const double* hi, double* out, std::int64_t cap,
        std::int64_t* offsets, std::int64_t* source_ids) noexcept {
    if(!x || !y || !height || !lo || !hi || !offsets || cap<0 ||
       i0<0 || j0<0 || i1<i0 || j1<j0 || i1>=columns || j1>=rows ||
       !(lo[0]<hi[0]) || !(lo[1]<hi[1]))return -1;
    const auto width=i1-i0;
    const auto cells=width*(j1-j0);
    if(cells && (!out || !source_ids))return -1;
    try {
        // Match the old raster query's local NoData rule before clipping: all
        // four corners of every selected cell must be finite, even when only
        // one triangle contributes a nonempty fragment at the window boundary.
        for(auto j=j0;j<j1;++j)for(auto i=i0;i<i1;++i) {
            const auto k=j*columns+i;
            if(!(std::isfinite(height[k]) && std::isfinite(height[k+1]) &&
                 std::isfinite(height[k+columns]) && std::isfinite(height[k+columns+1]) &&
                 std::isfinite(x[i]) && std::isfinite(x[i+1]) &&
                 std::isfinite(y[j]) && std::isfinite(y[j+1])))return -2;
        }
        std::int64_t used=0,pieces=0;offsets[0]=0;
        for(int family=0;family<2;++family) {
            for(auto j=j0;j<j1;++j)for(auto i=i0;i<i1;++i) {
                const auto k=j*columns+i;
                Polygon p{};
                p[0]={x[i],y[j],height[k]};
                if(family==0) {
                    p[1]={x[i+1],y[j],height[k+1]};
                    p[2]={x[i+1],y[j+1],height[k+columns+1]};
                } else {
                    p[1]={x[i+1],y[j+1],height[k+columns+1]};
                    p[2]={x[i],y[j+1],height[k+columns]};
                }
                const auto id=family*cells+(j-j0)*width+i-i0;
                const int status=append_source_piece(p,lo,hi,id,out,cap,offsets,source_ids,used,pieces);
                if(status)return status==-2 ? -5 : status;
            }
        }
        return pieces;
    }catch(...){return -3;}
}

// result per piece: nx,ny,nz,lower,upper,iterations,status.
SCPSC_EXPORT int scpsc_payload_brackets(const double* V,std::int64_t nv,const std::int64_t* offsets,
       std::int64_t pieces,const double* seeds,double tol,int max_iter,double* output) noexcept {
    if(!V || !offsets || !seeds || !output || pieces<1 || nv<1 || offsets[0]!=0 || offsets[pieces]!=nv || !(tol>0.) || max_iter<1)return -1;
    for(int k=0;k<6;++k)if(!std::isfinite(seeds[k]))return -1;
    for(std::int64_t k=0;k<3*nv;++k)if(!std::isfinite(V[k]))return -1;
    try{
        for(std::int64_t i=0;i<pieces;++i) {
            const auto count=offsets[i+1]-offsets[i];if(count<1 || count>7)return -1;
            const auto r=piece_geometry::distance(V+3*offsets[i],int(count),seeds,tol,max_iter);
            for(int k=0;k<3;++k)output[7*i+k]=double(r.n[k]);
            output[7*i+3]=double(r.lower);output[7*i+4]=double(r.upper);
            output[7*i+5]=r.iterations;output[7*i+6]=r.status;
        }return 0;
    }catch(...){return -2;}
}
