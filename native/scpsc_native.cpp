// SCPSC segment engine. No per-vertex or nearest-selection Python dispatch.
// NumPy's native ufunc loops are numerical dependencies, not Python callbacks:
// use the SAME matmul/acos/cos kernels as the reference (including its BLAS).
// Do not replace them by scalar algebra, fast-math, or handwritten SIMD.
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

extern "C" int scpsc_payload_brackets(const double*, std::int64_t,
    const std::int64_t*, std::int64_t, const double*, double, int, double*) noexcept;

namespace {
using Index = std::int64_t;
using Vec = std::array<double, 3>;
using Clock = std::chrono::steady_clock;
struct PythonError {};
struct Ref {
    PyObject* p;
    explicit Ref(PyObject* value=nullptr) : p(value) {}
    ~Ref() { Py_XDECREF(p); }
    Ref(const Ref&) = delete;
    Ref& operator=(const Ref&) = delete;
    PyObject* release() { auto q=p; p=nullptr; return q; }
};
struct Loop { PyUFuncGenericFunction fn=nullptr; void* data=nullptr; };
Loop acos_loop, cos_loop, sin_loop, matmul_loop, minimum_loop, add_loop;
PyObject* solver_error = nullptr;
PyObject* plane_failure = nullptr;
std::vector<PyObject*> kernel_owners;

void check_python() { if (PyErr_Occurred()) throw PythonError{}; }
[[noreturn]] void fail(const char* message) {
    PyErr_SetString(solver_error, message); throw PythonError{};
}
void checkpoint(PyObject* callback) {
    if (callback == Py_None) return;
    Ref result(PyObject_CallFunction(callback, "s", "corridor"));
    if (!result.p) throw PythonError{};
}
double elapsed(Clock::time_point t) {
    return std::chrono::duration<double>(Clock::now()-t).count();
}
Loop get_loop(PyObject* numpy, const char* name, int nargs) {
    PyObject* obj = PyObject_GetAttrString(numpy, name);
    if (!obj) throw PythonError{};
    kernel_owners.push_back(obj);
    if (!PyObject_TypeCheck(obj, &PyUFunc_Type)) {
        PyErr_SetString(PyExc_RuntimeError, "SCPSC requires unmodified NumPy native ufuncs");
        throw PythonError{};
    }
    auto* u = reinterpret_cast<PyUFuncObject*>(obj);
    for (int i=0; i<u->ntypes; ++i) {
        bool match = u->nargs == nargs;
        for (int k=0; k<nargs && match; ++k) match = u->types[i*nargs+k] == NPY_DOUBLE;
        if (match) return {u->functions[i], u->data[i]};
    }
    PyErr_SetString(PyExc_RuntimeError, "SCPSC: missing NumPy float64 native kernel");
    throw PythonError{};
}
void unary(Loop kernel, const double* x, double* y, npy_intp count) {
    char* data[] = {reinterpret_cast<char*>(const_cast<double*>(x)), reinterpret_cast<char*>(y)};
    npy_intp shape[] = {count}, strides[] = {8,8};
    kernel.fn(data, shape, strides, kernel.data);
    check_python();
}
// The native gufunc takes outer_count,m,n,p followed by operand/core strides.
// This preserves matrix-vector vs matrix-matrix dispatch, which is NOT the
// same operation as sum(x*y,axis=1) at the bit level.
void matmul(const double* a, const double* b, double* out,
            npy_intp m, npy_intp n, npy_intp p,
            npy_intp a_m, npy_intp a_n, npy_intp b_n, npy_intp b_p) {
    char* data[] = {reinterpret_cast<char*>(const_cast<double*>(a)),
                    reinterpret_cast<char*>(const_cast<double*>(b)), reinterpret_cast<char*>(out)};
    npy_intp shape[] = {1,m,n,p};
    npy_intp strides[] = {0,0,0, a_m,a_n,b_n,b_p, p*8,8};
    matmul_loop.fn(data, shape, strides, matmul_loop.data);
    check_python();
}
double dot(const double* a, const double* b, npy_intp n=3) {
    double out;
    matmul(a,b,&out,1,n,1,n*8,8,8,8);
    return out;
}
// Finite values only. NumPy minimum/maximum select the SECOND equal value;
// Python min/max select the FIRST. Keep those two semantics separate.
double npmin(double a, double b) { return a < b ? a : b; }
double npmax(double a, double b) { return a > b ? a : b; }
double reduce_min(const double* x, npy_intp n) {
    double out = x[0];
    if (n>1) {
        char* data[] = {reinterpret_cast<char*>(&out),
                       reinterpret_cast<char*>(const_cast<double*>(x+1)), reinterpret_cast<char*>(&out)};
        npy_intp shape[] = {n-1}, strides[] = {0,8,0};
        minimum_loop.fn(data, shape, strides, minimum_loop.data);
        check_python();
    }
    return out;
}
double rownorm(const double* x) { return std::sqrt((x[0]*x[0]+x[1]*x[1])+x[2]*x[2]); }
Vec normal(const double* x) {
    Vec n{x[0],x[1],std::min(0.0,x[2])};
    const double length=std::sqrt(dot(n.data(),n.data()));
    if (!std::isfinite(length)) fail("PIECE_SOCP: invalid normal");
    if (length<=1e-12) fail("PIECE_SOCP: zero normal");
    for(auto& value:n) value/=length;
    return n;
}
// Eq. (9), support of the fixed envelope from Eq. (8) along each supplied normal.
std::vector<double> supports(const double* a, npy_intp count, double ell, double rL, double rQ, double theta) {
    std::vector<double> lengths(count), gamma(count), values(count);
    for(npy_intp i=0;i<count;++i) {
        lengths[i]=rownorm(a+3*i);
        if (!(lengths[i]>0.) || !std::isfinite(lengths[i])) {
            PyErr_SetString(PyExc_ValueError,"Envelope requires nonzero finite face normals");
            throw PythonError{};
        }
        gamma[i]=npmin(npmax(a[3*i+2]/lengths[i],-1.),1.);
    }
    unary(acos_loop,gamma.data(),values.data(),count);
    for(npy_intp i=0;i<count;++i) gamma[i]=npmax(0.,values[i]-theta);
    unary(cos_loop,gamma.data(),values.data(),count);
    for(npy_intp i=0;i<count;++i) values[i]=lengths[i]*npmax(rL,ell*values[i]+rQ);
    return values;
}
// Same count=1 kernels and arithmetic as supports(), without three heap arrays.
double support_one(const double* a,double ell,double rL,double rQ,double theta,
                   std::array<double,3>& scratch) {
    double& length=scratch[0];double& gamma=scratch[1];double& value=scratch[2];
    length=rownorm(a);
    if (!(length>0.) || !std::isfinite(length)) {
        PyErr_SetString(PyExc_ValueError,"Envelope requires nonzero finite face normals");
        throw PythonError{};
    }
    gamma=npmin(npmax(a[2]/length,-1.),1.);
    unary(acos_loop,&gamma,&value,1);
    gamma=npmax(0.,value-theta);
    unary(cos_loop,&gamma,&value,1);
    value=length*npmax(rL,ell*value+rQ);
    return value;
}
double maxabs(const double* a, std::size_t n) {
    double result=0.;for(std::size_t i=0;i<n;++i) result=std::max(result,std::abs(a[i]));return result;
}
double sum_guard_terms(double a, double b, double c) {
#if PY_VERSION_HEX >= 0x030C0000
    // CPython >=3.12 changed builtin sum(float) to compensated summation.
    // TASK's Python 3.10 takes the sequential branch below.
    double high=0., low=0.;
    for(double x : {a,b,c}) {
        double t=high+x;
        if(std::abs(high)>=std::abs(x)) low+=(high-t)+x;
        else low+=(x-t)+high;
        high=t;
    }
    return low!=0. && std::isfinite(low) ? high+low : high;
#else
    return ((0.+a)+b)+c;
#endif
}
double guard(const double* a, std::size_t na, const double* b, std::size_t nb,
             double ell, double rL, double rQ) {
    const double parameters[]={ell,rL,rQ};
    return (256.*std::numeric_limits<double>::epsilon())*
        (1.+sum_guard_terms(maxabs(a,na),maxabs(b,nb),maxabs(parameters,3)));
}


double reduce_sum(const double* values, npy_intp n, npy_intp stride=8) {
    double sum=0.;
    char* data[]={reinterpret_cast<char*>(&sum),
                  reinterpret_cast<char*>(const_cast<double*>(values)),reinterpret_cast<char*>(&sum)};
    npy_intp shape[]={n}, strides[]={0,stride,0};
    add_loop.fn(data,shape,strides,add_loop.data);check_python();return sum;
}
struct OracleBounds {Vec normal;double lower,upper,guard;};
OracleBounds oracle_bounds(const double* v, Index size,const double* seeds,
                           const double* x,const double* z,double ell,double rL,double rQ,double theta,
                           std::array<double,3>& support_scratch,const std::array<double,2>& theta_trig) {
    const auto n=normal(x);
    const double h=support_one(n.data(),ell,rL,rQ,theta,support_scratch);
    const double g=guard(v,3*size,seeds,6,ell,rL,rQ);
    double vp_projection[7],sp[2];
    matmul(v,n.data(),vp_projection,size,3,1,24,8,8,8);
    matmul(seeds,n.data(),sp,2,3,1,24,8,8,8);
    const double gap=((reduce_min(vp_projection,size)-npmax(sp[0],sp[1]))-h)/std::sqrt(dot(n.data(),n.data()));
    const double lower=std::max(0.,gap-g);
    std::array<double,7> weights;
    std::array<double,4> masses;
    for(int i=0;i<7;++i)weights[i]=npmax(0.,z[i]);
    for(int i=0;i<4;++i)masses[i]=npmax(0.,z[7+i]);
    const double ws=reduce_sum(weights.data(),7),ms=reduce_sum(masses.data(),4);
    if(ws<=0. || ms<=0.)fail("PIECE_SOCP_UNRESOLVED: missing feasible-witness weights");
    for(auto& a:weights)a/=ws;
    for(auto& a:masses)a/=ms;
    const double alpha=reduce_sum(masses.data(),2,16),beta=reduce_sum(masses.data()+1,2,16);
    const double mix[]={masses[0]+masses[1],masses[2]+masses[3]};
    Vec k;matmul(mix,seeds,k.data(),1,2,3,16,8,24,8);
    const double radius=alpha*rL+beta*rQ;
    for(int i=0;i<3;++i)k[i]+=radius*n[i];
    const double st=theta_trig[0],ct=theta_trig[1];
    Vec qcap{st,0.,ct};
    const double length=std::sqrt(dot(n.data(),n.data(),2));
    if(length>0.)for(int i=0;i<2;++i)qcap[i]=(st*n[i])/length;
    if(length<=1e-8 && z[18]>0.) {
        double direction[]={-z[19]/z[18],-z[20]/z[18]};
        const double divisor=std::max(1.,std::sqrt(dot(direction,direction,2)));
        for(int i=0;i<2;++i){direction[i]/=divisor;qcap[i]=st*direction[i];}
        qcap[2]=ct;
    }
    for(int i=0;i<3;++i)k[i]+=(beta*ell)*qcap[i];
    double padded[21];for(int i=0;i<7;++i)for(int j=0;j<3;++j)padded[3*i+j]=v[3*std::min<Index>(i,size-1)+j];
    Vec o;matmul(weights.data(),padded,o.data(),1,7,3,56,8,24,8);
    o[2]=std::min(o[2],k[2]);
    Vec d;for(int i=0;i<3;++i)d[i]=o[i]-k[i];
    const double upper=std::sqrt(dot(d.data(),d.data()))+g;
    return {n,lower,upper,g};
}

// Original verification deliberately computes a FLOAT64 world mean first,
// then promotes every subtraction, multiply and sum to native long double.
// Eq. (10), final vertex-side margin against the owner plane of each complete patch.
double coverage_minimum(const double* world, Index vertex_count, const Index* offsets,
                        Index piece_count, const double* A, const double* b,
                        Index face_count, const Index* owners) {
    Vec mean{0.,0.,0.};
    for(Index i=0;i<vertex_count;++i) for(int k=0;k<3;++k) mean[k]+=world[3*i+k];
    for(auto& x:mean) x/=static_cast<double>(vertex_count);
    std::vector<long double> local_bounds(face_count);
    std::vector<double> lengths(face_count);
    for(Index f=0;f<face_count;++f) {
        long double projection=0.L;
        for(int k=0;k<3;++k) projection+=static_cast<long double>(A[3*f+k])*static_cast<long double>(mean[k]);
        local_bounds[f]=static_cast<long double>(b[f])-projection;
        lengths[f]=rownorm(A+3*f);
    }
    long double minimum=std::numeric_limits<long double>::infinity();
    for(Index j=0;j<piece_count;++j) {
        const Index f=owners[j];
        for(Index i=offsets[j];i<offsets[j+1];++i) {
            long double products[3];
            for(int k=0;k<3;++k) {
                const long double local=static_cast<long double>(world[3*i+k])-static_cast<long double>(mean[k]);
                products[k]=local*static_cast<long double>(A[3*f+k]);
            }
            const long double value=((((0.L+products[0])+products[1])+products[2])-local_bounds[f])/
                                     static_cast<long double>(lengths[f]);
            minimum=minimum<value ? minimum:value;
        }
    }
    return static_cast<double>(minimum);
}

PyArrayObject* array_arg(PyObject* o, int typ, int ndim, const char* name) {
    if (!PyArray_Check(o)) {PyErr_Format(PyExc_ValueError,"%s must be an ndarray",name);throw PythonError{};}
    auto* a=reinterpret_cast<PyArrayObject*>(o);
    if (PyArray_TYPE(a)!=typ || PyArray_NDIM(a)!=ndim || !PyArray_ISCARRAY_RO(a) || !PyArray_ISNOTSWAPPED(a)) {
        PyErr_Format(PyExc_ValueError,"%s must be native-endian, C-contiguous, aligned, with the required dtype/rank",name);
        throw PythonError{};
    }
    return a;
}
void validate_geometry(PyArrayObject* vertices, PyArrayObject* offsets) {
    if(PyArray_DIM(vertices,1)!=3 || PyArray_DIM(offsets,0)<2) {
        PyErr_SetString(PyExc_ValueError,"PIECE_GEOMETRY: invalid native array shape");throw PythonError{};
    }
    const auto* o=static_cast<const Index*>(PyArray_DATA(offsets));
    const auto* v=static_cast<const double*>(PyArray_DATA(vertices));
    const Index count=PyArray_DIM(offsets,0)-1, nv=PyArray_DIM(vertices,0);
    if(o[0]!=0 || o[count]!=nv) {PyErr_SetString(PyExc_ValueError,"PIECE_GEOMETRY: invalid native offsets");throw PythonError{};}
    for(Index j=0;j<count;++j) if(o[j]<0 || o[j+1]<=o[j] || o[j+1]>nv || o[j+1]-o[j]>7) {
        PyErr_SetString(PyExc_ValueError,"PIECE_GEOMETRY: invalid native fragment size");throw PythonError{};
    }
    for(Index i=0;i<3*nv;++i) if(!std::isfinite(v[i])) {
        PyErr_SetString(PyExc_ValueError,"PIECE_GEOMETRY: nonfinite native vertex");throw PythonError{};
    }
}
PyObject* output_array(int type, const std::vector<npy_intp>& dims, const void* data) {
    PyObject* obj=PyArray_SimpleNew(static_cast<int>(dims.size()),const_cast<npy_intp*>(dims.data()),type);
    if(!obj) throw PythonError{};
    auto* arr=reinterpret_cast<PyArrayObject*>(obj);
    if(PyArray_NBYTES(arr)>0) std::memcpy(PyArray_DATA(arr),data,PyArray_NBYTES(arr));
    return obj;
}
void put(PyObject* d,const char* key,PyObject* value) {
    Ref owned(value);if(!owned.p || PyDict_SetItemString(d,key,owned.p)<0)throw PythonError{};
}
struct Face {
    Index selected, source_id, covered, full;
    double lower, upper, gap, margin;
};
[[noreturn]] void seed_failure(Index j,double upper,double margin,Index covered,Index count) {
    Ref report(Py_BuildValue("{s:s,s:s,s:L,s:d,s:d,s:L,s:L,s:O}",
        "schema","SCPSC_WHOLE_FRAGMENT_SEPARATION_V1",
        "message","a source fragment cannot retain the required envelope seed",
        "selected_piece",static_cast<long long>(j),"distance_upper_m",upper,
        "minimum_normal_margin_m",margin,"covered_piece_count",static_cast<long long>(covered),
        "source_piece_count",static_cast<long long>(count),"physical_task_infeasibility_proven",Py_False));
    if(!report.p)throw PythonError{};
    Ref exc(PyObject_CallFunctionObjArgs(plane_failure,report.p,nullptr));
    if(!exc.p)throw PythonError{};
    PyErr_SetObject(plane_failure,exc.p);throw PythonError{};
}

PyObject* separate(PyObject*,PyObject* args) {
    PyObject *vertices_obj,*offsets_obj,*ids_obj,*seeds_obj,*parameters_obj,*callback,*check;
    int fast,lazy,max_iter;
    if(!PyArg_ParseTuple(args,"OOOOOiiiOO",&vertices_obj,&offsets_obj,&ids_obj,&seeds_obj,
                         &parameters_obj,&fast,&lazy,&max_iter,&callback,&check))return nullptr;
    try {
        auto* vertices=array_arg(vertices_obj,NPY_DOUBLE,2,"vertices");
        auto* offset_array=array_arg(offsets_obj,NPY_INT64,1,"offsets");
        auto* ids_array=array_arg(ids_obj,NPY_INT64,1,"source_ids");
        auto* seeds=array_arg(seeds_obj,NPY_DOUBLE,2,"seeds");
        auto* par=array_arg(parameters_obj,NPY_DOUBLE,1,"parameters");
        validate_geometry(vertices,offset_array);
        const Index count=PyArray_DIM(offset_array,0)-1,nv=PyArray_DIM(vertices,0);
        if(PyArray_DIM(ids_array,0)!=count || PyArray_DIM(seeds,0)!=2 || PyArray_DIM(seeds,1)!=3 ||
           PyArray_DIM(par,0)!=8 || max_iter<1 || !PyCallable_Check(callback) ||
           (check!=Py_None && !PyCallable_Check(check))) {
            PyErr_SetString(PyExc_ValueError,"SCPSC: invalid native argument shape/options");throw PythonError{};
        }
        const auto* world=static_cast<const double*>(PyArray_DATA(vertices));
        const auto* sw=static_cast<const double*>(PyArray_DATA(seeds));
        const auto* offsets=static_cast<const Index*>(PyArray_DATA(offset_array));
        const auto* ids=static_cast<const Index*>(PyArray_DATA(ids_array));
        const auto* p=static_cast<const double*>(PyArray_DATA(par));
        const double ell=p[0],rL=p[1],rQ=p[2],theta=p[3],eps=p[4],seed_tol=p[5],tol=p[6],nearest_tol=p[7];
        for(int k=0;k<8;++k) if(!std::isfinite(p[k])) {PyErr_SetString(PyExc_ValueError,"SCPSC: nonfinite option");throw PythonError{};}
        for(int k=0;k<6;++k) if(!std::isfinite(sw[k])) {PyErr_SetString(PyExc_ValueError,"SCPSC: nonfinite seed");throw PythonError{};}
        if(std::min({ell,rL,rQ})<=0. || theta<0. || theta>=std::acos(-1.)/2. || eps<0. || seed_tol<0. || tol<=0. || nearest_tol<=0.) {
            PyErr_SetString(PyExc_ValueError,"SCPSC: invalid native options");throw PythonError{};
        }
        Vec origin;std::array<double,6> s;
        std::vector<double> v(3*nv),geom(7*count),normals(3*count),lengths(count),lower(count),upper(count),projection(nv);
        for(int k=0;k<3;++k) origin[k]=((0.+sw[k])+sw[3+k])/2.;
        for(int k=0;k<6;++k) s[k]=sw[k]-origin[k%3];
        for(Index i=0;i<3*nv;++i) v[i]=world[i]-origin[i%3];
        checkpoint(check);
        auto tick=Clock::now();
        const double geom_tol=std::min(nearest_tol*.05,tol*10.);
        const int status=scpsc_payload_brackets(v.data(),nv,offsets,count,s.data(),geom_tol,max_iter,geom.data());
        if(status!=0)fail("PIECE_DISTANCE_FAILED: native geometry status");
        for(double x:geom)if(!std::isfinite(x))fail("PIECE_DISTANCE_FAILED: nonfinite geometry");
        for(Index j=0;j<count;++j)if(geom[7*j+3]<0. || geom[7*j+4]<geom[7*j+3]-geom_tol)fail("PIECE_DISTANCE_FAILED: inconsistent geometric bounds");
        const double payload_seconds=elapsed(tick);
        Index iterations=0;
        for(Index j=0;j<count;++j) {
            const double len=rownorm(geom.data()+7*j);
            for(int k=0;k<3;++k)normals[3*j+k]=geom[7*j+k]/len;
            lengths[j]=rownorm(normals.data()+3*j);
            iterations+=static_cast<Index>(geom[7*j+5]);
        }
        const auto support=supports(normals.data(),count,ell,rL,rQ,theta);
        for(Index j=0;j<count;++j) for(Index i=offsets[j];i<offsets[j+1];++i)
            projection[i]=((0.+v[3*i]*normals[3*j])+v[3*i+1]*normals[3*j+1])+v[3*i+2]*normals[3*j+2];
        std::vector<double> seed_projections(2*count);
        matmul(normals.data(),s.data(),seed_projections.data(),count,3,2,24,8,8,24);
        const double g=guard(v.data(),v.size(),s.data(),6,ell,rL,rQ);
        const double wg=guard(world,3*nv,origin.data(),3,ell,rL,rQ);
        std::vector<unsigned char> available(count),full_done(count,0),active(count,1);
        std::vector<Index> owners(count,-1);
        std::vector<Index> active_indices(count);
        Index fast_count=0,full_solves=0,active_count=count,vertex_projections=0;
        for(Index j=0;j<count;++j) {
            active_indices[j]=j;
            const double source_min=reduce_min(projection.data()+offsets[j],offsets[j+1]-offsets[j]);
            const double seed_max=npmax(seed_projections[2*j],seed_projections[2*j+1]);
            lower[j]=npmax(0.,((source_min-seed_max)-support[j])/lengths[j]-g);
            upper[j]=npmax(0.,geom[7*j+4]-rL)+g;
            bool eligible=support[j]<=rL*lengths[j] && upper[j]-lower[j]<=nearest_tol &&
                          geom[7*j+6]==1. && lower[j]>eps+g;
            fast_count+=eligible;available[j]=eligible && fast;
        }
        std::vector<double> out_n,out_b;
        std::vector<Face> faces;
        double nearest_seconds=0.,coverage_seconds=0.;
        Ref seed_view;
        std::array<double,3> support_scratch;
        auto solve_full=[&](Index j) {
            if(full_done[j])fail("PIECE_DISTANCE_UNCERTAIN: repeated unresolved cached distance");
            checkpoint(check);
            Ref vertex_view(output_array(NPY_DOUBLE,{offsets[j+1]-offsets[j],3},v.data()+3*offsets[j]));
            if (!seed_view.p) {
                seed_view.p=output_array(NPY_DOUBLE,{2,3},s.data());
                PyArray_CLEARFLAGS(reinterpret_cast<PyArrayObject*>(seed_view.p),NPY_ARRAY_WRITEABLE);
            }
            Ref answer(PyObject_CallFunction(callback,"LOOd",static_cast<long long>(j),vertex_view.p,seed_view.p,nearest_tol*.25));
            if(!answer.p)throw PythonError{};
            if(!PyTuple_Check(answer.p) || PyTuple_GET_SIZE(answer.p)!=3) {
                PyErr_SetString(PyExc_ValueError,"SCPSC oracle must return (normal, lower, upper)");throw PythonError{};
            }
            auto* nn=array_arg(PyTuple_GET_ITEM(answer.p,0),NPY_DOUBLE,1,"oracle normal");
            if(PyArray_DIM(nn,0)!=3){PyErr_SetString(PyExc_ValueError,"SCPSC: invalid oracle normal");throw PythonError{};}
            const double lb=PyFloat_AsDouble(PyTuple_GET_ITEM(answer.p,1));
            const double ub=PyFloat_AsDouble(PyTuple_GET_ITEM(answer.p,2));check_python();
            const auto* data=static_cast<const double*>(PyArray_DATA(nn));
            if(!std::isfinite(lb)||!std::isfinite(ub)||!std::isfinite(data[0])||!std::isfinite(data[1])||!std::isfinite(data[2]))fail("PIECE_SOCP_UNRESOLVED: nonfinite solution");
            std::copy(data,data+3,normals.data()+3*j);
            lower[j]=std::max(lower[j],lb);upper[j]=std::min(upper[j],ub);
            if(lower[j]>upper[j]+g)fail("PIECE_DISTANCE_BRACKET_CONFLICT");
            upper[j]=std::max(lower[j],upper[j]);full_done[j]=1;available[j]=1;++full_solves;
        };
        if(!lazy)for(Index j=0;j<count;++j)if(!available[j])solve_full(j);
        while(active_count) {
            checkpoint(check);tick=Clock::now();
            Index j=-1,blocker=-1;
            for(;;) {
                j=blocker=-1;
                for(Index i:active_indices) {
                    if(j<0 || upper[i]<upper[j])j=i;
                    if(blocker<0 || lower[i]<lower[blocker])blocker=i;
                }
                if(lower[blocker]<upper[j]-nearest_tol){solve_full(blocker);continue;}
                if(!available[j]){solve_full(j);continue;}
                break;
            }
            const double gap=std::max(0.,upper[j]-lower[blocker]);
            nearest_seconds+=elapsed(tick);
            const auto n=normal(normals.data()+3*j);
            const Index size=offsets[j+1]-offsets[j];
            std::array<double,7> selected_projection;
            matmul(v.data()+3*offsets[j],n.data(),selected_projection.data(),size,3,1,24,8,8,8);
            const double b_local=(reduce_min(selected_projection.data(),size)-eps)-wg;
            const double b_world=b_local+dot(n.data(),origin.data());
            long double product[3];for(int k=0;k<3;++k)product[k]=static_cast<long double>(n[k])*static_cast<long double>(origin[k]);
            const double actual_b=static_cast<double>(static_cast<long double>(b_world)-(((0.L+product[0])+product[1])+product[2]));
            const double h=support_one(n.data(),ell,rL,rQ,theta,support_scratch);
            double seed_pr[2];matmul(s.data(),n.data(),seed_pr,2,3,1,24,8,8,8);
            const double margin=((actual_b-npmax(seed_pr[0],seed_pr[1]))-h)/std::sqrt(dot(n.data(),n.data()));
            if(margin < -seed_tol) {
                if(upper[j]<=(eps+wg)+seed_tol) seed_failure(j,upper[j],margin,count-active_count,count);
                fail("PIECE_SOCP_CERTIFICATE_UNCERTAIN: selected normal lost positive seed separation");
            }
            tick=Clock::now();
            matmul(v.data(),n.data(),projection.data(),nv,3,1,24,8,8,8);  // Eq. (10), all vertex projections for this plane.
            vertex_projections+=nv;
            // The projection kernel and its shape stay identical. A fragment
            // with an assigned face never changes owner, so skip its reduction.
            const Index face=static_cast<Index>(faces.size());
            Index new_covered=0,remaining=0;
            for(Index i:active_indices) {
                const double minimum=reduce_min(projection.data()+offsets[i],offsets[i+1]-offsets[i]);
                if(minimum-actual_b>=eps+g){owners[i]=face;active[i]=0;++new_covered;}
                else active_indices[remaining++]=i;
            }
            active_indices.resize(static_cast<std::size_t>(remaining));
            if(active[j])fail("PIECE_COVERAGE_NUMERICALLY_UNCERTAIN: target fragment not certified");
            active_count-=new_covered;
            out_n.insert(out_n.end(),n.begin(),n.end());out_b.push_back(b_world);
            faces.push_back({j,ids[j],new_covered,full_done[j],lower[j],upper[j],gap,margin});
            coverage_seconds+=elapsed(tick);
            if(static_cast<Index>(faces.size())>count){PyErr_SetString(PyExc_AssertionError,"each round must remove a previously uncovered piece");throw PythonError{};}
        }
        tick=Clock::now();
        const double minimum=coverage_minimum(world,nv,offsets,count,out_n.data(),out_b.data(),faces.size(),owners.data());
        if(!(minimum>=0.))fail("PIECE_FINAL_COVERAGE_FAILED");
        const double certify_seconds=elapsed(tick);
        const npy_intp nf=static_cast<npy_intp>(faces.size());
        // Returning the constructed planes is part of construction. The other
        // arrays and fields below exist only for certificates and reports.
        Ref out(PyDict_New());if(!out.p)throw PythonError{};
        put(out.p,"A",output_array(NPY_DOUBLE,{nf,3},out_n.data()));
        put(out.p,"b",output_array(NPY_DOUBLE,{nf},out_b.data()));
        tick=Clock::now();
        std::vector<Index> face_ids(nf*4);
        std::vector<double> face_values(nf*4);
        for(npy_intp i=0;i<nf;++i) {
            const auto& f=faces[i];
            face_ids[4*i]=f.selected;face_ids[4*i+1]=f.source_id;face_ids[4*i+2]=f.covered;face_ids[4*i+3]=f.full;
            face_values[4*i]=f.lower;face_values[4*i+1]=f.upper;face_values[4*i+2]=f.gap;face_values[4*i+3]=f.margin;
        }
        put(out.p,"owners",output_array(NPY_INT64,{count},owners.data()));
        put(out.p,"face_ids",output_array(NPY_INT64,{nf,4},face_ids.data()));
        put(out.p,"face_values",output_array(NPY_DOUBLE,{nf,4},face_values.data()));
        put(out.p,"local_vertices",output_array(NPY_DOUBLE,{nv,3},v.data()));
        put(out.p,"local_seeds",output_array(NPY_DOUBLE,{2,3},s.data()));
        put(out.p,"coverage_minimum",PyFloat_FromDouble(minimum));
        put(out.p,"iterations",PyLong_FromLongLong(iterations));
        put(out.p,"fast_count",PyLong_FromLongLong(fast_count));
        put(out.p,"full_solves",PyLong_FromLongLong(full_solves));
        put(out.p,"vertex_projections",PyLong_FromLongLong(vertex_projections));
        put(out.p,"payload_seconds",PyFloat_FromDouble(payload_seconds));
        put(out.p,"nearest_seconds",PyFloat_FromDouble(nearest_seconds));
        put(out.p,"coverage_seconds",PyFloat_FromDouble(coverage_seconds));
        put(out.p,"certify_seconds",PyFloat_FromDouble(certify_seconds));
        put(out.p,"statistics_seconds",PyFloat_FromDouble(elapsed(tick)));
        return out.release();
    } catch(const PythonError&) {return nullptr;}
      catch(const std::bad_alloc&) {return PyErr_NoMemory();}
      catch(const std::exception& e) {PyErr_SetString(PyExc_RuntimeError,e.what());return nullptr;}
}


PyObject* solve_oracle(PyObject*,PyObject* args) {
    PyObject *vo,*so,*po,*callback;double tolerance;
    if(!PyArg_ParseTuple(args,"OOOdO",&vo,&so,&po,&tolerance,&callback))return nullptr;
    try {
        auto* v=array_arg(vo,NPY_DOUBLE,2,"vertices");
        auto* seeds=array_arg(so,NPY_DOUBLE,2,"seeds");
        auto* par=array_arg(po,NPY_DOUBLE,1,"parameters");
        const Index size=PyArray_DIM(v,0);
        if(size<1 || size>7 || PyArray_DIM(v,1)!=3 || PyArray_DIM(seeds,0)!=2 ||
           PyArray_DIM(seeds,1)!=3 || PyArray_DIM(par,0)!=5 || !PyCallable_Check(callback) ||
           !(tolerance>0.) || !std::isfinite(tolerance)) {
            PyErr_SetString(PyExc_ValueError,"SCPSC: invalid oracle arguments");throw PythonError{};
        }
        const auto* vp=static_cast<double*>(PyArray_DATA(v));
        const auto* sp=static_cast<double*>(PyArray_DATA(seeds));
        const auto* p=static_cast<double*>(PyArray_DATA(par));
        for(Index i=0;i<3*size;++i)if(!std::isfinite(vp[i]))fail("PIECE_SOCP_UNRESOLVED: nonfinite solution");
        for(int i=0;i<6;++i)if(!std::isfinite(sp[i]))fail("PIECE_SOCP_UNRESOLVED: nonfinite solution");
        for(int i=0;i<5;++i)if(!std::isfinite(p[i]))fail("PIECE_SOCP_UNRESOLVED: nonfinite solution");
        OracleBounds bounds{};
        std::array<double,3> support_scratch;
        std::array<double,2> theta_trig;
        unary(sin_loop,p+3,theta_trig.data(),1);
        unary(cos_loop,p+3,theta_trig.data()+1,1);
        for(int attempt=0;attempt<2;++attempt) {
            Ref answer(PyObject_CallFunction(callback,"Od",vo,p[4]*(attempt ? .01:1.)));
            if(!answer.p)throw PythonError{};
            if(!PyTuple_Check(answer.p) || PyTuple_GET_SIZE(answer.p)!=4) {
                PyErr_SetString(PyExc_ValueError,"SCPSC raw oracle must return (x, z, status, record)");throw PythonError{};
            }
            const char* state=PyUnicode_AsUTF8(PyTuple_GET_ITEM(answer.p,2));check_python();
            if(std::strcmp(state,"Solved") && std::strcmp(state,"AlmostSolved") &&
               std::strcmp(state,"MaxIterations") && std::strcmp(state,"InsufficientProgress")) {
                PyErr_Format(solver_error,"PIECE_SOCP_UNRESOLVED: %s",state);throw PythonError{};
            }
            auto* x=array_arg(PyTuple_GET_ITEM(answer.p,0),NPY_DOUBLE,1,"oracle x");
            auto* z=array_arg(PyTuple_GET_ITEM(answer.p,1),NPY_DOUBLE,1,"oracle z");
            auto* rec=PyTuple_GET_ITEM(answer.p,3);
            if(PyArray_DIM(x,0)!=7 || PyArray_DIM(z,0)!=21 || !PyDict_Check(rec))fail("PIECE_SOCP_UNRESOLVED: nonfinite solution");
            const auto* xp=static_cast<double*>(PyArray_DATA(x));const auto* zp=static_cast<double*>(PyArray_DATA(z));
            for(int i=0;i<7;++i)if(!std::isfinite(xp[i]))fail("PIECE_SOCP_UNRESOLVED: nonfinite solution");
            for(int i=0;i<21;++i)if(!std::isfinite(zp[i]))fail("PIECE_SOCP_UNRESOLVED: nonfinite solution");
            bounds=oracle_bounds(vp,size,sp,xp,zp,p[0],p[1],p[2],p[3],support_scratch,theta_trig);
            const auto statistics_started=Clock::now();
            put(rec,"distance_lower_m",PyFloat_FromDouble(bounds.lower));
            put(rec,"distance_upper_m",PyFloat_FromDouble(bounds.upper));
            put(rec,"geometric_primal_dual_gap_m",PyFloat_FromDouble(bounds.upper-bounds.lower));
            put(rec,"native_report_statistics_seconds",PyFloat_FromDouble(elapsed(statistics_started)));
            if(bounds.lower<=bounds.upper+bounds.guard && bounds.upper-bounds.lower<=tolerance) {
                Ref n(output_array(NPY_DOUBLE,{3},bounds.normal.data()));
                return Py_BuildValue("OddO",n.p,bounds.lower,std::max(bounds.lower,bounds.upper),rec);
            }
        }
        char message[240];
        std::snprintf(message,sizeof(message),"PIECE_SOCP_DISTANCE_UNCERTAIN: valid plane but nearest-distance bracket did not close; gap=%.9g m",bounds.upper-bounds.lower);
        fail(message);
    }catch(const PythonError&){return nullptr;}
     catch(const std::bad_alloc&){return PyErr_NoMemory();}
     catch(const std::exception& e){PyErr_SetString(PyExc_RuntimeError,e.what());return nullptr;}
}

PyObject* verify(PyObject*,PyObject* args) {
    PyObject *vo,*oo,*ao,*bo,*owno;
    if(!PyArg_ParseTuple(args,"OOOOO",&vo,&oo,&ao,&bo,&owno))return nullptr;
    try {
        auto* v=array_arg(vo,NPY_DOUBLE,2,"vertices");auto* o=array_arg(oo,NPY_INT64,1,"offsets");
        auto* a=array_arg(ao,NPY_DOUBLE,2,"normals");auto* b=array_arg(bo,NPY_DOUBLE,1,"bounds");
        auto* ow=array_arg(owno,NPY_INT64,1,"owners");validate_geometry(v,o);
        const Index count=PyArray_DIM(o,0)-1,nf=PyArray_DIM(a,0);
        const auto* ap=static_cast<const double*>(PyArray_DATA(a));
        const auto* bp=static_cast<const double*>(PyArray_DATA(b));
        const auto* op=static_cast<const Index*>(PyArray_DATA(ow));
        bool valid=PyArray_DIM(a,1)==3 && nf>0 && PyArray_DIM(b,0)==nf && PyArray_DIM(ow,0)==count;
        if(valid) {
            for(Index j=0;j<count;++j) valid=valid && op[j]>=0 && op[j]<nf;
            for(Index f=0;f<nf;++f)valid=valid && std::isfinite(bp[f]) && std::isfinite(ap[3*f]) &&
                std::isfinite(ap[3*f+1]) && std::isfinite(ap[3*f+2]) && ap[3*f+2]<=0. && rownorm(ap+3*f)>0.;
        }
        if(!valid){PyErr_SetString(PyExc_ValueError,"INVALID_PIECE_COVERAGE_WITNESS");throw PythonError{};}
        return PyFloat_FromDouble(coverage_minimum(static_cast<const double*>(PyArray_DATA(v)),PyArray_DIM(v,0),
            static_cast<const Index*>(PyArray_DATA(o)),count,ap,bp,nf,op));
    }catch(const PythonError&){return nullptr;}
     catch(const std::bad_alloc&){return PyErr_NoMemory();}
     catch(const std::exception& e){PyErr_SetString(PyExc_RuntimeError,e.what());return nullptr;}
}
// Test hook for the numerical dependency bridge; not called by production.
PyObject* support_test(PyObject*,PyObject* args) {
    PyObject* obj;double ell,rL,rQ,theta;
    if(!PyArg_ParseTuple(args,"Odddd",&obj,&ell,&rL,&rQ,&theta))return nullptr;
    try {
        auto* a=array_arg(obj,NPY_DOUBLE,2,"normals");
        if(PyArray_DIM(a,1)!=3){PyErr_SetString(PyExc_ValueError,"normals must be Nx3");throw PythonError{};}
        auto out=supports(static_cast<double*>(PyArray_DATA(a)),PyArray_DIM(a,0),ell,rL,rQ,theta);
        return output_array(NPY_DOUBLE,{PyArray_DIM(a,0)},out.data());
    }catch(const PythonError&){return nullptr;}catch(const std::bad_alloc&){return PyErr_NoMemory();}
}
PyMethodDef methods[]={
    {"solve_oracle",solve_oracle,METH_VARARGS,"Native SOCP retries and feasible-witness bounds; unchanged Python solver calls."},
    {"separate",separate,METH_VARARGS,"Execute the complete SCPSC segment numeric flow."},
    {"verify",verify,METH_VARARGS,"Check whole-fragment ownership in native long double."},
    {"_support_test",support_test,METH_VARARGS,"Differential-test hook for fixed support."},
    {nullptr,nullptr,0,nullptr}};
void free_module(void*) {
    for(auto* x:kernel_owners) {Py_XDECREF(x);}
    kernel_owners.clear();
    Py_CLEAR(solver_error);Py_CLEAR(plane_failure);
}
PyModuleDef module={PyModuleDef_HEAD_INIT,"_scpsc_native",nullptr,-1,methods,nullptr,nullptr,nullptr,free_module};
} // namespace

PyMODINIT_FUNC PyInit__scpsc_native() {
    import_array();import_umath();
    try {
        Ref numpy(PyImport_ImportModule("numpy"));if(!numpy.p)throw PythonError{};
        acos_loop=get_loop(numpy.p,"arccos",2);cos_loop=get_loop(numpy.p,"cos",2);
        sin_loop=get_loop(numpy.p,"sin",2);add_loop=get_loop(numpy.p,"add",3);
        matmul_loop=get_loop(numpy.p,"matmul",3);minimum_loop=get_loop(numpy.p,"minimum",3);
        Ref errors(PyImport_ImportModule("scpsc.plane_contract"));if(!errors.p)throw PythonError{};
        solver_error=PyObject_GetAttrString(errors.p,"SeedPlaneSolverError");
        plane_failure=PyObject_GetAttrString(errors.p,"SeedPlaneFailure");
        if(!solver_error || !plane_failure)throw PythonError{};
        Ref out(PyModule_Create(&module));if(!out.p)throw PythonError{};
        if(PyModule_AddIntConstant(out.p,"longdouble_mantissa_bits",std::numeric_limits<long double>::digits)<0)throw PythonError{};
        return out.release();
    }catch(const PythonError&){free_module(nullptr);return nullptr;}
     catch(const std::bad_alloc&){free_module(nullptr);return PyErr_NoMemory();}
}
