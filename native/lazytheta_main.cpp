// Geometry-only I/O. Lazy Theta* kernel retained verbatim from the supplied Sept28 source.
#include "closed_voxel_union.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Point = std::array<double, 3>;
using Cell = std::array<int, 3>;
using Clock = std::chrono::steady_clock;
static double distance(const Point& a, const Point& b) {
  double v = 0; for (int k=0;k<3;++k) v += (a[k]-b[k])*(a[k]-b[k]);
  return std::sqrt(v);
}
struct Stats {
  std::string status = "ERROR", detail;
  size_t expanded = 0, los_checks = 0, parent_repairs = 0;
  double elapsed = 0, length = 0;
  bool search_found = false;
};

struct Grid {
  Cell n{};
  double h = 0;
  Point origin{}, start{}, goal{};
  unsigned seed = 0;
  std::vector<uint8_t> data;
  bool valid(Cell c) const {
    return c[0]>=0 && c[1]>=0 && c[2]>=0 && c[0]<n[0] && c[1]<n[1] && c[2]<n[2];
  }
  int id(Cell c) const { return c[0]+n[0]*(c[1]+n[1]*c[2]); }
  Cell cell(int i) const { return {i%n[0], (i/n[0])%n[1], i/(n[0]*n[1])}; }
  bool free(Cell c) const { return valid(c) && data[id(c)] != 0; }
  Point center(Cell c) const { return {origin[0]+h*c[0],origin[1]+h*c[1],origin[2]+h*c[2]}; }
  Point center(int i) const { return center(cell(i)); }
  Cell nearest(Point p) const {
    Cell c; for(int k=0;k<3;++k) c[k]=int(std::floor((p[k]-origin[k])/h+0.5));
    return c;
  }
  void read(const std::string& path) {
    std::ifstream in(path);
    std::string grid_file;
    std::getline(in,grid_file);
    if(!grid_file.empty() && grid_file.back()=='\r') grid_file.pop_back();
    in >> n[0] >> n[1] >> n[2] >> h;
    for(double& x: origin) in >> x;
    for(double& x: start) in >> x;
    for(double& x: goal) in >> x;
    in >> seed;
    if(!in || h<=0 || n[0]<=0 || n[1]<=0 || n[2]<=0)
      throw std::runtime_error("invalid request dimensions or spacing");
    std::string legacy_limit;
    if(in >> legacy_limit)
      std::cerr << "Deprecated native wall-clock limit ignored: " << legacy_limit << '\n';
    size_t size=size_t(n[0])*n[1]*n[2];
    if(size>size_t(std::numeric_limits<int>::max())) throw std::runtime_error("grid index exceeds int range");
    for(auto p: {origin,start,goal}) for(double x:p)
      if(!std::isfinite(x)) throw std::runtime_error("nonfinite request position");
    data.resize(size);
    std::ifstream raw(grid_file,std::ios::binary);
    if(!raw.read(reinterpret_cast<char*>(data.data()),data.size()))
      throw std::runtime_error("missing or short free_grid.bin");
  }

  bool covered_grid_point(const Point& u) const {
    return closed_voxel::covered_grid_point(u, [this](Cell c) { return free(c); });
  }
  bool los(Point p,Point q) const {
    return closed_voxel::segment(p, q, origin, h, [this](Cell c) { return free(c); });
  }
};

static std::vector<Point> lazy_theta(const Grid& map,int start,int goal,Stats& stats) {
  const double inf=std::numeric_limits<double>::infinity();
  struct Node { double f,g; int id; };
  struct Worse { bool operator()(const Node& a,const Node& b) const {
    if(a.f!=b.f) return a.f>b.f;
    if(a.g!=b.g) return a.g<b.g;
    return a.id>b.id;
  }};
  std::priority_queue<Node,std::vector<Node>,Worse> open;
  std::vector<double> g(map.data.size(),inf);
  std::vector<int> parent(map.data.size(),-1);
  std::vector<uint8_t> closed(map.data.size(),0);
  auto visible=[&](int a,int b) { ++stats.los_checks; return map.los(map.center(a),map.center(b)); };
  auto neighbors=[&](int s) {
    std::vector<int> out; const Cell c=map.cell(s);
    for(int z=-1;z<=1;++z) for(int y=-1;y<=1;++y) for(int x=-1;x<=1;++x) {
      if(x==0 && y==0 && z==0) continue;
      Cell q{c[0]+x,c[1]+y,c[2]+z};
      // Neighbor centers in two certified closed voxels are always joined
      // inside their union, including diagonal contact. No long LOS needed.
      if(map.free(q)) out.push_back(map.id(q));
    }
    return out;
  };
  auto dist=[&](int a,int b) { return distance(map.center(a),map.center(b)); };
  g[start]=0; parent[start]=start;
  open.push({dist(start,goal),0,start});
  while(!open.empty()) {
    Node top=open.top(); open.pop(); int s=top.id;
    if(closed[s] || top.g!=g[s]) continue;
    // SetVertex, Algorithm 3: validate the optimistic parent only on pop.
    std::vector<int> n;
    if(!visible(parent[s],s)) {
      ++stats.parent_repairs; n=neighbors(s); g[s]=inf; parent[s]=-1;
      for(int v:n) if(closed[v] && g[v]+dist(v,s)<g[s]) {
        g[s]=g[v]+dist(v,s); parent[s]=v;
      }
      if(parent[s]<0) throw std::runtime_error("Lazy Theta SetVertex has no closed visible neighbor");
    }
    ++stats.expanded;
    if(s==goal) {
      stats.search_found=true;
      std::vector<Point> route;
      for(int v=goal;;v=parent[v]) { route.push_back(map.center(v)); if(v==start) break; }
      std::reverse(route.begin(),route.end()); return route;
    }
    closed[s]=1;
    if(n.empty()) n=neighbors(s);
    // ComputeCost uses ONLY the optimistic Path 2. No LOS here.
    for(int v:n) if(!closed[v]) {
      const int p=parent[s]; const double candidate=g[p]+dist(p,v);
      if(candidate<g[v]) {
        g[v]=candidate; parent[v]=p;
        open.push({candidate+dist(v,goal),candidate,v});
      }
    }
  }
  stats.status="NO_PATH"; return {};
}


static std::string quote(const std::string& s) {
  std::ostringstream out;out << '"';
  for(unsigned char c:s){if(c=='"'||c=='\\')out<<'\\'<<c;else if(c=='\n')out<<"\\n";
    else if(c>=32)out<<c;}
  out << '"';return out.str();
}
int main(int argc,char** argv){
  if(argc!=4){std::cerr<<"usage: lazytheta_runner request.txt route.txt status.json\n";return 2;}
  Stats stats;std::vector<Point> route;const auto begin=Clock::now();int code=0;
  try{
    Grid map;map.read(argv[1]);
    const Cell start=map.nearest(map.start),goal=map.nearest(map.goal);
    if(!map.free(start)||!map.free(goal))throw std::runtime_error("ANCHOR_VOXEL_NOT_FREE");
    if(!map.los(map.start,map.center(start))||!map.los(map.center(goal),map.goal))
      throw std::runtime_error("ANCHOR_TO_GRID_CONNECTOR_NOT_COVERED");
    route=lazy_theta(map,map.id(start),map.id(goal),stats);
    if(route.empty()){stats.status="NO_PATH";code=3;}
    else{
      if(distance(map.start,route.front())>1e-10)route.insert(route.begin(),map.start);
      if(distance(map.goal,route.back())>1e-10)route.push_back(map.goal);
      for(size_t i=1;i<route.size();++i){
        if(!map.los(route[i-1],route[i]))throw std::runtime_error("FINAL_ROUTE_VISIBILITY_FAILED"); // Eq. (14), covered by the certified voxel union.
        stats.length+=distance(route[i-1],route[i]);
      }
      stats.status="ROUTE_FOUND";
      std::ofstream out(argv[2]);out<<std::setprecision(17);
      if(!out)throw std::runtime_error("Cannot write route");
      for(auto v:route)out<<v[0]<<' '<<v[1]<<' '<<v[2]<<'\n';
    }
  }catch(const std::exception& e){stats.status="INPUT_OR_GEOMETRY_ERROR";stats.detail=e.what();code=2;}
  stats.elapsed=std::chrono::duration<double>(Clock::now()-begin).count();
  std::ofstream out(argv[3]);out<<std::setprecision(17)<<"{\"status\":"<<quote(stats.status)
    <<",\"detail\":"<<quote(stats.detail)<<",\"algorithm\":\"LAZY_THETA_STAR\",\"heuristic_weight\":1"
    <<",\"R_calls\":0,\"expanded\":"<<stats.expanded<<",\"los_checks\":"<<stats.los_checks
    <<",\"parent_repairs\":"<<stats.parent_repairs<<",\"seconds\":"<<stats.elapsed
    <<",\"length_m\":"<<stats.length<<"}\n";
  return code;
}
