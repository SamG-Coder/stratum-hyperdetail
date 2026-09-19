// CPU reference and regression tests for the SAME authored CUDA used by WebGPU.
// No claim of hardware GPU frame rate is made by this harness.
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <stdexcept>
#define __device__
#define __global__
#define __shared__
#define __syncthreads() ((void)0)
struct Index {unsigned x=0,y=0,z=0;};
thread_local Index threadIdx,blockIdx,blockDim,gridDim;
struct float2 {float x,y;};struct float3 {float x,y,z;};struct float4{float x,y,z,w;};
float2 make_float2(float x,float y){return {x,y};}float3 make_float3(float x,float y,float z){return {x,y,z};}float4 make_float4(float x,float y,float z,float w){return{x,y,z,w};}
float3 operator+(float3 a,float3 b){return{a.x+b.x,a.y+b.y,a.z+b.z};}float3 operator-(float3 a,float3 b){return{a.x-b.x,a.y-b.y,a.z-b.z};}
float3 operator*(float3 a,float b){return{a.x*b,a.y*b,a.z*b};}float3 operator*(float b,float3 a){return a*b;}float3 operator*(float3 a,float3 b){return{a.x*b.x,a.y*b.y,a.z*b.z};}
float3 operator/(float3 a,float b){return{a.x/b,a.y/b,a.z/b};}
unsigned int atomicAdd(unsigned int* p,unsigned int v){auto old=*p;*p+=v;return old;}
#include "../Stratum.cu"
// Frozen pre-refactor emitter: check identities, not only a new implementation against itself.
namespace reference {
constexpr int CITY=64,PER_CLUSTER=32,PRIMS=2048,PS=16,REQUESTS=8,NODES=4096,MS=12;
constexpr float HALF_CITY=1152;
int pageIndex(int x,int z){return imod(z,16)*16+imod(x,16);}
bool inCity(int x,int z){return x>=-32&&x<32&&z>=-32&&z<32;}
int worldIndex(int x,int z){return(z+32)*64+x+32;}
#include "reference/assets-original.cu"
#define localPoint legacyLocalPoint
#define worldNormal legacyWorldNormal
#define leafNormal legacyLeafNormal
#define foliageHit legacyFoliageHit
#include "reference/trace-original.cu"
#undef localPoint
#undef worldNormal
#undef leafNormal
#undef foliageHit
}
void scalar(){threadIdx={};blockIdx={};blockDim={1,1,1};}
template<class F>void dispatch(int n,F f){for(int i=0;i<n;i++){blockDim={64,1,1};blockIdx={unsigned(i/64),0,0};threadIdx={unsigned(i%64),0,0};f();}}
void require(bool ok,const char* name){if(!ok)throw std::runtime_error(name);std::cout<<"PASS "<<name<<"\n";}
std::vector<Feature> enumerate(Lot l,int g){Sink count=newSink(1,{0,0,0},{0,0,1},FAR);count=authoredGroup(l,g,count);std::vector<Feature> out;for(int i=0;i<count.count;i++){Sink s=newSink(2,{0,0,0},{0,0,1},FAR);s.target=i;s=authoredGroup(l,g,s);out.push_back(s.feature);}return out;}
std::vector<float> bounds(Lot l){std::vector<float> n(GROUP_NODES*8);for(int g=0;g<CLUSTERS;g++){Sink s=newSink(1,{0,0,0},{0,0,1},FAR);s=authoredGroup(l,g,s);int b=(CLUSTERS+g)*8;n[b]=s.lo.x;n[b+1]=s.lo.y;n[b+2]=s.lo.z;n[b+3]=s.count;n[b+4]=s.hi.x;n[b+5]=s.hi.y;n[b+6]=s.hi.z;}
for(int j=CLUSTERS-1;j>=1;j--){int b=j*8,a=j*16,c=a+8;for(int k=0;k<3;k++){n[b+k]=std::min(n[a+k],n[c+k]);n[b+4+k]=std::max(n[a+4+k],n[c+4+k]);}n[b+3]=n[a+3]+n[c+3];}return n;}
float delta(float3 a,float3 b){return std::max({std::fabs(a.x-b.x),std::fabs(a.y-b.y),std::fabs(a.z-b.z)});}
void testGrammar(){
 std::vector<float> world(64*64*8),req(256*8),p(2048*16);int queue[2]={1,0};long checked=0;int maxcount=0;
 for(int type=0;type<=4;type++)for(int v=0;v<3;v++){
  Lot l;l.w=11+v;l.d=10.5f+v;l.floors=3+v;l.storey=3.8f;l.h=4.2f+l.floors*l.storey;l.type=type;l.mat=v%2;l.seed=1788+v*317;l.roofScale=1;l.turn=0;l.ox=0;l.oz=0;l.palette=0;
  int b=reference::worldIndex(0,0)*8;world[b]=l.w;world[b+1]=l.d;world[b+2]=l.h;world[b+3]=type;world[b+4]=l.mat;world[b+5]=l.floors;world[b+6]=l.seed;req[2]=5;
  for(int g=0;g<CLUSTERS;g++){
   blockDim={64,1,1};blockIdx={0,0,0};threadIdx={unsigned(g),0,0};reference::generatePages(world.data(),req.data(),queue,p.data());
   auto features=enumerate(l,g);maxcount=std::max(maxcount,int(features.size()));
   for(int j=0;j<32;j++){int q=(g*32+j)*16;if(p[q+3]<0)continue;if(features.size()<=size_t(j))throw std::runtime_error("feature dropped");const auto& f=features[j];
    if(delta(f.p,{p[q],p[q+1],p[q+2]})>.0001f||delta(f.h,{p[q+4],p[q+5],p[q+6]})>.0001f||f.shape!=int(p[q+3])||f.material!=int(p[q+7])||f.turn!=int(p[q+8])||f.seed!=p[q+9])throw std::runtime_error("Original grammar mismatch type="+std::to_string(type)+" g="+std::to_string(g)+" feature="+std::to_string(j));checked++;
   }
  }
 }
 std::cout<<"Original feature records compared: "<<checked<<"; maximum group size: "<<maxcount<<"\n";
 require(checked>5000&&maxcount<=MAX_FEATURES&&maxcount>32,"exact original geometry/material/seed identity and unclipped hyperdetail additions");
}
void testQueries(){
 unsigned origin[4]={0,0,0,0};int rays=0,hits=0;float worst=0;std::set<int> families;
 for(int v=0;v<24;v++){
  Tag tag=offsetTag(origin,v*71-830,v*39-191);Lot l=describeLot(tag,1788);if(v<6)l.type=v;families.insert(l.type);
  auto n=bounds(l);std::vector<Feature> all;for(int g=0;g<CLUSTERS;g++){auto f=enumerate(l,g);all.insert(all.end(),f.begin(),f.end());}
  for(int i=0;i<96;i++){
   float a=hash1(i*111+v)*6.28318f;float3 ro={cosf(a)*27,hash1(i+v*7+199)*65-0.1f,sinf(a)*27};float3 target={hash1(i+17)*18-9,hash1(i+59)*l.h,hash1(i+83)*18-9},rd=norm3(target-ro);float best=FAR;
   for(auto f:all)best=std::min(best,featureHit(f,ro,rd,0));Sink hit=queryLot(l,n.data(),0,ro,rd,FAR,0);worst=std::max(worst,std::fabs(best-hit.t));rays++;if(hit.fid>=0){hits++;auto no=featureNormal(hit.feature,ro+rd*hit.t);if(!std::isfinite(no.x+no.y+no.z))throw std::runtime_error("non-finite normal");}
  }
 }
 std::cout<<"Bounds vs exhaustive queries: "<<rays<<"; hits "<<hits<<"; max distance difference "<<worst<<"\n";
 require(worst<.001f&&hits>1000,"accelerated queries agree with exhaustive exact features across all families");
 // Cylinder caps and complete lower ring are the cases omitted by the broken shared sink.
 Feature f;f.p={0,0,0};f.h={1,2,1};f.shape=2;f.turn=0;f.material=6;f.seed=1;
 require(std::fabs(featureHit(f,{0,5,0},{0,-1,0},0)-3)<.0001f,"cylinder top cap is not lost");
 f.shape=7;f.h={2,2,.25f};require(featureHit(f,{0,-1.9f,4},{0,0,-1},0)<FAR,"full ring retains lower half and front cap");
}
void testOrigin(){
 unsigned o[4]={0,0,0,0};auto a=offsetTag(o,-1,-1);require(a.xl==0xffffffffu&&a.xh==0xffffffffu&&a.zh==0xffffffffu,"64-bit negative coordinate borrow");
 unsigned near[4]={0xfffffff0u,37,9,0};auto b=offsetTag(near,32,0);require(b.xl==16&&b.xh==38,"64-bit positive carry");
 auto l=describeLot(b,1788);unsigned next[4]={b.xl,b.xh,b.zl,b.zh};auto l2=describeLot(offsetTag(next,0,0),1788);
 require(l.w==l2.w&&l.h==l2.h&&l.seed==l2.seed,"revisiting a distant coordinate reconstructs the same lot");
 float c[64]={},input[32]={};unsigned origin[4]={0,0,0,0};scalar();initCamera(c,origin,1788);c[0]=18.25f;c[2]=-18.25f;stepCamera(c,origin,input,0,1600,900);
 require(c[0]<18&&c[0]>=-18&&c[2]>=-18&&c[2]<18&&origin[0]==1&&origin[2]==0xffffffffu,"floating origin rebases without a finite city boundary");
 c[0]=0;c[2]=0;for(int i=0;i<130;i++)stepCamera(c,origin,input,0,1600,900);float3 r=rayDirection(c,92,51,1600,900);for(int i=0;i<120;i++)stepCamera(c,origin,input,0,1600,900);
 require(c[19]==64&&delta(r,rayDirection(c,92,51,1600,900))==0,"stationary 64-sample sequence converges and stops jittering");
 std::set<int> seeds;std::set<int> palette;for(int i=0;i<128;i++){Lot q=describeLot(offsetTag(o,i-64,i*3),1788);seeds.insert(int(q.seed));palette.insert(int(q.palette));}
 require(seeds.size()>120&&palette.size()>3,"seeded architectural and district palette variation");
}
void render(const std::string& path,int width,int height,int samples){
 std::vector<float> c(64),input(32),world(WORLD_LOTS*LOT_FLOATS),nodes(WORLD_LOTS*GROUP_NODES*8),hit(width*height*4),surface(hit.size()),linear(hit.size()),history(hit.size());
 std::vector<unsigned> origin(4),tags(WORLD_LOTS*4),queue(WORLD_LOTS+1),pixels(width*height);
 scalar();initCamera(c.data(),origin.data(),1788);stepCamera(c.data(),origin.data(),input.data(),0,width,height);clearQueue(queue.data());dispatch(WORLD_LOTS,[&]{prepareLots(origin.data(),c.data(),world.data(),tags.data(),queue.data());});
 std::cout<<"Building bound-only acceleration for "<<queue[0]<<" lots.\n";
 #pragma omp parallel for schedule(dynamic,8)
 for(int slot=0;slot<WORLD_LOTS;slot++){auto l=readLot(world.data(),slot);auto n=bounds(l);std::copy(n.begin(),n.end(),nodes.begin()+slot*GROUP_NODES*8);}
 for(int sample=0;sample<samples;sample++){
  c[19]=float(sample+1);
  #pragma omp parallel for schedule(dynamic,4)
  for(int y=0;y<height;y++)for(int x=0;x<width;x++){blockDim={8,8,1};blockIdx={unsigned(x/8),unsigned(y/8),0};threadIdx={unsigned(x%8),unsigned(y%8),0};tracePrimary(world.data(),nodes.data(),origin.data(),c.data(),hit.data(),surface.data(),width,height,0,height);}
  #pragma omp parallel for schedule(dynamic,4)
  for(int y=0;y<height;y++)for(int x=0;x<width;x++){blockDim={8,8,1};blockIdx={unsigned(x/8),unsigned(y/8),0};threadIdx={unsigned(x%8),unsigned(y%8),0};shadePixels(world.data(),origin.data(),c.data(),hit.data(),surface.data(),linear.data(),width,height,0,height);}
  #pragma omp parallel for schedule(dynamic,4)
  for(int y=0;y<height;y++)for(int x=0;x<width;x++){blockDim={8,8,1};blockIdx={unsigned(x/8),unsigned(y/8),0};threadIdx={unsigned(x%8),unsigned(y%8),0};resolveFrame(linear.data(),history.data(),pixels.data(),c.data(),hit.data(),surface.data(),width,height);}
  std::cout<<"Rendered sample "<<sample+1<<"\n";
 }
 require(std::all_of(linear.begin(),linear.end(),[](float v){return std::isfinite(v);}),"finite full scene lighting");
 std::ofstream f(path,std::ios::binary);f<<"P6\n"<<width<<" "<<height<<"\n255\n";for(auto p:pixels){char rgb[3]={char(p&255u),char((p>>8)&255u),char((p>>16)&255u)};f.write(rgb,3);}
 std::cout<<"Saved CPU reference "<<path<<"\n";
}
void fixtures(){
 std::cout<<"[";bool first=true;
 for(int type=0;type<6;type++)for(int g: {0,1,6,31,32,56,60,62,63}){
  float input[17]={float(g),1,12,11.5f,23.2f,float(type),1,5,1788,3.8f,1,0,55,0,0,-1,0};float out[16]={};scalar();probeGrammar(input,out);
  if(!first)std::cout<<",";first=false;std::cout<<"{\"input\":[";for(int i=0;i<17;i++){if(i)std::cout<<",";std::cout<<input[i];}std::cout<<"],\"expected\":[";for(int i=0;i<16;i++){if(i)std::cout<<",";std::cout<<out[i];}std::cout<<"]}";
 }std::cout<<"]";
}
int main(int argc,char**argv){try{
 if(argc>1&&std::string(argv[1])=="fixtures"){fixtures();return 0;}
 if(argc>1&&std::string(argv[1])=="render"){render(argc>2?argv[2]:"scene.ppm",argc>3?std::stoi(argv[3]):480,argc>4?std::stoi(argv[4]):300,argc>5?std::stoi(argv[5]):4);return 0;}
 testGrammar();testQueries();testOrigin();std::cout<<"ALL NATIVE CHECKS PASSED\n";return 0;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<"\n";return 1;}}
