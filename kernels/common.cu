// STRATUM Infinite: one authored feature grammar, bounded acceleration, no proxy models.
#define CELL 36.0f
#define WORLD_SIDE 128
#define WORLD_LOTS 16384
#define LOT_FLOATS 16
#define CLUSTERS 64
#define GROUP_NODES 128
#define MAX_FEATURES 64
#define BUILD_CHUNK 2048
#define BUILD_CHUNKS 8
#define FAR 2000.0f
#define SCENE_TOP 100.0f
#define PI 3.141592653589793f
#define NN_INPUTS 8
#define NN_HIDDEN 24
#define NN_PARAMS 291
struct Lot {
 float w; float d; float h; int type; int mat; int floors; float seed;
 float storey; float roofScale; int turn; float ox; float oz; float palette;
};
struct Tag { unsigned int xl; unsigned int xh; unsigned int zl; unsigned int zh; };
struct Feature { float3 p; float3 h; int shape; int material; int turn; float seed; };
struct Sink {
 int mode; int count; int target; float t; float3 ro; float3 rd;
 float3 lo; float3 hi; Feature feature; int fid; float pixelCone; float roofBase; float roofScale;
};
__device__ float clampf(float x,float a,float b){return fminf(b,fmaxf(a,x));}
__device__ float sat(float x){return clampf(x,0.0f,1.0f);}
__device__ float lerpf(float a,float b,float t){return a+(b-a)*t;}
__device__ float fractf(float x){return x-floorf(x);}
__device__ float smoothf(float a,float b,float x){float t=sat((x-a)/(b-a));return t*t*(3.0f-2.0f*t);}
__device__ int imod(int x,int n){int r=x%n;return r<0?r+n:r;}
__device__ float dot3(float3 a,float3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
__device__ float length3(float3 a){return sqrtf(dot3(a,a));}
__device__ float3 norm3(float3 a){return a/fmaxf(0.000001f,length3(a));}
__device__ float3 cross3(float3 a,float3 b){return make_float3(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);}
__device__ float3 mix3(float3 a,float3 b,float t){return a+(b-a)*t;}
__device__ float3 min3(float3 a,float3 b){return make_float3(fminf(a.x,b.x),fminf(a.y,b.y),fminf(a.z,b.z));}
__device__ float3 max3(float3 a,float3 b){return make_float3(fmaxf(a.x,b.x),fmaxf(a.y,b.y),fmaxf(a.z,b.z));}
__device__ float3 abs3(float3 a){return make_float3(fabsf(a.x),fabsf(a.y),fabsf(a.z));}
// Integer hashing controls topology. It does not depend on vendor-specific sin precision.
__device__ unsigned int hashU(unsigned int x){x^=x>>16;x*=2146121005u;x^=x>>15;x*=2221713035u;x^=x>>16;return x;}
__device__ float hash1(int x){return (float)(hashU((unsigned int)x)&16777215u)/16777216.0f;}
__device__ float hash2(int x,int y,int seed){return (float)(hashU((unsigned int)x*1973u+(unsigned int)y*9277u+(unsigned int)seed*26699u)&16777215u)/16777216.0f;}
__device__ float noise2(float x,float y){
 int ix=(int)floorf(x);int iy=(int)floorf(y);float fx=fractf(x);float fy=fractf(y);
 fx=fx*fx*(3.0f-2.0f*fx);fy=fy*fy*(3.0f-2.0f*fy);
 return lerpf(lerpf(hash2(ix,iy,13),hash2(ix+1,iy,13),fx),lerpf(hash2(ix,iy+1,13),hash2(ix+1,iy+1,13),fx),fy);
}
__device__ float fbm2(float x,float y){
 float v=0.0f;float a=0.5f;for(int i=0;i<5;i++){v+=noise2(x,y)*a;x=x*2.07f+11.7f;y=y*2.03f-8.3f;a*=0.5f;}return v;
}
__device__ float frequencyWeight(float footprint,float freq){return 1.0f-smoothf(0.35f,1.1f,footprint*freq);}
__device__ float3 cameraForward(const float* C){return make_float3(sinf(C[3])*cosf(C[4]),sinf(C[4]),cosf(C[3])*cosf(C[4]));}
__device__ float3 cameraRight(const float* C){return make_float3(cosf(C[3]),0.0f,-sinf(C[3]));}
__device__ float3 cameraPosition(const float* C){return make_float3(C[0],C[1],C[2]);}
__device__ float3 sunDirection(const float* C){return norm3(make_float3(cosf(C[7])*cosf(C[8]),sinf(C[8]),sinf(C[7])*cosf(C[8])));}
__device__ float radicalInverse(int value,int base){
 float inv=1.0f/(float)base,place=inv,result=0.0f;
 for(int j=0;j<12;j++){if(value<=0)break;result+=(float)(value%base)*place;value/=base;place*=inv;}
 return result;
}
__device__ float3 rayDirection(const float* C,int x,int y,int width,int height){
 int sample=(int)C[19]-1;float jx=0.0f,jy=0.0f;
 if(sample>0){jx=radicalInverse(sample,2)-0.5f;jy=radicalInverse(sample,3)-0.5f;}
 float sx=((float)x+0.5f+jx-(float)width*0.5f)/(float)height*1.08f;
 float sy=-((float)y+0.5f+jy-(float)height*0.5f)/(float)height*1.08f;
 float3 f=cameraForward(C),r=cameraRight(C),u=cross3(f,r);
 return norm3(f+r*sx+u*sy);
}
__device__ float safeInv(float d){return 1.0f/(fabsf(d)>0.0000001f?d:(d<0.0f?-0.0000001f:0.0000001f));}
__device__ float2 boxRange(float3 ro,float3 rd,float3 lo,float3 hi){
 float3 a=make_float3((lo.x-ro.x)*safeInv(rd.x),(lo.y-ro.y)*safeInv(rd.y),(lo.z-ro.z)*safeInv(rd.z));
 float3 b=make_float3((hi.x-ro.x)*safeInv(rd.x),(hi.y-ro.y)*safeInv(rd.y),(hi.z-ro.z)*safeInv(rd.z));
 float3 mn=min3(a,b);float3 mx=max3(a,b);
 return make_float2(fmaxf(fmaxf(mn.x,mn.y),mn.z),fminf(fminf(mx.x,mx.y),mx.z));
}
__device__ float2 masonryUV(float3 p,float3 n){return fabsf(n.y)>0.5f?make_float2(p.x,p.z):(fabsf(n.x)>0.5f?make_float2(p.z,p.y):make_float2(p.x,p.y));}
// Bounded geometric relief. Fine pores are shading detail; masonry joints are actual relief.
__device__ float masonryHeight(float u,float v,int material){
 float bw=material==1?0.46f:0.92f;float bh=material==1?0.215f:0.46f;
 int row=(int)floorf(v/bh);float x=fractf(u/bw+(float)imod(row,2)*0.5f);float y=fractf(v/bh);
 float edge=fminf(fminf(x,1.0f-x)*bw,fminf(y,1.0f-y)*bh);
 float grain=noise2(u*24.0f,v*24.0f);
 float chippedEdge=edge-0.006f*smoothf(0.34f,0.73f,noise2(u*117.0f,v*117.0f));
 return -0.018f*(1.0f-smoothf(0.008f,0.023f,chippedEdge))+0.0025f*grain;
}
__device__ unsigned int spread10(unsigned int x){x&=1023u;x=(x|(x<<16))&0x030000FFu;x=(x|(x<<8))&0x0300F00Fu;x=(x|(x<<4))&0x030C30C3u;x=(x|(x<<2))&0x09249249u;return x;}
__device__ unsigned int morton3(float x,float y,float z){
 unsigned int a=(unsigned int)(sat(x)*1023.0f);unsigned int b=(unsigned int)(sat(y)*1023.0f);unsigned int c=(unsigned int)(sat(z)*1023.0f);
 return spread10(a)|(spread10(b)<<1)|(spread10(c)<<2);
}
__device__ unsigned int packRGBA(float3 c){unsigned int r=(unsigned int)(sat(c.x)*255.0f);unsigned int g=(unsigned int)(sat(c.y)*255.0f);unsigned int b=(unsigned int)(sat(c.z)*255.0f);return r|(g<<8)|(b<<16)|4278190080u;}
