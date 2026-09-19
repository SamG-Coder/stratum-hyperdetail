// STRATUM — compute-streamed geometry. Distances are metres.
// Supported CUDA C subset; this file is also compiled by the native test harness.
#define CITY 64
#define CELL 36.0f
#define HALF_CITY 1152.0f
#define CACHE_SIDE 16
#define PAGES 256
#define CLUSTERS 64
#define PER_CLUSTER 32
#define PRIMS 2048
#define NODES 4096
#define PS 16
#define MS 12
#define REQUESTS 8
#define GENERATE_BUDGET 6
#define NN_INPUTS 8
#define NN_HIDDEN 24
#define NN_PARAMS 291
#define NN_BATCH 64
#define NN_WS 64
#define PI 3.141592653589793f
#define FAR 30000.0f
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
__device__ float3 rayDirection(const float* C,int x,int y,int width,int height){
 // Low-discrepancy sub-pixel sequence. Avoid the old two irrational multiples of frame
 // number, whose correlated motion was visible as diagonal crawling/jitter in fine facades.
 float jitterX=0.0f;float jitterY=0.0f;
 if(C[15]<0.5f){int fi=(int)C[6];unsigned int hx=hashU((unsigned int)(fi*2+1));unsigned int hy=hashU((unsigned int)(fi*2+2));
  jitterX=((float)(hx&65535u)/65536.0f)-0.5f;jitterY=((float)(hy&65535u)/65536.0f)-0.5f;}
 float sx=((float)x+0.5f+jitterX-(float)width*0.5f)/(float)height*1.08f;
 float sy=-((float)y+0.5f+jitterY-(float)height*0.5f)/(float)height*1.08f;
 float3 f=cameraForward(C);float3 r=cameraRight(C);float3 u=cross3(f,r);
 return norm3(f+r*sx+u*sy);
}
__device__ float safeInv(float d){return 1.0f/(fabsf(d)>0.0000001f?d:(d<0.0f?-0.0000001f:0.0000001f));}
__device__ float2 boxRange(float3 ro,float3 rd,float3 lo,float3 hi){
 float3 a=make_float3((lo.x-ro.x)*safeInv(rd.x),(lo.y-ro.y)*safeInv(rd.y),(lo.z-ro.z)*safeInv(rd.z));
 float3 b=make_float3((hi.x-ro.x)*safeInv(rd.x),(hi.y-ro.y)*safeInv(rd.y),(hi.z-ro.z)*safeInv(rd.z));
 float3 mn=min3(a,b);float3 mx=max3(a,b);
 return make_float2(fmaxf(fmaxf(mn.x,mn.y),mn.z),fminf(fminf(mx.x,mx.y),mx.z));
}
// Generic analytic feature intersection used by BOTH the page writer and direct authored
// ray-query sink. This is intentionally independent of cache/page storage.
__device__ float featureHit(float3 ro,float3 rd,float3 cp,float3 h,int shape,int turn,float best){
 float3 p=ro-cp,d=rd;if(turn%2==1){p=make_float3(p.z,p.y,-p.x);d=make_float3(d.z,d.y,-d.x);}
 float3 lo=h*-1.0f,hi=h;if(shape==3||shape==4||shape==7)lo=make_float3(-h.x,0.0f,-h.z);
 float2 range=boxRange(p,d,lo,hi);if(range.y<fmaxf(0.001f,range.x))return best;float t=range.x>0.001f?range.x:range.y;
 if(shape==1){float3 a=make_float3(p.x/h.x,p.y/h.y,p.z/h.z),v=make_float3(d.x/h.x,d.y/h.y,d.z/h.z);float aa=dot3(v,v),bb=dot3(a,v),cc=dot3(a,a)-1.0f,disc=bb*bb-aa*cc;if(disc<0.0f)return best;t=(-bb-sqrtf(disc))/aa;if(t<=0.001f)t=(-bb+sqrtf(disc))/aa;}
 if(shape==2){float aa=d.x*d.x/(h.x*h.x)+d.z*d.z/(h.z*h.z),bb=p.x*d.x/(h.x*h.x)+p.z*d.z/(h.z*h.z),cc=p.x*p.x/(h.x*h.x)+p.z*p.z/(h.z*h.z)-1.0f,q=FAR,disc=bb*bb-aa*cc;
  if(disc>=0.0f&&aa>0.0000001f){float r=sqrtf(disc),u=(-bb-r)/aa,v=(-bb+r)/aa;if(u>0.001f&&fabsf(p.y+d.y*u)<=h.y)q=u;if(v>0.001f&&fabsf(p.y+d.y*v)<=h.y)q=fminf(q,v);}t=q;}
 if(shape==3){float near=fmaxf(0.001f,range.x),far=range.y;for(int k=0;k<2;k++){float sign=k==0?1.0f:-1.0f,numer=h.y-(p.y+sign*p.x*h.y/h.x),denom=d.y+sign*d.x*h.y/h.x;if(fabsf(denom)<0.000001f){if(numer<0.0f)return best;}else{float u=numer/denom;if(denom>0.0f)far=fminf(far,u);else near=fmaxf(near,u);}}if(far<near)return best;t=near;}
 if(shape==4||shape==7){float q=FAR,aa=d.x*d.x/(h.x*h.x)+d.y*d.y/(h.y*h.y),bb=p.x*d.x/(h.x*h.x)+p.y*d.y/(h.y*h.y);
  for(int ring=0;ring<2;ring++){float rad=ring==0?1.0f:0.77f,cc=p.x*p.x/(h.x*h.x)+p.y*p.y/(h.y*h.y)-rad*rad,disc=bb*bb-aa*cc;if(disc>=0.0f&&aa>0.000001f){float r=sqrtf(disc);for(int k=0;k<2;k++){float u=(-bb+(k==0?-r:r))/aa;if(u>0.001f&&fabsf(p.z+d.z*u)<=h.z&&(shape==7||p.y+d.y*u>=0.0f))q=fminf(q,u);}}}t=q;}
 return t>0.001f&&t<best?t:best;
}
__device__ int worldIndex(int cx,int cz){return (cz+CITY/2)*CITY+cx+CITY/2;}
__device__ int pageIndex(int cx,int cz){return imod(cz,CACHE_SIDE)*CACHE_SIDE+imod(cx,CACHE_SIDE);}
__device__ bool inCity(int cx,int cz){return cx>=-CITY/2&&cx<CITY/2&&cz>=-CITY/2&&cz<CITY/2;}
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
