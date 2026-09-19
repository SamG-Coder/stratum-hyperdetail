// Endless deterministic world addressing. Two uint32 words per axis hold a 64-bit cell.
// The camera stays near zero; geometry and materials are evaluated in lot-local metres.
__device__ Tag offsetTag(const unsigned int* Origin,int dx,int dz){
 Tag t;t.xl=Origin[0]+(unsigned int)dx;t.xh=Origin[1];
 if(dx>=0&&t.xl<Origin[0])t.xh++;if(dx<0&&t.xl>Origin[0])t.xh--;
 t.zl=Origin[2]+(unsigned int)dz;t.zh=Origin[3];
 if(dz>=0&&t.zl<Origin[2])t.zh++;if(dz<0&&t.zl>Origin[2])t.zh--;
 return t;
}
__device__ unsigned int tagHash(Tag t,int seed){
 return hashU(t.xl*1973u+hashU(t.xh+719u)+t.zl*9277u+hashU(t.zh+173u)+(unsigned int)seed*26699u);
}
__device__ Lot describeLot(Tag t,int seed){
 unsigned int key=tagHash(t,seed);Tag district;
 district.xl=(t.xl>>3)|(t.xh<<29);district.xh=t.xh>>3;
 district.zl=(t.zl>>3)|(t.zh<<29);district.zh=t.zh>>3;
 unsigned int dk=tagHash(district,seed+47);float r=(float)(key&65535u)/65536.0f;
 float density=(float)(dk&65535u)/65536.0f;
 Lot l;l.type=r>0.94f?1:0;if(r<0.10f+density*0.045f)l.type=3;
 // A sparse connected canal network with bridges at regular crossing streets.
 if((t.xl&63u)==28u)l.type=(t.zl&3u)==0u?4:5;
 if((key&255u)==93u)l.type=2;
 l.floors=2+(int)((hashU(key+11u)&65535u)*(1.0f/65536.0f)*5.0f);
 if(density>0.7f&&l.floors<4)l.floors=4;
 l.storey=3.2f+1.15f*(float)(hashU(key+127u)&65535u)/65536.0f;
 l.w=9.4f+4.3f*(float)(hashU(key+31u)&65535u)/65536.0f;
 l.d=9.0f+4.7f*(float)(hashU(key+83u)&65535u)/65536.0f;
 l.h=4.2f+(float)l.floors*l.storey;if(l.type==2){l.floors=6;l.h=4.2f+6.0f*l.storey;}
 l.mat=(int)((key>>4)&1u);l.seed=(float)(hashU(key+271u)&65535u);
 l.turn=(int)((key>>6)&3u);l.ox=((float)(hashU(key+311u)&65535u)/65536.0f-0.5f)*0.48f;
 l.oz=((float)(hashU(key+313u)&65535u)/65536.0f-0.5f)*0.48f;
 if(l.type>=3){l.ox=0.0f;l.oz=0.0f;l.turn=0;}
 l.roofScale=0.85f+0.4f*(float)(hashU(key+317u)&65535u)/65536.0f;
 l.palette=(float)((dk>>18)&7u);
 return l;
}
__device__ Lot readLot(const float* World,int slot){
 int b=slot*LOT_FLOATS;Lot l;l.w=World[b];l.d=World[b+1];l.h=World[b+2];l.type=(int)World[b+3];
 l.mat=(int)World[b+4];l.floors=(int)World[b+5];l.seed=World[b+6];l.storey=World[b+7];
 l.roofScale=World[b+8];l.turn=(int)World[b+9];l.ox=World[b+10];l.oz=World[b+11];l.palette=World[b+12];return l;
}
__device__ int slotFor(const unsigned int* Origin,int dx,int dz){
 int x=(int)((Origin[0]+(unsigned int)dx)&127u),z=(int)((Origin[2]+(unsigned int)dz)&127u);
 return z*WORLD_SIDE+x;
}
__device__ float3 localCellFor(const unsigned int* Origin,int slot){
 int x=imod(slot%WORLD_SIDE-(int)(Origin[0]&127u)+WORLD_SIDE/2,WORLD_SIDE)-WORLD_SIDE/2;
 int z=imod(slot/WORLD_SIDE-(int)(Origin[2]&127u)+WORLD_SIDE/2,WORLD_SIDE)-WORLD_SIDE/2;
 return make_float3((float)x*CELL,0.0f,(float)z*CELL);
}
__device__ float3 turnLocal(float3 p,int turn){
 if(turn==1)return make_float3(p.z,p.y,-p.x);
 if(turn==2)return make_float3(-p.x,p.y,-p.z);
 if(turn==3)return make_float3(-p.z,p.y,p.x);return p;
}
__device__ float3 turnWorld(float3 p,int turn){return turnLocal(p,(4-turn)%4);}
__global__ void initCamera(float* C,unsigned int* Origin,int seed){
 if(blockIdx.x!=0||threadIdx.x!=0)return;
 for(int i=0;i<64;i++)C[i]=0.0f;for(int i=0;i<4;i++)Origin[i]=0u;
 C[0]=112.0f;C[1]=76.0f;C[2]=155.0f;C[3]=-2.53f;C[4]=-0.30f;
 C[7]=-0.55f;C[8]=0.45f;C[9]=0.93f;C[10]=0.28f;C[12]=38.0f;
 C[14]=(float)seed;C[15]=1.0f;C[19]=1.0f;C[30]=1.0f;
}
__global__ void stepCamera(float* C,unsigned int* Origin,const float* I,float dt,int width,int height){
 if(blockIdx.x!=0||threadIdx.x!=0)return;
 C[5]+=dt;C[6]+=1.0f;C[13]=(float)height;C[20]=(float)width/(float)height;C[15]=I[23]>0.5f?1.0f:0.0f;
 int bm=(int)I[8];
 if(bm>0){
  for(int i=0;i<4;i++)Origin[i]=0u;
  if(bm==1){C[0]=112.0f;C[1]=76.0f;C[2]=155.0f;C[3]=-2.53f;C[4]=-0.30f;C[12]=38.0f;}
  if(bm==2){C[0]=18.0f;C[1]=1.75f;C[2]=49.0f;C[3]=-2.80f;C[4]=0.17f;C[12]=6.0f;}
  if(bm==3||bm==6){Tag tag=offsetTag(Origin,0,0);Lot lot=describeLot(tag,(int)C[14]);C[0]=17.0f;C[1]=6.9f;C[2]=0.0f;C[3]=-PI*0.5f;C[4]=0.015f;C[12]=0.8f;}
  if(bm==4){C[0]=400.0f;C[1]=650.0f;C[2]=750.0f;C[3]=-2.61f;C[4]=-0.66f;C[12]=180.0f;}
  if(bm==5){C[0]=28.0f*CELL+20.0f;C[1]=8.0f;C[2]=75.0f;C[3]=-2.65f;C[4]=-0.04f;C[12]=12.0f;}
  C[15]=1.0f;
 }
 if(I[3]!=0.0f||I[4]!=0.0f){C[3]+=I[3]*0.0022f;C[4]=clampf(C[4]-I[4]*0.0022f,-1.53f,1.53f);C[15]=1.0f;}
 if(I[7]!=0.0f)C[12]=clampf(C[12]*expf(-I[7]*0.0015f),0.03f,800.0f);
 float speed=C[12]*(I[5]>0.5f?3.0f:1.0f)*(I[6]>0.5f?0.15f:1.0f);
 float3 move=cameraForward(C)*I[0]+cameraRight(C)*I[1]+make_float3(0.0f,I[2],0.0f);
 if(dot3(move,move)>0.0f){move=norm3(move)*speed*dt;C[0]+=move.x;C[1]=clampf(C[1]+move.y,0.08f,1500.0f);C[2]+=move.z;C[15]=1.0f;}
 if(I[9]!=0.0f){C[7]+=I[9]*dt*0.35f;C[15]=1.0f;}
 if(I[10]>0.0f){C[11]=(float)(((int)C[11]+1)%4);C[15]=1.0f;}
 if(I[13]!=0.0f){C[9]=clampf(C[9]+I[13]*dt,0.25f,3.0f);C[15]=1.0f;}
 if(I[14]>0.0f){C[18]=1.0f-C[18];C[15]=1.0f;}
 if(I[16]>0.0f)C[22]=1.0f-C[22];
 if(C[22]>0.5f){C[3]+=dt*0.032f;float3 f=cameraForward(C);C[0]+=f.x*dt*14.0f;C[2]+=f.z*dt*14.0f;C[15]=1.0f;}
 int dx=(int)floorf((C[0]+18.0f)/CELL),dz=(int)floorf((C[2]+18.0f)/CELL);
 if(dx!=0||dz!=0){Tag t=offsetTag(Origin,dx,dz);Origin[0]=t.xl;Origin[1]=t.xh;Origin[2]=t.zl;Origin[3]=t.zh;C[0]-=(float)dx*CELL;C[2]-=(float)dz*CELL;}
 // A deterministic finite accumulation converges, then stops changing. No hashed jitter crawl.
 C[19]=C[15]>0.5f?1.0f:fminf(64.0f,C[19]+1.0f);
}
__global__ void clearQueue(unsigned int* Queue){if(blockIdx.x==0&&threadIdx.x==0)Queue[0]=0u;}
__global__ void prepareLots(const unsigned int* Origin,const float* C,float* World,unsigned int* Tags,unsigned int* Queue){
 int slot=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(slot>=WORLD_LOTS)return;
 float3 cell=localCellFor(Origin,slot);Tag t=offsetTag(Origin,(int)(cell.x/CELL),(int)(cell.z/CELL));
 int b=slot*LOT_FLOATS,k=slot*4;
 bool same=World[b+15]>0.5f&&Tags[k]==t.xl&&Tags[k+1]==t.xh&&Tags[k+2]==t.zl&&Tags[k+3]==t.zh;
 if(same)return;
 Lot l=describeLot(t,(int)C[14]);World[b]=l.w;World[b+1]=l.d;World[b+2]=l.h;World[b+3]=(float)l.type;
 World[b+4]=(float)l.mat;World[b+5]=(float)l.floors;World[b+6]=l.seed;World[b+7]=l.storey;World[b+8]=l.roofScale;
 World[b+9]=(float)l.turn;World[b+10]=l.ox;World[b+11]=l.oz;World[b+12]=l.palette;World[b+15]=1.0f;
 Tags[k]=t.xl;Tags[k+1]=t.xh;Tags[k+2]=t.zl;Tags[k+3]=t.zh;
 unsigned int index=atomicAdd(&Queue[0],1u);Queue[index+1u]=(unsigned int)slot;
}
__global__ void planBounds(const unsigned int* Queue,unsigned int* Args){
 int chunk=(int)threadIdx.x;if(chunk>=BUILD_CHUNKS)return;
 int count=(int)Queue[0]-chunk*BUILD_CHUNK;count=count<0?0:(count>BUILD_CHUNK?BUILD_CHUNK:count);
 int b=chunk*21;Args[b]=(unsigned int)count;Args[b+1]=1u;Args[b+2]=1u;
 int level=1;for(int first=32;first>=1;first/=2){Args[b+level*3]=(unsigned int)((count*first+63)/64);Args[b+level*3+1]=1u;Args[b+level*3+2]=1u;level++;}
}
