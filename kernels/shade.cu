// Lighting remains procedural: exact feature materials/normals, building-mass sunlight,
// sky reflection and contact shading. This is not path tracing or neural rendering.
__device__ float buildingShadow(const float* World,const unsigned int* Origin,float3 p,float3 light,int worldSeed){
 if(light.y<0.005f)return 0.05f;
 float limit=fminf(1500.0f,fmaxf(0.0f,(SCENE_TOP-p.y)/light.y));
 float3 ro=p+light*0.08f;int cx=(int)floorf((ro.x+18.0f)/CELL),cz=(int)floorf((ro.z+18.0f)/CELL);
 int sx=light.x>0.0f?1:-1,sz=light.z>0.0f?1:-1;
 float tx=((float)cx*CELL+(sx>0?18.0f:-18.0f)-ro.x)*safeInv(light.x),tz=((float)cz*CELL+(sz>0?18.0f:-18.0f)-ro.z)*safeInv(light.z);
 float dx=CELL*fabsf(safeInv(light.x)),dz=CELL*fabsf(safeInv(light.z)),t=0.0f;
 for(int step=0;step<64;step++){
  if(t>limit)break;Lot l;
  if(cx>=-64&&cx<64&&cz>=-64&&cz<64)l=readLot(World,slotFor(Origin,cx,cz));else l=describeLot(offsetTag(Origin,cx,cz),worldSeed);
  float3 cp=make_float3((float)cx*CELL+l.ox,0.0f,(float)cz*CELL+l.oz);
  if(l.type<3){
   float3 lr=turnLocal(ro-cp,l.turn),ld=turnLocal(light,l.turn);
   float2 span=boxRange(lr,ld,make_float3(-l.w+0.05f,4.1f,-l.d+0.05f),make_float3(l.w-0.05f,l.h,l.d-0.05f));
   if(span.y>fmaxf(0.06f,span.x)&&span.x<limit)return 0.06f;
  }
  if(tx<tz){t=tx;tx+=dx;cx+=sx;}else{t=tz;tz+=dz;cz+=sz;}
 }
 return 1.0f;
}
__device__ float ambientVisibility(const float* World,const unsigned int* Origin,float3 p){
 int cx=(int)floorf((p.x+18.0f)/CELL),cz=(int)floorf((p.z+18.0f)/CELL);float occ=0.0f;
 for(int z=-1;z<=1;z++)for(int x=-1;x<=1;x++){
  Lot l=readLot(World,slotFor(Origin,cx+x,cz+z));if(l.type>=3)continue;
  float3 q=turnLocal(p-make_float3((float)(cx+x)*CELL+l.ox,0.0f,(float)(cz+z)*CELL+l.oz),l.turn);
  float xx=fmaxf(0.0f,fabsf(q.x)-l.w),zz=fmaxf(0.0f,fabsf(q.z)-l.d),dist=sqrtf(xx*xx+zz*zz);
  if(dist>0.06f)occ+=sat((l.h-p.y)/(dist+l.h+1.0f))*expf(-dist*0.12f)*0.48f;
 }
 return clampf(1.0f-occ,0.35f,1.0f);
}
__device__ float3 paletteTint(float palette,int mat){
 if(mat!=0&&mat!=1&&mat!=13)return make_float3(1.0f,1.0f,1.0f);
 int p=(int)palette;
 if(p==0)return make_float3(1.04f,0.95f,0.85f);
 if(p==1)return make_float3(0.80f,0.89f,1.02f);
 if(p==2)return make_float3(1.12f,0.89f,0.83f);
 if(p==3)return make_float3(1.13f,1.12f,1.04f);
 if(p==4)return make_float3(0.83f,0.84f,0.79f);
 if(p==5)return make_float3(0.91f,1.02f,0.91f);
 if(p==6)return make_float3(1.03f,0.95f,1.02f);
 return make_float3(1.0f,1.0f,1.0f);
}
__global__ void shadePixels(const float* World,const unsigned int* Origin,const float* C,const float* Hit,const float* Surface,float* Linear,int width,int height,int rowStart,int rowCount){
 int x=(int)(blockIdx.x*blockDim.x+threadIdx.x),y=rowStart+(int)(blockIdx.y*blockDim.y+threadIdx.y);
 if(x>=width||y>=height||y>=rowStart+rowCount)return;int b=(y*width+x)*4;
 float3 rd=rayDirection(C,x,y,width,height),ro=cameraPosition(C),sun=sunDirection(C);float t=Hit[b];int id=(int)Hit[b+1];
 float3 result=sky(rd,sun);
 if(id!=-10000){
  float3 p=ro+rd*t,n=make_float3(Surface[b],Surface[b+1],Surface[b+2]);
  int mat=(int)Hit[b+2],slot=(int)Surface[b+3];float seed=Hit[b+3];Lot lot;
  float3 q=p,qn=n;float palette=0.0f;
  if(slot>=0){lot=readLot(World,slot);float3 cp=localCellFor(Origin,slot)+make_float3(lot.ox,0.0f,lot.oz);q=turnLocal(p-cp,lot.turn);qn=turnLocal(n,lot.turn);palette=lot.palette;}
  else{
   int cx=(int)floorf((p.x+18.0f)/CELL),cz=(int)floorf((p.z+18.0f)/CELL);slot=slotFor(Origin,cx,cz);lot=readLot(World,slot);
   q=p-make_float3((float)cx*CELL,0.0f,(float)cz*CELL);seed=lot.seed;
   mat=fabsf(q.x)<16.5f&&fabsf(q.z)<16.5f?11:10;if(lot.type==5)mat=9;
  }
  if(mat==7&&dot3(n,rd)>0.0f){n=n*-1.0f;qn=qn*-1.0f;}
  float footprint=fmaxf(0.00005f,t*1.08f/(float)height);
  // Grazing surfaces cover a larger area than a normal-facing surface.
  float surfaceFootprint=footprint/fmaxf(0.18f,fabsf(dot3(n,rd)));
  float2 uv=masonryUV(q,qn);float3 base=substrateMean(mat);
  if(mat==0||mat==1)base=substrate(uv.x,uv.y,seed,mat);
  float3 albedo=surfaceColor(q,qn,mat,seed,surfaceFootprint,base)*paletteTint(palette,mat);
  float3 originalN=n;
  if((mat==0||mat==1)&&fabsf(qn.y)<0.5f){
   float eps=fmaxf(0.002f,surfaceFootprint*0.6f);
   float du=(masonryHeight(uv.x+eps,uv.y,mat)-masonryHeight(uv.x-eps,uv.y,mat))/(2.0f*eps);
   float dv=(masonryHeight(uv.x,uv.y+eps,mat)-masonryHeight(uv.x,uv.y-eps,mat))/(2.0f*eps);
   if(surfaceFootprint<0.005f){float me=fmaxf(0.00016f,surfaceFootprint*0.8f);du+=(microRelief(uv.x+me,uv.y,surfaceFootprint)-microRelief(uv.x-me,uv.y,surfaceFootprint))/(2.0f*me);dv+=(microRelief(uv.x,uv.y+me,surfaceFootprint)-microRelief(uv.x,uv.y-me,surfaceFootprint))/(2.0f*me);}
   float3 tangent=fabsf(qn.x)>0.5f?make_float3(0.0f,0.0f,1.0f):make_float3(1.0f,0.0f,0.0f);
   float fade=frequencyWeight(surfaceFootprint,5.0f);float3 localN=norm3(qn-tangent*(du*fade)-make_float3(0.0f,dv*fade,0.0f));n=turnWorld(localN,lot.turn);
  }
  if(mat==9)n=norm3(n+make_float3(cosf(q.x*3.7f+q.z*1.1f)*0.06f,0.0f,sinf(q.z*4.4f-q.x*0.9f)*0.08f));
  float nl=fmaxf(0.0f,dot3(n,sun)),nv=fmaxf(0.03f,-dot3(n,rd));
  float shadow=C[18]>0.5f?1.0f:buildingShadow(World,Origin,p+originalN*0.035f,sun,(int)C[14]);
  float ao=ambientVisibility(World,Origin,p),rough=0.75f;
  if(mat==3||mat==6)rough=0.38f;if(mat==4)rough=0.18f;if(mat==9)rough=0.12f;if(mat==8)rough=0.3f;
  float3 halfv=norm3(sun-rd);float nh=fmaxf(0.0f,dot3(n,halfv)),vh=fmaxf(0.0f,-dot3(rd,halfv));
  float a=rough*rough,denom=nh*nh*(a*a-1.0f)+1.0f,distribution=a*a/(PI*denom*denom+0.000001f);
  float k=(rough+1.0f)*(rough+1.0f)/8.0f,geometry=nv/(nv*(1.0f-k)+k)*nl/(nl*(1.0f-k)+k+0.00001f);
  float f0=mat==3||mat==6||mat==8?0.35f:0.045f,fresnel=f0+(1.0f-f0)*powf(1.0f-vh,5.0f);
  float spec=distribution*geometry*fresnel/(4.0f*nv*nl+0.0001f);
  float3 direct=make_float3(2.25f,1.72f,1.18f)*(nl*shadow);
  float skyWeight=0.22f+0.29f*sat(n.y*0.5f+0.5f);float3 ambient=make_float3(0.25f,0.34f,0.44f)*(skyWeight*ao);
  ambient=ambient+make_float3(0.65f,0.43f,0.25f)*((1.0f-sat(n.y))*0.10f*ao);
  result=albedo*(direct+ambient)+make_float3(1.0f,0.87f,0.63f)*(spec*nl*shadow*2.1f);
  if(mat==4||mat==9){float3 reflection=rd-n*(2.0f*dot3(rd,n));float fr=0.10f+0.72f*powf(1.0f-nv,4.0f);result=mix3(result,sky(reflection,sun)*0.60f,fr);
   if(mat==4){float room=hash1((int)seed);result=result*0.73f+make_float3(0.13f,0.075f,0.028f)*(room>0.78f?0.3f:0.03f);}}
  if(mat==8)result=result+make_float3(1.1f,0.62f,0.19f)*0.8f;
  if(mat==7)result=result+albedo*(powf(fmaxf(0.0f,dot3(rd,sun)),4.0f)*0.35f);
  // Screen-space contact shading is applied in resolve; no painted far-facade substitute.
  float fog=1.0f-expf(-t*0.00068f);fog=lerpf(fog,1.0f,smoothf(FAR*0.68f,FAR,t));
  float3 haze=sky(norm3(make_float3(rd.x,0.004f,rd.z)),sun);result=mix3(result,haze,fog);
  if(C[11]==1.0f){result=paletteTint(lot.palette,0)*make_float3(0.4f,0.55f,0.6f);}
  if(C[11]==2.0f){int g=id>=0?id/MAX_FEATURES:0;result=make_float3(0.15f+0.75f*hash1(g),0.15f+0.75f*hash1(g+31),0.15f+0.75f*hash1(g+98));}
  if(C[11]==3.0f)result=(n+make_float3(1.0f,1.0f,1.0f))*0.5f;
 }
 Linear[b]=result.x;Linear[b+1]=result.y;Linear[b+2]=result.z;Linear[b+3]=1.0f;
}
__device__ float aces(float x){return sat(x*(2.51f*x+0.03f)/(x*(2.43f*x+0.59f)+0.14f));}
__global__ void resolveFrame(const float* Linear,float* History,unsigned int* Pixels,const float* C,const float* Hit,const float* Surface,int width,int height){
 int x=(int)(blockIdx.x*blockDim.x+threadIdx.x),y=(int)(blockIdx.y*blockDim.y+threadIdx.y);if(x>=width||y>=height)return;int b=(y*width+x)*4;
 float3 c=make_float3(Linear[b],Linear[b+1],Linear[b+2]);
 if(C[11]<0.5f&&(int)Hit[b+1]!=-10000&&Hit[b]<250.0f){
  float3 ro=cameraPosition(C),point=ro+rayDirection(C,x,y,width,height)*Hit[b];
  float3 n=make_float3(Surface[b],Surface[b+1],Surface[b+2]);float occ=0.0f;
  float radius=clampf((float)height/(fmaxf(Hit[b],0.5f)*1.08f)*0.8f,3.0f,28.0f);
  for(int k=0;k<8;k++){float a=(float)k*PI*0.25f+0.2f;int xx=(int)clampf((float)x+cosf(a)*radius,0.0f,(float)(width-1)),yy=(int)clampf((float)y+sinf(a)*radius,0.0f,(float)(height-1));int j=(yy*width+xx)*4;
   if((int)Hit[j+1]!=-10000){float3 other=ro+rayDirection(C,xx,yy,width,height)*Hit[j],delta=other-point;float len=length3(delta);if(len>0.04f&&len<2.0f)occ+=fmaxf(0.0f,dot3(n,delta/len)-0.12f)*expf(-len*1.4f)*0.28f;}}
  c=c*clampf(1.0f-occ,0.55f,1.0f);
 }
 float oldCount=History[b+3],count=C[19]<=1.0f?1.0f:fminf(64.0f,oldCount+1.0f);
 float3 old=make_float3(History[b],History[b+1],History[b+2]);
 if(C[19]>=64.0f&&oldCount>=64.0f)c=old;
 else c=mix3(old,c,1.0f/count);
 History[b]=c.x;History[b+1]=c.y;History[b+2]=c.z;History[b+3]=count;
 float sx=((float)x/(float)width-0.5f)*1.3f,sy=((float)y/(float)height-0.5f)*1.3f;
 c=c*(C[9]*(1.0f-(sx*sx+sy*sy)*0.12f));
 c=make_float3(powf(aces(c.x),1.0f/2.2f),powf(aces(c.y),1.0f/2.2f),powf(aces(c.z),1.0f/2.2f));Pixels[y*width+x]=packRGBA(c);
}
