// Analytic geometry ray visibility. Nearest hit wins; no alpha-painted billboards.
__device__ float3 localPoint(float3 p,int turn){if(turn%2==1)return make_float3(p.z,p.y,-p.x);return p;}
__device__ float3 worldNormal(float3 p,int turn){if(turn%2==1)return make_float3(-p.z,p.y,p.x);return p;}
// A foliage envelope contains stable, leaf-sized discs, generated at ray time.
// The envelope itself is never painted as an opaque sphere.
__device__ float3 leafNormal(int x,int y,int z,int seed){
 return norm3(make_float3(hash2(x+y*19,z,seed)-0.5f,0.25f+hash2(x-y,z+91,seed)*0.7f,hash2(z+y*31,x,seed+17)-0.5f));
}
__device__ float foliageHit(float3 ro,float3 rd,float3 centre,float3 radius,int seed,float begin,float end){
 const float step=0.18f;
 float t=fmaxf(0.001f,begin)+0.0001f;float3 pos=ro+rd*t;
 int ix=(int)floorf(pos.x/step);int iy=(int)floorf(pos.y/step);int iz=(int)floorf(pos.z/step);
 int sx=rd.x>0.0f?1:-1;int sy=rd.y>0.0f?1:-1;int sz=rd.z>0.0f?1:-1;
 float tx=((float)(ix+(sx>0?1:0))*step-ro.x)*safeInv(rd.x);
 float ty=((float)(iy+(sy>0?1:0))*step-ro.y)*safeInv(rd.y);
 float tz=((float)(iz+(sz>0?1:0))*step-ro.z)*safeInv(rd.z);
 float dx=step*fabsf(safeInv(rd.x));float dy=step*fabsf(safeInv(rd.y));float dz=step*fabsf(safeInv(rd.z));
 for(int k=0;k<48;k++){
  if(t>end)break;float exit=fminf(tx,fminf(ty,tz));
  float3 c=make_float3(((float)ix+0.5f)*step,((float)iy+0.5f)*step,((float)iz+0.5f)*step);
  float3 q=c-centre;float ell=q.x*q.x/(radius.x*radius.x)+q.y*q.y/(radius.y*radius.y)+q.z*q.z/(radius.z*radius.z);
  if(ell<1.0f&&hash2(ix+iy*113,iz,seed)>0.20f){
   float3 n=leafNormal(ix,iy,iz,seed);float den=dot3(rd,n);
   if(fabsf(den)>0.00001f){float u=dot3(c-ro,n)/den;
    if(u>=t&&u<=exit){float3 pp=ro+rd*u-c;float3 tangent=norm3(cross3(n,make_float3(0.0f,0.0f,1.0f)));float3 bitangent=cross3(n,tangent);float a=dot3(pp,tangent)/0.125f;float b=dot3(pp,bitangent)/0.062f;if(a*a+b*b<1.0f)return u;}
   }
  }
  if(tx<ty&&tx<tz){t=tx;tx+=dx;ix+=sx;}else if(ty<tz){t=ty;ty+=dy;iy+=sy;}else{t=tz;tz+=dz;iz+=sz;}
 }
 return FAR;
}

__device__ float featureHit(Feature feature,float3 ro,float3 rd,float pixelCone){
 int shape=feature.shape;if(shape<0)return FAR;
 float3 cp=feature.p,h=feature.h;int turn=feature.turn;
 float3 p=localPoint(ro-cp,turn);float3 d=localPoint(rd,turn);
 float2 range=boxRange(p,d,h*-1.0f,h);
 if(shape==3||shape==4)range=boxRange(p,d,make_float3(-h.x,0.0f,-h.z),h);
 if(range.y<fmaxf(0.001f,range.x))return FAR;
 float t=range.x>0.001f?range.x:range.y;
 if(shape==1){
  float3 a=make_float3(p.x/h.x,p.y/h.y,p.z/h.z);float3 v=make_float3(d.x/h.x,d.y/h.y,d.z/h.z);
  float aa=dot3(v,v);float bb=dot3(a,v);float cc=dot3(a,a)-1.0f;float disc=bb*bb-aa*cc;if(disc<0.0f)return FAR;
  t=(-bb-sqrtf(disc))/aa;if(t<=0.001f)t=(-bb+sqrtf(disc))/aa;
 }
 if(shape==2){
  float a=d.x*d.x/(h.x*h.x)+d.z*d.z/(h.z*h.z);
  float bb=p.x*d.x/(h.x*h.x)+p.z*d.z/(h.z*h.z);float cc=p.x*p.x/(h.x*h.x)+p.z*p.z/(h.z*h.z)-1.0f;
  float best=FAR;float disc=bb*bb-a*cc;
  if(disc>=0.0f&&a>0.0000001f){float u=(-bb-sqrtf(disc))/a;float v=(-bb+sqrtf(disc))/a;
   if(u>0.001f&&fabsf(p.y+d.y*u)<=h.y)best=u;
   if(v>0.001f&&fabsf(p.y+d.y*v)<=h.y)best=fminf(best,v);
  }
  for(int k=0;k<2;k++)if(fabsf(d.y)>0.000001f){float cy=k==0?-h.y:h.y;float u=(cy-p.y)/d.y;float xx=(p.x+d.x*u)/h.x;float zz=(p.z+d.z*u)/h.z;if(u>0.001f&&xx*xx+zz*zz<=1.0f)best=fminf(best,u);}
  t=best;
 }
 if(shape==3){
  float near=fmaxf(0.001f,range.x);float far=range.y;
  // Clip the AABB interval by the two sloping half-spaces of a triangular prism.
  for(int k=0;k<2;k++){
   float sign=k==0?1.0f:-1.0f;float numer=h.y-(p.y+sign*p.x*h.y/h.x);float denom=d.y+sign*d.x*h.y/h.x;
   if(fabsf(denom)<0.000001f){if(numer<0.0f)return FAR;}
   else {float u=numer/denom;if(denom>0.0f)far=fminf(far,u);else near=fmaxf(near,u);}
  }
  if(far<near)return FAR;t=near;
 }
 if(shape==4||shape==7){
  float best=FAR;float a=d.x*d.x/(h.x*h.x)+d.y*d.y/(h.y*h.y);
  float bb=p.x*d.x/(h.x*h.x)+p.y*d.y/(h.y*h.y);
  for(int ring=0;ring<2;ring++){
   float rad=ring==0?1.0f:0.77f;float cc=p.x*p.x/(h.x*h.x)+p.y*p.y/(h.y*h.y)-rad*rad;float disc=bb*bb-a*cc;
   if(disc>=0.0f&&a>0.000001f)for(int k=0;k<2;k++){float u=(-bb+(k==0?-sqrtf(disc):sqrtf(disc)))/a;if(u>0.001f&&fabsf(p.z+d.z*u)<=h.z&&(shape==7||p.y+d.y*u>=0.0f))best=fminf(best,u);}
  }
  if(fabsf(d.z)>0.000001f)for(int k=0;k<2;k++){float u=((k==0?-h.z:h.z)-p.z)/d.z;float xx=(p.x+d.x*u)/h.x;float yy=(p.y+d.y*u)/h.y;float rr=xx*xx+yy*yy;if(u>0.001f&&rr<=1.0f&&rr>=0.77f*0.77f&&(shape==7||yy>=0.0f))best=fminf(best,u);}
  t=best;
 }
 if(shape==1&&feature.material==7&&t<FAR){
  t=foliageHit(ro,rd,cp,h,(int)feature.seed,range.x,range.y);
 }
 if(t<=0.001f)return FAR;
 // Close-range mortar relief is ray-evaluated, not claimed as millions of polygons.
 if(shape==0&&(feature.material==0||feature.material==1)&&h.y>1.0f){
  float3 hit=p+d*t;float ex=fabsf(fabsf(hit.x)-h.x);float ez=fabsf(fabsf(hit.z)-h.z);
  if(fabsf(fabsf(hit.y)-h.y)>0.01f){
   float3 n=ex<ez?make_float3(hit.x<0.0f?-1.0f:1.0f,0.0f,0.0f):make_float3(0.0f,0.0f,hit.z<0.0f?-1.0f:1.0f);
   float denom=dot3(n,d);
   if(denom<-0.08f){float initial=t;for(int k=0;k<4;k++){float3 wp=ro+rd*t;float3 wn=worldNormal(n,turn);float2 uv=masonryUV(wp,wn);float displacement=masonryHeight(uv.x,uv.y,feature.material)*frequencyWeight(t*pixelCone,40.0f);t=initial+displacement/denom;}}
  }
 }
 return t;
}
__device__ float3 featureNormal(Feature feature,float3 point){
 int shape=feature.shape,turn=feature.turn;float3 p=localPoint(point-feature.p,turn),h=feature.h;float3 n=make_float3(0.0f,1.0f,0.0f);
 if(shape==1)n=norm3(make_float3(p.x/(h.x*h.x),p.y/(h.y*h.y),p.z/(h.z*h.z)));
 else if(shape==2){if(fabsf(fabsf(p.y)-h.y)<0.01f)n=make_float3(0.0f,p.y<0.0f?-1.0f:1.0f,0.0f);else n=norm3(make_float3(p.x/(h.x*h.x),0.0f,p.z/(h.z*h.z)));}
 else if(shape==4||shape==7){
  if(fabsf(fabsf(p.z)-h.z)<0.008f)n=make_float3(0.0f,0.0f,p.z<0.0f?-1.0f:1.0f);
  else {float rr=p.x*p.x/(h.x*h.x)+p.y*p.y/(h.y*h.y);n=norm3(make_float3(p.x/(h.x*h.x),p.y/(h.y*h.y),0.0f))*(rr<0.8f?-1.0f:1.0f);}
 }
 else if(shape==3){
  if(fabsf(fabsf(p.z)-h.z)<0.006f)n=make_float3(0.0f,0.0f,p.z<0.0f?-1.0f:1.0f);
  else if(p.y<0.003f)n=make_float3(0.0f,-1.0f,0.0f);
  else n=norm3(make_float3(p.x<0.0f?-h.y/h.x:h.y/h.x,1.0f,0.0f));
 }else{
  float3 a=make_float3(fabsf(fabsf(p.x)-h.x),fabsf(fabsf(p.y)-h.y),fabsf(fabsf(p.z)-h.z));
  n=a.x<a.y&&a.x<a.z?make_float3(p.x<0.0f?-1.0f:1.0f,0.0f,0.0f):(a.y<a.z?make_float3(0.0f,p.y<0.0f?-1.0f:1.0f,0.0f):make_float3(0.0f,0.0f,p.z<0.0f?-1.0f:1.0f));
  // Small bevel normals make stone edges catch light without changing gross visibility.
  float bevel=fminf(0.035f,fminf(fminf(h.x,h.y),h.z)*0.18f);
  float3 q=make_float3(fmaxf(fabsf(p.x)-h.x+bevel,0.0f)*(p.x<0.0f?-1.0f:1.0f),fmaxf(fabsf(p.y)-h.y+bevel,0.0f)*(p.y<0.0f?-1.0f:1.0f),fmaxf(fabsf(p.z)-h.z+bevel,0.0f)*(p.z<0.0f?-1.0f:1.0f));
  if(dot3(q,q)>0.000001f)n=norm3(q);
 }
 if(shape==1&&feature.material==7){return leafNormal((int)floorf(point.x/0.18f),(int)floorf(point.y/0.18f),(int)floorf(point.z/0.18f),(int)feature.seed);}
 return worldNormal(n,turn);
}
