// One sink for exact ray queries, conservative bounds, and reference materialisation.
__device__ Sink newSink(int mode,float3 ro,float3 rd,float best){
 Sink s;s.mode=mode;s.count=0;s.target=-1;s.t=best;s.ro=ro;s.rd=rd;
 s.lo=make_float3(1000000.0f,1000000.0f,1000000.0f);
 s.hi=make_float3(-1000000.0f,-1000000.0f,-1000000.0f);
 s.feature.p=make_float3(0.0f,0.0f,0.0f);s.feature.h=make_float3(1.0f,1.0f,1.0f);
 s.feature.shape=-1;s.feature.material=0;s.feature.turn=0;s.feature.seed=0.0f;s.fid=-1;s.pixelCone=0.0f;s.roofBase=FAR;s.roofScale=1.0f;
 return s;
}
__device__ Sink emitFeature(Sink s,float3 p,float3 h,int shape,int mat,int turn,float seed){
 if(p.y>=s.roofBase){p.y=s.roofBase+(p.y-s.roofBase)*s.roofScale;h.y*=s.roofScale;}
 Feature f;f.p=p;f.h=h;f.shape=shape;f.material=mat;f.turn=turn;f.seed=seed;
 int id=s.count;s.count++;
 if(s.mode==1){
  float3 extent=h;if(turn%2==1)extent=make_float3(h.z,h.y,h.x);
  float3 lo=p-extent,hi=p+extent;if(shape==3||shape==4)lo.y=p.y;
  // Includes the entire permitted masonry displacement; used only for rejection.
  float3 pad=make_float3(0.025f,0.025f,0.025f);
  s.lo=min3(s.lo,lo-pad);s.hi=max3(s.hi,hi+pad);
 }else if(s.mode==2){if(id==s.target){s.feature=f;s.fid=id;}}
 else{
  float t=featureHit(f,s.ro,s.rd,s.pixelCone);
  if(t>0.001f&&t<s.t){s.t=t;s.feature=f;s.fid=id;}
 }
 return s;
}
__device__ Sink emitWall(Sink sink,int face,float cx,float cz,float w,float d,float u,float y,float out,float hw,float hy,float hd,int mat,float seed){
 float3 p=make_float3(cx+u,y,cz+d+out),h=make_float3(hw,hy,hd);
 if(face==1){p=make_float3(cx+w+out,y,cz+u);h=make_float3(hd,hy,hw);}
 if(face==2)p=make_float3(cx+u,y,cz-d-out);
 if(face==3){p=make_float3(cx-w-out,y,cz+u);h=make_float3(hd,hy,hw);}
 return emitFeature(sink,p,h,0,mat,0,seed);
}
