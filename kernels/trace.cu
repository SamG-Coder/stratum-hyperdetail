// Every visible lot uses the exact shared assets.cu grammar; bounds only reject missed groups.
__device__ float nodeNear(const float* Nodes,int node,float3 ro,float3 rd,float best){
 int b=node*8;if(Nodes[b+3]<0.5f)return FAR;
 float2 range=boxRange(ro,rd,make_float3(Nodes[b],Nodes[b+1],Nodes[b+2]),make_float3(Nodes[b+4],Nodes[b+5],Nodes[b+6]));
 if(range.y<fmaxf(0.001f,range.x)||range.x>best)return FAR;
 return fmaxf(0.001f,range.x);
}
__device__ Sink queryLot(Lot lot,const float* Nodes,int slot,float3 ro,float3 rd,float best,float cone){
 Sink hit=newSink(0,ro,rd,best);hit.pixelCone=cone;
 int stack[8];int top=0,node=1,steps=0;
 while(node>0&&steps<GROUP_NODES){
  steps++;float near=nodeNear(Nodes,slot*GROUP_NODES+node,ro,rd,hit.t);
  if(near< hit.t&&node<CLUSTERS){
   int left=node*2,right=left+1;
   float ln=nodeNear(Nodes,slot*GROUP_NODES+left,ro,rd,hit.t),rn=nodeNear(Nodes,slot*GROUP_NODES+right,ro,rd,hit.t);
   bool lh=ln<hit.t,rh=rn<hit.t;
   if(lh&&rh){int next=ln<rn?left:right;int other=ln<rn?right:left;stack[top]=other;top++;node=next;}
   else if(lh)node=left;else if(rh)node=right;else{node=0;if(top>0){top--;node=stack[top];}}
  }else{
   if(near<hit.t&&node>=CLUSTERS){
    Sink s=newSink(0,ro,rd,hit.t);s.pixelCone=cone;s=authoredGroup(lot,node-CLUSTERS,s);
    if(s.fid>=0&&s.t<hit.t){hit=s;hit.fid=(node-CLUSTERS)*MAX_FEATURES+s.fid;}
   }
   node=0;if(top>0){top--;node=stack[top];}
  }
 }
 return hit;
}
__global__ void tracePrimary(const float* World,const float* Nodes,const unsigned int* Origin,const float* C,float* Hit,float* Surface,int width,int height,int rowStart,int rowCount){
 int x=(int)(blockIdx.x*blockDim.x+threadIdx.x),y=rowStart+(int)(blockIdx.y*blockDim.y+threadIdx.y);
 if(x>=width||y>=height||y>=rowStart+rowCount)return;
 int b=(y*width+x)*4;float3 ro=cameraPosition(C),rd=rayDirection(C,x,y,width,height);
 float best=FAR;int fid=-10000,material=10,foundSlot=-1;float seed=0.0f;float3 normal=make_float3(0.0f,1.0f,0.0f);
 if(rd.y<-0.000001f){float ground=-ro.y/rd.y;if(ground>0.001f&&ground<best){best=ground;fid=-1;}}
 // Vertical slab bounds the work without defining a finite horizontal city edge.
 float near=0.001f,far=best;
 if(fabsf(rd.y)>0.0000001f){float a=(-0.5f-ro.y)/rd.y,c=(SCENE_TOP-ro.y)/rd.y;near=fmaxf(near,fminf(a,c));far=fminf(far,fmaxf(a,c));}
 else if(ro.y>SCENE_TOP||ro.y<-0.5f)far=-1.0f;
 if(near<=far){
  float t=near+0.0002f;float3 start=ro+rd*t;
  int cx=(int)floorf((start.x+18.0f)/CELL),cz=(int)floorf((start.z+18.0f)/CELL);
  int sx=rd.x>0.0f?1:-1,sz=rd.z>0.0f?1:-1;
  float tx=((float)cx*CELL+(sx>0?18.0f:-18.0f)-ro.x)*safeInv(rd.x);
  float tz=((float)cz*CELL+(sz>0?18.0f:-18.0f)-ro.z)*safeInv(rd.z);
  float dx=CELL*fabsf(safeInv(rd.x)),dz=CELL*fabsf(safeInv(rd.z));
  for(int step=0;step<160;step++){
   if(t>best||t>far||t>FAR)break;
   int slot=slotFor(Origin,cx,cz);Lot lot=readLot(World,slot);
   float3 cp=make_float3((float)cx*CELL+lot.ox,0.0f,(float)cz*CELL+lot.oz);
   float3 lr=turnLocal(ro-cp,lot.turn),ld=turnLocal(rd,lot.turn);
   if(nodeNear(Nodes,slot*GROUP_NODES+1,lr,ld,best)<best){
    Sink hit=queryLot(lot,Nodes,slot,lr,ld,best,1.08f/(float)height);
    if(hit.fid>=0&&hit.t<best){best=hit.t;fid=hit.fid;material=hit.feature.material;seed=hit.feature.seed;
     normal=turnWorld(featureNormal(hit.feature,lr+ld*best),lot.turn);foundSlot=slot;}
   }
   if(tx<tz){t=tx;tx+=dx;cx+=sx;}else{t=tz;tz+=dz;cz+=sz;}
  }
 }
 Hit[b]=best;Hit[b+1]=(float)fid;Hit[b+2]=(float)material;Hit[b+3]=seed;
 Surface[b]=normal.x;Surface[b+1]=normal.y;Surface[b+2]=normal.z;Surface[b+3]=(float)foundSlot;
}
