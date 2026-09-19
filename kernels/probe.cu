// Bounded real-GPU/CPU parity probe. No test shader substitutes a simplified building.
__global__ void probeGrammar(const float* Inputs,float* Output){
 int g=(int)Inputs[0],index=(int)Inputs[1];Lot l;
 l.w=Inputs[2];l.d=Inputs[3];l.h=Inputs[4];l.type=(int)Inputs[5];l.mat=(int)Inputs[6];l.floors=(int)Inputs[7];l.seed=Inputs[8];
 l.storey=Inputs[9];l.roofScale=Inputs[10];l.turn=0;l.ox=0.0f;l.oz=0.0f;l.palette=0.0f;
 Sink s=newSink(2,make_float3(0.0f,0.0f,0.0f),make_float3(0.0f,0.0f,1.0f),FAR);s.target=index;s=authoredGroup(l,g,s);
 Output[0]=(float)s.count;Output[1]=(float)s.feature.shape;Output[2]=(float)s.feature.material;Output[3]=s.feature.seed;
 Output[4]=s.feature.p.x;Output[5]=s.feature.p.y;Output[6]=s.feature.p.z;
 Output[7]=s.feature.h.x;Output[8]=s.feature.h.y;Output[9]=s.feature.h.z;Output[10]=(float)s.feature.turn;
 float3 ro=make_float3(Inputs[11],Inputs[12],Inputs[13]),rd=make_float3(Inputs[14],Inputs[15],Inputs[16]);
 Sink hit=newSink(0,ro,rd,FAR);hit=authoredGroup(l,g,hit);Output[11]=hit.t;Output[12]=(float)hit.fid;
 float3 n=make_float3(0.0f,0.0f,0.0f);if(hit.fid>=0)n=featureNormal(hit.feature,ro+rd*hit.t);
 Output[13]=n.x;Output[14]=n.y;Output[15]=n.z;
}
