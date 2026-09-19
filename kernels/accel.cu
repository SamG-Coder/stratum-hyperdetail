// Only bounding hierarchies are resident. No 4,096 x 2,048 primitive array is allocated.
__global__ void buildGroupBounds(const unsigned int* Queue,const float* World,float* Nodes,int queueBase){
 int q=queueBase+(int)blockIdx.x,g=(int)threadIdx.x;if(q>=(int)Queue[0]||g>=CLUSTERS)return;
 int slot=(int)Queue[q+1];Lot lot=readLot(World,slot);Sink sink=newSink(1,make_float3(0.0f,0.0f,0.0f),make_float3(0.0f,0.0f,1.0f),FAR);
 sink=authoredGroup(lot,g,sink);int b=(slot*GROUP_NODES+CLUSTERS+g)*8;
 Nodes[b]=sink.lo.x;Nodes[b+1]=sink.lo.y;Nodes[b+2]=sink.lo.z;Nodes[b+3]=(float)sink.count;
 Nodes[b+4]=sink.hi.x;Nodes[b+5]=sink.hi.y;Nodes[b+6]=sink.hi.z;Nodes[b+7]=(float)g;
}
__global__ void reduceGroupBounds(const unsigned int* Queue,float* Nodes,int first,int queueBase){
 int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);int q=queueBase+i/first,j=i%first+first;
 if(q>=(int)Queue[0]||q>=queueBase+BUILD_CHUNK)return;int slot=(int)Queue[q+1];
 int b=(slot*GROUP_NODES+j)*8,a=(slot*GROUP_NODES+j*2)*8,c=a+8;
 for(int k=0;k<3;k++){Nodes[b+k]=fminf(Nodes[a+k],Nodes[c+k]);Nodes[b+4+k]=fmaxf(Nodes[a+4+k],Nodes[c+4+k]);}
 Nodes[b+3]=Nodes[a+3]+Nodes[c+3];Nodes[b+7]=0.0f;
}
