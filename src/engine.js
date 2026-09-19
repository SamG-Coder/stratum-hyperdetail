import {GpuRuntime} from '../vendor/cuda-webshader/runtime/runtime.js';
export const ABI=Object.freeze({WORLD_SIDE:128,WORLD_LOTS:16384,LOT_FLOATS:16,CLUSTERS:64,GROUP_NODES:128,MAX_FEATURES:64,BUILD_CHUNK:2048,BUILD_CHUNKS:8,RAY_DISTANCE:2000});
export const BUILD_ID='infinite-shared-grammar-1';
const STAGES=['World descriptors','Bounds refresh','Exact visibility','Materials + light','Resolve'];
export class Engine{
 constructor(canvas){this.canvas=canvas;this.buffers={};this.kernels={};this.input=new Float32Array(32);this.frames=0;this.errors=[];this.timings=null;this.timingBusy=false;this.disposed=false;}
 async init({width=1600,height=900,seed=1788,recompile=false,onProgress=()=>{},onError=()=>{},device=null,adapter=null,warmup=true}={}){
  const ownsDevice=!device;
  if(!device){if(!navigator.gpu)throw Error('WebGPU requires a supported browser on localhost or HTTPS.');adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});if(!adapter)throw Error('No WebGPU adapter available.');device=await adapter.requestDevice({requiredFeatures:adapter.features.has('timestamp-query')?['timestamp-query']:[]});}
  this.device=device;this.adapter=adapter;this.onProgress=onProgress;
  this.runtime=new GpuRuntime(device,{adapter,ownsDevice,uniformCapacity:262144,onError:e=>{this.errors.push(String(e?.message||e));onError(e);}});this.info=this.runtime.describe();this.context=this.canvas?.getContext('webgpu')??null;
  const response=await fetch(new URL('../generated/manifest.json',import.meta.url));if(!response.ok)throw Error('Run npm run build before starting the server.');
  this.manifest=await response.json();if(!this.manifest.some(k=>k.entry==='buildGroupBounds'))throw Error('Old generated shaders. Run npm run build, then reload.');
  let compile;const sources=new Map();if(recompile)({compile}=await import('../vendor/cuda-webshader/compiler/compiler.js'));
  for(let i=0;i<this.manifest.length;i++){
   const spec=this.manifest[i];onProgress('Compile '+spec.entry,i/this.manifest.length*.6);let artifact;
   if(compile){for(const n of spec.dependencies)if(!sources.has(n))sources.set(n,await(await fetch(new URL('../kernels/'+n+'.cu',import.meta.url))).text());artifact=compile(spec.dependencies.map(n=>sources.get(n)).join('\n'),{entry:spec.entry,workgroupSize:spec.workgroupSize});}
   else{const r=await fetch(new URL('../generated/'+spec.entry+'.json',import.meta.url));if(!r.ok)throw Error('Missing generated '+spec.entry);artifact=await r.json();}
   this.kernels[spec.entry]=await this.runtime.kernel(artifact);
  }
  const a=ABI,sizes={World:a.WORLD_LOTS*a.LOT_FLOATS,Tags:a.WORLD_LOTS*4,Nodes:a.WORLD_LOTS*a.GROUP_NODES*8,Origin:4,C:64,I:32,Queue:a.WORLD_LOTS+1,Args:a.BUILD_CHUNKS*21,Stats:256};
  for(const [name,n]of Object.entries(sizes))this.buffers[name]=this.runtime.createBuffer(n*4,{label:'STRATUM '+name,usage:name==='Args'?GPUBufferUsage.INDIRECT:0});
  if(device.features.has('timestamp-query')){this.queries=device.createQuerySet({type:'timestamp',count:10});this.queryResolve=device.createBuffer({size:256,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});this.queryRead=device.createBuffer({size:80,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});}
  await this.resize(width,height);this.runtime.batch().dispatch(this.bind('initCamera',{seed}),[1]).submit();await this.runtime.idle();
  if(warmup){
   this.prepare(0,false);
   // Start-up/teleport work is divided into separate bounded submissions, not one giant kernel.
   for(let chunk=0;chunk<a.BUILD_CHUNKS;chunk++){this.refreshChunk(chunk,this.runtime.batch()).submit();await this.runtime.idle();onProgress('Build exact group bounds '+(chunk+1)+'/'+a.BUILD_CHUNKS,.6+.4*(chunk+1)/a.BUILD_CHUNKS);}
  }
  onProgress('Ready',1);return this;
 }
 bind(entry,scalars={}){const k=this.kernels[entry];if(!k)throw Error('Unknown kernel '+entry);const b=Object.fromEntries(k.artifact.metadata.bindings.map(v=>{if(!this.buffers[v.name])throw Error('Missing '+entry+'.'+v.name);return[v.name,this.buffers[v.name]];}));return k.bind(b,scalars);}
 async resize(width,height){
  if(!Number.isFinite(width)||!Number.isFinite(height))throw Error('Invalid render dimensions');width=Math.max(64,Math.ceil(width/64)*64);height=Math.max(64,Math.ceil(height/8)*8);
  if(width>this.device.limits.maxTextureDimension2D||height>this.device.limits.maxTextureDimension2D||width*height*16>this.device.limits.maxStorageBufferBindingSize)throw Error('Resolution exceeds this GPU buffer limit.');
  if(width===this.width&&height===this.height)return;await this.runtime.idle();
  for(const n of ['Hit','Surface','Linear','History','Pixels'])if(this.buffers[n])this.runtime.destroyBuffer(this.buffers[n]);
  this.width=width;this.height=height;for(const n of ['Hit','Surface','Linear','History'])this.buffers[n]=this.runtime.createBuffer(width*height*16,{label:n});this.buffers.Pixels=this.runtime.createBuffer(width*height*4,{label:'Final RGBA8'});
  if(this.canvas){this.canvas.width=width;this.canvas.height=height;}if(this.context)this.context.configure({device:this.device,format:'rgba8unorm',usage:GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT,alphaMode:'opaque'});
  this.calls={};for(const n of ['clearQueue','prepareLots','planBounds'])this.calls[n]=this.bind(n);this.calls.stepCamera=this.bind('stepCamera',{width,height,dt:0});
  this.calls.tracePrimary=this.bind('tracePrimary',{width,height,rowStart:0,rowCount:64});this.calls.shadePixels=this.bind('shadePixels',{width,height,rowStart:0,rowCount:64});this.calls.resolveFrame=this.bind('resolveFrame',{width,height});
  this.chunks=[];for(let i=0;i<ABI.BUILD_CHUNKS;i++){const queueBase=i*ABI.BUILD_CHUNK;const calls=[this.bind('buildGroupBounds',{queueBase})];for(let first=32;first>=1;first/=2)calls.push(this.bind('reduceGroupBounds',{first,queueBase}));this.chunks.push(calls);}
  // Changing resolution starts a new sampling sequence without changing world identity.
  this.input[8]=this.frames===0?0:this.input[8];this.resizeDirty=true;
 }
 stage(index,timed){return this.runtime.batch({label:STAGES[index],...(timed?{timestampWrites:{querySet:this.queries,beginningOfPassWriteIndex:index*2,endOfPassWriteIndex:index*2+1}}:{})});}
 prepare(dt,timed){
  this.input[23]=this.resizeDirty?1:0;this.resizeDirty=false;this.runtime.write(this.buffers.I,this.input);this.input[23]=0;const b=this.stage(0,timed),c=this.calls;
  c.stepCamera.setScalars({width:this.width,height:this.height,dt:Math.max(0,Math.min(.1,dt))});b.dispatch(c.stepCamera,[1]).dispatch(c.clearQueue,[1]).dispatch(c.prepareLots,[ABI.WORLD_LOTS/64]).dispatch(c.planBounds,[1]).submit();
 }
 refreshChunk(chunk,batch){for(let level=0;level<7;level++)batch.dispatch(this.chunks[chunk][level],[1],{resource:this.buffers.Args,offset:(chunk*21+level*3)*4});return batch;}
 frame(dt=1/60,{draw=true}={}){
  if(this.disposed)throw Error('Renderer disposed');const timed=!!this.queries&&!this.timingBusy&&draw&&this.frames%24===0;
  this.prepare(dt,timed);const accel=this.stage(1,timed);for(let c=0;c<ABI.BUILD_CHUNKS;c++)this.refreshChunk(c,accel);accel.submit();
  if(draw){
   const trace=this.stage(2,timed),shade=this.stage(3,timed);const rows=64;
   for(let y=0;y<this.height;y+=rows){const count=Math.min(rows,this.height-y);trace.dispatch(this.calls.tracePrimary.setScalars({rowStart:y,rowCount:count}),[Math.ceil(this.width/8),Math.ceil(count/8)]);shade.dispatch(this.calls.shadePixels.setScalars({rowStart:y,rowCount:count}),[Math.ceil(this.width/8),Math.ceil(count/8)]);}
   trace.submit();shade.submit();const resolve=this.stage(4,timed);resolve.dispatch(this.calls.resolveFrame,[Math.ceil(this.width/8),Math.ceil(this.height/8)]).endPass();
   if(this.context)resolve.encoder.copyBufferToTexture({buffer:this.buffers.Pixels.gpuBuffer,bytesPerRow:this.width*4,rowsPerImage:this.height},{texture:this.context.getCurrentTexture()},[this.width,this.height]);
   if(timed){resolve.encoder.resolveQuerySet(this.queries,0,10,this.queryResolve,0);resolve.encoder.copyBufferToBuffer(this.queryResolve,0,this.queryRead,0,80);}resolve.submit();if(timed)this.readTimings();
  }
  this.frames++;
 }
 async readTimings(){this.timingBusy=true;try{await this.queryRead.mapAsync(GPUMapMode.READ);const t=new BigUint64Array(this.queryRead.getMappedRange());this.timings=STAGES.map((label,i)=>({label,ms:Number(t[2*i+1]-t[2*i])/1e6}));this.queryRead.unmap();}catch{this.timings=null;}finally{this.timingBusy=false;}}
 async inspect(){
  const [c,o,q]=await Promise.all([this.runtime.read(this.buffers.C),this.runtime.read(this.buffers.Origin,Uint32Array),this.runtime.read(this.buffers.Queue,Uint32Array,4,0)]);
  const signed=(lo,hi)=>{const n=(BigInt(hi)<<32n)|BigInt(lo);return BigInt.asIntN(64,n).toString();};
  return{build:BUILD_ID,mode:'exact-shared-grammar',camera:Array.from(c),origin:{x:signed(o[0],o[1]),z:signed(o[2],o[3])},refreshedLots:q[0],accelerationLots:ABI.WORLD_LOTS,far:ABI.RAY_DISTANCE,geometryModels:1,primitivePageAllocation:0,accelerationBytes:this.buffers.Nodes.byteLength,allocatedBytes:Object.values(this.buffers).reduce((a,b)=>a+b.byteLength,0),width:this.width,height:this.height,frames:this.frames,timings:this.timings,adapter:this.info,errors:this.errors};
 }
 async screenshotPixels(){return this.runtime.read(this.buffers.Pixels,Uint8Array);}
 async reset(seed=1788){await this.runtime.idle();this.input.fill(0);this.runtime.batch().clear(this.buffers.World).clear(this.buffers.Tags).clear(this.buffers.History).dispatch(this.bind('initCamera',{seed}),[1]).submit();this.frame(0,{draw:false});await this.runtime.idle();}
 async dispose(){if(this.disposed)return;await this.runtime.idle();this.disposed=true;this.queries?.destroy();this.queryResolve?.destroy();this.queryRead?.destroy();this.runtime.dispose();}
}
