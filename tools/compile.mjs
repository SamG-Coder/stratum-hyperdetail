import fs from 'node:fs/promises';
import {compile} from '../vendor/cuda-webshader/compiler/compiler.js';
const root=new URL('../',import.meta.url);
const units=['common','world','geometry','sink','assets','accel','trace','materials','shade','probe'];
const rules={
 initCamera:['world',[1,1,1]],stepCamera:['world',[1,1,1]],clearQueue:['world',[1,1,1]],prepareLots:['world',[64,1,1]],planBounds:['world',[8,1,1]],
 buildGroupBounds:['accel',[64,1,1]],reduceGroupBounds:['accel',[64,1,1]],tracePrimary:['trace',[8,8,1]],shadePixels:['shade',[8,8,1]],resolveFrame:['shade',[8,8,1]],probeGrammar:['probe',[1,1,1]],
};
const dependencies={world:['common','world'],accel:['common','world','geometry','sink','assets','accel'],trace:['common','world','geometry','sink','assets','trace'],shade:['common','world','materials','shade'],probe:['common','world','geometry','sink','assets','probe']};
const texts=Object.fromEntries(await Promise.all(units.map(async n=>[n,await fs.readFile(new URL('kernels/'+n+'.cu',root),'utf8')])));
const stage=new URL('.build-generated/',root);await fs.rm(stage,{recursive:true,force:true});await fs.mkdir(stage,{recursive:true});
const manifest=[];const failures=[];
for(const [entry,[unit,workgroupSize]] of Object.entries(rules)){
 try{
  const deps=dependencies[unit],artifact=compile(deps.map(n=>texts[n]).join('\n'),{entry,workgroupSize});
  if(artifact.metadata.bindings.length>8)throw Error('More than eight storage buffers');
  if(artifact.metadata.workgroupStorageBytes>16384)throw Error('Workgroup scratch exceeds portable 16 KiB budget');
  const {ast,kernel,...portable}=artifact;
  await fs.writeFile(new URL(entry+'.json',stage),JSON.stringify(portable));await fs.writeFile(new URL(entry+'.wgsl',stage),artifact.wgsl);
  manifest.push({entry,file:unit,workgroupSize,dependencies:deps,bindings:artifact.metadata.bindings.map(b=>b.name),uniformBytes:artifact.metadata.uniformSize});
  console.log('OK '+entry+' | '+artifact.metadata.bindings.length+' buffers | '+artifact.wgsl.length+' WGSL bytes');
 }catch(e){failures.push(entry);console.error('FAILED '+entry+': '+(e.stack||e));}
}
if(failures.length){await fs.rm(stage,{recursive:true,force:true});throw Error('Build failed; previous generated artifacts retained: '+failures.join(', '));}
await fs.writeFile(new URL('manifest.json',stage),JSON.stringify(manifest,null,2));
await fs.rm(new URL('generated/',root),{recursive:true,force:true});await fs.rename(stage,new URL('generated/',root));
await fs.writeFile(new URL('Stratum.cu',root),'// Combined authored CUDA; build from split sources.\n'+units.map(n=>'\n// ===== '+n+' =====\n'+texts[n]).join('\n'));
