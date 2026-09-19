import test from 'node:test';import assert from 'node:assert/strict';import fs from 'node:fs/promises';
import {compile} from '../vendor/cuda-webshader/compiler/compiler.js';import {Engine,ABI} from '../src/engine.js';import {installConstants,mockDevice,mockCanvas} from './mock-device.mjs';
const root=new URL('../',import.meta.url);const manifest=JSON.parse(await fs.readFile(new URL('generated/manifest.json',root),'utf8'));
test('every shipped shader matches authored CUDA and portable binding/scratch budgets',async()=>{
 assert.equal(manifest.length,11);const cache=new Map();
 for(const s of manifest){for(const n of s.dependencies)if(!cache.has(n))cache.set(n,await fs.readFile(new URL('kernels/'+n+'.cu',root),'utf8'));const a=compile(s.dependencies.map(n=>cache.get(n)).join('\n'),{entry:s.entry,workgroupSize:s.workgroupSize});
 assert.equal(a.wgsl,await fs.readFile(new URL('generated/'+s.entry+'.wgsl',root),'utf8'),s.entry);assert.ok(a.metadata.bindings.length<=8);assert.ok(a.metadata.workgroupStorageBytes<=16384);}
});
test('acceleration stores bounds, not per-building primitive pages; no macro model switch',async()=>{
 const common=await fs.readFile(new URL('kernels/common.cu',root),'utf8'),trace=await fs.readFile(new URL('kernels/trace.cu',root),'utf8');
 for(const name of ['WORLD_SIDE','WORLD_LOTS','LOT_FLOATS','CLUSTERS','GROUP_NODES','MAX_FEATURES','BUILD_CHUNK','BUILD_CHUNKS'])assert.match(common,new RegExp('#define '+name+' '+ABI[name]+'(?:\\s|$)'));
 assert.equal(ABI.GROUP_NODES,ABI.CLUSTERS*2);assert.ok(ABI.RAY_DISTANCE<(ABI.WORLD_SIDE/2-2)*36);
 assert.doesNotMatch(trace,/macroHit|proceduralFacadeDepth|pageHit|if\(cached\)/);assert.match(trace,/authoredGroup/);
});
test('runtime bindings, bounded tiles, indirect work and presentation contract',async()=>{
 installConstants();const device=mockDevice(),canvas=mockCanvas(),old=globalThis.fetch;globalThis.fetch=async url=>{const data=await fs.readFile(new URL(url));return{ok:true,json:async()=>JSON.parse(data),text:async()=>data.toString()};};
 try{const e=await new Engine(canvas).init({device,adapter:{info:{vendor:'TEST DOUBLE'}},width:128,height:96,warmup:false});e.frame();assert.ok(device.calls.some(c=>c[0]==='present'));assert.ok(device.calls.some(c=>c[0]==='indirect'));assert.equal(e.buffers.P,undefined);assert.equal(e.buffers.Nodes.byteLength,64*1048576);await e.resize(192,128);e.frame();await e.reset(42);await e.dispose();assert.ok(e.disposed);}finally{globalThis.fetch=old;}
});
test('both query and conservative bounds use one feature grammar with no store capacity truncation',async()=>{
 const assets=await fs.readFile(new URL('kernels/assets.cu',root),'utf8'),sink=await fs.readFile(new URL('kernels/sink.cu',root),'utf8');assert.match(assets,/Sink authoredGroup/);assert.match(sink,/s\.mode==1/);assert.match(sink,/featureHit\(f,/);assert.doesNotMatch(sink,/n>=PER_CLUSTER|count>=32/);
});

test('production runtime boots from CUDA sources rather than generated shader artifacts',async()=>{
 const engine=await fs.readFile(new URL('src/engine.js',root),'utf8'),pages=await fs.readFile(new URL('tools/pages.mjs',root),'utf8');
 assert.match(engine,/CUDA → WebGPU/);assert.match(engine,/compiler\/compiler\.js/);assert.match(engine,/kernels\/.*\.cu/);
 assert.doesNotMatch(engine,/generated\/manifest\.json/);assert.doesNotMatch(engine,/generated\/.*\.json/);
 assert.doesNotMatch(pages,/['"]generated['"]/);assert.match(pages,/generated\/ is intentionally not deployed/);
});
