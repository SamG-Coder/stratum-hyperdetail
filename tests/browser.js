import {GpuRuntime} from '../vendor/cuda-webshader/runtime/runtime.js';
const log=document.querySelector('#log');const say=t=>{if(log)log.textContent+=t+'\n';console.log(t);};
export async function runChecks(){
 const adapter=await navigator.gpu?.requestAdapter({powerPreference:'low-power'});if(!adapter)throw Error('No WebGPU adapter');const device=await adapter.requestDevice();const errors=[];const runtime=new GpuRuntime(device,{adapter,ownsDevice:true,onError:e=>errors.push(String(e))});
 const manifest=await(await fetch('../generated/manifest.json')).json();let probe;
 for(const spec of manifest){const a=await(await fetch('../generated/'+spec.entry+'.json')).json();const k=await runtime.kernel(a);if(spec.entry==='probeGrammar')probe=k;say('PASS real WebGPU pipeline: '+spec.entry);}
 const fixtures=await(await fetch('./probes.json')).json();let max=0;
 for(let i=0;i<fixtures.length;i++){
  const f=fixtures[i],input=runtime.createBuffer(new Float32Array(f.input)),output=runtime.createBuffer(16*4);
  runtime.batch().dispatch(probe.bind({Inputs:input,Output:output}),[1]).submit();const result=await runtime.read(output);
  for(let j=0;j<16;j++){const e=Math.abs(result[j]-f.expected[j]);max=Math.max(max,e);if(!Number.isFinite(result[j])||e>Math.max(.002,Math.abs(f.expected[j])*.00002))throw Error('GPU parity mismatch fixture '+i+' channel '+j+': '+result[j]+' vs '+f.expected[j]);}
  runtime.destroyBuffer(input);runtime.destroyBuffer(output);
 }
 if(errors.length)throw Error(errors.join('\n'));say('PASS '+fixtures.length+' GPU/CPU feature-query fixtures; worst absolute difference '+max);
 const report={passed:true,pipelines:manifest.length,fixtures:fixtures.length,maxDifference:max,adapter:runtime.describe()};runtime.dispose();window.gpuResult=report;return report;
}
const button=document.querySelector('#run');if(button)button.onclick=async()=>{button.disabled=true;try{const r=await runChecks();say(JSON.stringify(r,null,2));}catch(e){window.gpuResult={passed:false,error:String(e.stack||e)};say('FAIL '+e.stack);}finally{button.disabled=false;}};
window.runGPUChecks=runChecks;
