import fs from 'node:fs/promises';
const root=new URL('../',import.meta.url),out=new URL('../dist/',import.meta.url);
await fs.rm(out,{recursive:true,force:true});await fs.mkdir(out,{recursive:true});
for(const name of ['index.html','style.css','src','kernels','vendor','tests/browser.html','tests/browser.js','tests/probes.json'])await fs.cp(new URL(name,root),new URL(name,out),{recursive:true});
await fs.writeFile(new URL('build.json',out),JSON.stringify({build:'runtime-cuda-compiler-1',shaderSource:'kernels/*.cu',generatedShaderDependency:false,builtAt:new Date().toISOString()},null,2));
console.log('Static runtime-CUDA application copied to dist/. generated/ is intentionally not deployed.');
