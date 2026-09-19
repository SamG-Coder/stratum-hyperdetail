import fs from 'node:fs/promises';
const root=new URL('../',import.meta.url),out=new URL('../dist/',import.meta.url);
const manifest=JSON.parse(await fs.readFile(new URL('generated/manifest.json',root),'utf8'));
if(!manifest.some(k=>k.entry==='buildGroupBounds'))throw Error('Refusing to package stale generated shaders: buildGroupBounds missing');
if(!manifest.some(k=>k.entry==='tracePrimary'))throw Error('Refusing to package incomplete generated shaders: tracePrimary missing');
await fs.rm(out,{recursive:true,force:true});await fs.mkdir(out,{recursive:true});
for(const name of ['index.html','style.css','src','generated','kernels','vendor','tests/browser.html','tests/browser.js','tests/probes.json'])await fs.cp(new URL(name,root),new URL(name,out),{recursive:true});
await fs.writeFile(new URL('build.json',out),JSON.stringify({build:'infinite-shared-grammar-1',entries:manifest.map(k=>k.entry),builtAt:new Date().toISOString()},null,2));
console.log('Static WebGPU application copied to dist/ with current generated shaders.');
