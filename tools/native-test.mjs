import {spawnSync} from 'node:child_process';import os from 'node:os';import path from 'node:path';
const binary=path.join(os.tmpdir(),'stratum-native-'+process.pid+(process.platform==='win32'?'.exe':''));
const compiler=process.env.CXX||'g++';let r=spawnSync(compiler,['-std=c++17','-O2','tests/native_harness.cpp','-o',binary],{stdio:'inherit'});
if(r.error)throw Error('C++ compiler required for native parity tests. Install g++ or set CXX; use npm run test:source for source-only checks.');if(r.status!==0)process.exit(r.status||1);
r=spawnSync(binary,[],{stdio:'inherit'});process.exit(r.status||0);
