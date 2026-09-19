# STRATUM Hyperdetail — Infinite

**[Open the city](https://samg-coder.github.io/stratum-hyperdetail/)** · **[Run real-device GPU checks](https://samg-coder.github.io/stratum-hyperdetail/tests/browser.html)**

An endlessly traversable seeded city. **The original detailed building grammar is the geometry everywhere.** There is no near/far building-model switch, no macro-box fallback, and no 256-building island. No imported meshes, texture packs, pretrained models, or API calls.

## What changed

The former page/query refactor did not compile (`sinkWall` passed an undefined `hit`), discarded hit materials/normals, and changed some single-statement `if` scopes. This version uses a typed, value-returning sink. A feature retains its position, dimensions, analytic shape, orientation, material, seed and identity.

`kernels/assets.cu` supplies the **same exact feature emissions** to three consumers: direct ray intersection, conservative bounding hierarchy construction, and reference enumeration. Roofs remain triangular prisms; domes remain ellipsoids; cylinders keep both caps; arches/rings and procedural leaves retain their original geometry. Windows, mullions, sills, railings, dormers, chimneys, cornices, street furniture, fountains and bridges are not replaced by painted far-field windows.

The old 32-feature group cap is gone. The existing hyperdetail additions now survive instead of silently being dropped. Tests check that the first 32 records of each group match the frozen original renderer and that all group counts fit the 64-feature identity range.

## Endless world, bounded work

The camera uses **64-bit integer lot addresses** (two uint32 words per axis) and lot-local floating-point coordinates. Walking past the former 64×64 city edge generates new deterministic lots. Revisiting a location reconstructs its seed. Coordinates wrap only after the 64-bit address range; this is not a claim of literally unlimited numerical precision.

The visible range is **2,000 metres**, with a smooth atmospheric fade. Infinite exploration does not mean infinitely many objects can be drawn in one frame.

A scrolling 128×128 table stores descriptors and **64-group bounding hierarchies only**. The hierarchy allocation is 64 MiB; frame buffers are additional. No `world lots × 2,048 primitives` allocation exists. Entering a new lot refreshes affected hierarchy records, not the geometric representation of the city. Every ray intersects the original features of groups whose bounds it actually reaches.

## Detail and variation

There are no distance-selected geometry models. The visibility hierarchy rejects groups a ray cannot intersect. Fine material frequencies filter continuously with the projected/grazing-angle footprint. Subpixel geometric coverage uses a deterministic 64-sample stationary sequence, then stops updating: the previous endless random jitter is removed. Fast moving views can still alias; this is not motion-vector temporal reconstruction.

Lot seeds and district seeds vary widths, depths, storey counts/heights, orientations, small setbacks, roof rise and facade palettes. The original pitched-roof, dome, tower, park and canal families remain, with additional slatted benches and shop awnings. The architectural vocabulary is finite and will repeat; there are not infinitely many independently authored building styles.

Lighting uses procedural materials, building-mass sunlight occlusion, sky reflection and screen-space contact shading. **It is not full fine-geometry path tracing.** The previous neural material experiment is not part of the active renderer.

## Run

Windows: double-click `START.bat`. macOS/Linux: `./start.sh`.

Or:

```sh
npm run build
npm start
```

Open `http://localhost:8089`. Node.js 20+ and a working WebGPU browser are required. No npm package installation is needed for the app. The launcher rebuilds shaders so old generated WGSL cannot silently run instead of the new code.

Desktop starts at 1600 internal pixels wide; 1920, 1024 and 768 remain selectable. This is a quality setting, not a measured performance guarantee.

WASD moves, drag looks, Q/E changes height, Shift boosts, Ctrl slows, and the wheel changes travel speed. 1–6 select camera bookmarks. P enables a slow flight tour. V cycles lit/district/group/normal inspection. J toggles building shadows. H opens field notes, F toggles fullscreen, C hides the interface, K saves the computed frame.

## Build and verify

```sh
npm test
npm run pages
```

`npm test` first translates the CUDA entry points, checks generated-source and host/binding contracts, then compiles/runs the native C++ parity suite. It requires `g++` or `CXX` for the native checks. `npm run test:source` runs only the source/runtime-contract checks after building.

Build output is staged: an unsuccessful compile leaves the previous complete generated directory intact rather than shipping a partially updated shader set. GitHub Actions validates before deploying.

The frozen original emitter is under `tests/reference/`. Native checks cover original feature/material identity, hierarchy-vs-exhaustive ray hits, caps/rings, 64-bit coordinate carry/borrow, deterministic revisits, floating-origin rebasing, variation and stationary convergence.

`tests/browser.html` creates the actual WebGPU pipelines and runs 54 shader/CPU geometry fixtures. It reports the real adapter and fails on mismatches; mock tests are not presented as GPU validation.

### Validation of this build

See `docs/VALIDATION_INFINITE.md` and logs. CUDA translation, source/runtime-contract tests, native geometry tests and a CPU reference render were executed. Browser GPU execution could not be completed in the build environment: localhost navigation was blocked by browser policy. **RTX 5080 frame rate, Windows driver stability and real-browser image quality remain unmeasured.** The saved preview is explicitly a CPU reference render, not an image-generation result or a GPU benchmark.

## Source map

- `common.cu`, `world.cu`: integer coordinates, world descriptors, floating camera and stable sampling.
- `geometry.cu`: original analytic shape intersections and normals.
- `sink.cu`, `assets.cu`: the shared full-detail feature grammar and sinks.
- `accel.cu`, `trace.cu`: bound-only acceleration and exact ray queries.
- `materials.cu`, `shade.cu`: continuous material filtering and lighting.
- `src/engine.js`: resource limits, bounded submissions, indirect refresh, measurements and presentation.

The generated `Stratum.cu` combines the maintained split files. WGSL is compiler output. Vendored CUDA WebShader licences and notices are unchanged. The browser host/interface is JavaScript/HTML/CSS; `.cu` authors the world and rendering, rather than hardware CUDA executing inside a browser.
