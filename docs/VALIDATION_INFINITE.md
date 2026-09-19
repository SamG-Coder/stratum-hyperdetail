# Infinite shared-grammar validation

## Baseline reproduced before changing the renderer

The relevant repository files were reproduced byte-for-byte at main commit `bfa6ee47e195d2f6574de3d331abcae7324fd2b2`. The assets blob was `60b347e1c84fc492f0a38e9241d65aa2a0398d0a`; the vendored compiler blob was `2597dbb460926d752b950c40cfc7a27ad0f7f194`.

The baseline compile failed with `Unknown identifier 'hit'` in `sinkWall`. Its attempted sink refactor also discarded feature material/normal information and changed the scope of unbraced statements. The new implementation does not depend on those broken sink functions.

## Executed checks

`npm test` first rebuilds the generated shaders, then runs the source/runtime contract tests and native C++ tests. This build passed:

- All 11 active CUDA entry points translated using the actual vendored compiler. The generated WGSL matches a second source compilation. Storage bindings and workgroup scratch stay within the project's portable limits.
- Four Node tests, including host binding/resource agreement, indirect dispatch wiring, resize/reset/dispose and presentation with a device double. This is not hardware execution.
- 13,857 feature records compared with the original detailed generator. Original feature geometry, material, rotation and seed were retained. The shared grammar additionally emits features formerly lost to the 32-record per-cluster cap; the tested maximum group size is 56.
- 2,304 accelerated ray queries compared with exhaustive feature intersection over all building families; 1,952 hit geometry. Maximum hit-distance difference: zero for this test set.
- Independent cylinder top-cap and full-ring/lower-half regression checks.
- 64-bit coordinate carry/borrow, deterministic reconstruction after revisiting distant coordinates, floating-origin rebasing, stationary sample convergence, and architectural/palette variation.
- Native tests with AddressSanitizer and UndefinedBehaviorSanitizer: passed without reported sanitizer failures.
- `npm run pages`: passed.

These checks validate the tested inputs. They are not a proof of all views, all seeds or cross-driver equivalence.

## Visual reference and GPU limitation

A 480 x 300, four-sample CPU reference view was rendered from the same authored CUDA functions and visually inspected. It contains detailed roofs, dormers, chimneys, facades, balconies and park geometry across the view. It is not an AI-generated image or an RTX/browser performance measurement.

Real browser execution was attempted but navigation to the local test server was blocked by the environment (`ERR_BLOCKED_BY_ADMINISTRATOR`). Hardware GPU frame rate, driver stability and shader compilation on the user's adapter are therefore unverified. The shipped `tests/browser.html` compiles the active shaders and offers 54 CPU/GPU feature/ray parity fixtures for a real adapter; those fixtures were not executed here.

## Meaning of infinite and LOD

The world is endlessly traversable within a two-word, 64-bit coordinate space per horizontal axis, not mathematically unbounded. A floating origin keeps camera and feature coordinates small. The visible ray range is 2,000 metres with atmospheric fade. A rolling 128 x 128 descriptor/bounds window covers that range; it is not a finite city boundary.

Every visible building uses the same full authored feature grammar. There is no cheap near/far replacement. Bounds reject missed feature groups; smooth footprint filtering attenuates unresolvable material and relief frequencies. This does not claim a pixel integrates every subpixel geometric feature perfectly. Lighting still uses approximate building-mass shadows and sky reflections, not full path tracing.

## Reproduce

```sh
npm test
npm run pages
```

Native tests require a C++17 compiler (`g++` by default, or `CXX`). Browser rendering needs Node.js 20+ for the local server and WebGPU over localhost/HTTPS. No npm package installation is required for the compiler/runtime.
