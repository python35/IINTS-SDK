# Vendored three.js

- **Version**: 0.160.1
- **Source**: `node_modules/three/build/three.module.js` (npm package `three`, pinned in `package.json`)
- **License**: MIT (see the upstream project for the full license text: https://github.com/mrdoob/three.js)
- **Modifications**: none -- copied verbatim by `scripts/vendor-three.mjs`.
- **Why vendored**: the app has no bundler and a CSP of `script-src 'self'` with
  no import map, so a same-origin local file is the only way to use three.js
  without a CDN reference or weakening the CSP. Only the addon-free core build
  is used (no OrbitControls, no postprocessing) because three.js's `examples/jsm/*`
  addons import the core via a bare `"three"` specifier, which would need an
  `<script type="importmap">` -- and an inline importmap is inline script
  content, which this CSP (no `'unsafe-inline'`, no hash) does not allow.
- **To update**: bump the `three` version in `package.json`, run `npm install`,
  then `npm run vendor-three`.
