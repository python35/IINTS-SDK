#!/usr/bin/env node
// Copies three.js's single-file ESM core build into the frontend as a local,
// same-origin module. The app has no bundler and a CSP of `script-src 'self'`
// with no import map, so three.js cannot be pulled from a CDN or referenced
// via a bare `"three"` specifier -- it has to physically live under
// frontend/ and be imported by relative path.
//
// Run after `npm install` or whenever the pinned `three` version in
// package.json changes: `npm run vendor-three`.

import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const appRoot = dirname(here);

const source = join(appRoot, "node_modules", "three", "build", "three.module.js");
const destDir = join(appRoot, "frontend", "vendor", "three");
const dest = join(destDir, "three.module.js");

const pkg = JSON.parse(
  readFileSync(join(appRoot, "node_modules", "three", "package.json"), "utf8")
);

// Fail loudly rather than vendor something unexpected: an addon-free core
// build must have no bare-specifier imports of its own (three's own
// examples/jsm/* addons import the core via `from "three"`, which needs an
// import map this app's CSP can't support -- see VERSION.md). Loader classes
// inside the core (FileLoader, ImageBitmapLoader, ...) do call fetch(), but
// only when a caller explicitly invokes `.load(url)`; that's expected,
// unrelated to CSP/CDN concerns, and not checked here.
const contents = readFileSync(source, "utf8");
if (/^\s*import\s+.*from\s+["'](?!\.)/m.test(contents)) {
  throw new Error(
    "node_modules/three/build/three.module.js contains a bare-specifier import; " +
      "vendoring it as-is would break under this app's CSP (no import map). " +
      "Inspect the new three.js release before updating the pin."
  );
}

mkdirSync(destDir, { recursive: true });
copyFileSync(source, dest);

const versionNote = `# Vendored three.js

- **Version**: ${pkg.version}
- **Source**: \`node_modules/three/build/three.module.js\` (npm package \`three\`, pinned in \`package.json\`)
- **License**: ${pkg.license} (see the upstream project for the full license text: https://github.com/mrdoob/three.js)
- **Modifications**: none -- copied verbatim by \`scripts/vendor-three.mjs\`.
- **Why vendored**: the app has no bundler and a CSP of \`script-src 'self'\` with
  no import map, so a same-origin local file is the only way to use three.js
  without a CDN reference or weakening the CSP. Only the addon-free core build
  is used (no OrbitControls, no postprocessing) because three.js's \`examples/jsm/*\`
  addons import the core via a bare \`"three"\` specifier, which would need an
  \`<script type="importmap">\` -- and an inline importmap is inline script
  content, which this CSP (no \`'unsafe-inline'\`, no hash) does not allow.
- **To update**: bump the \`three\` version in \`package.json\`, run \`npm install\`,
  then \`npm run vendor-three\`.
`;
writeFileSync(join(destDir, "VERSION.md"), versionNote);

console.log(`Vendored three.js ${pkg.version} -> ${dest}`);
