/**
 * Render an HTML file to PDF with headless Chrome over CDP.
 *
 * Waits for MathJax and webfonts before printing, streams the PDF back
 * instead of pulling several megabytes through a single CDP message, and
 * asks Chrome for a document outline so the PDF gets real bookmarks.
 *
 * usage: node render_pdf.mjs <input.html> <output.pdf>
 */
import { spawn } from "node:child_process";
import { writeFile, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import WebSocket from "ws";

// Any Chromium build works. Override with CHROME_BIN; the default is the
// Chrome for Testing that puppeteer caches.
import { existsSync, readdirSync } from "node:fs";
import { homedir } from "node:os";

function findChrome() {
  if (process.env.CHROME_BIN) return process.env.CHROME_BIN;
  const cache = join(homedir(), ".cache", "puppeteer", "chrome");
  if (existsSync(cache)) {
    for (const rev of readdirSync(cache).sort().reverse()) {
      for (const sub of ["chrome-mac-arm64", "chrome-mac-x64", "chrome-linux64"]) {
        for (const tail of [
          "Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",
          "chrome",
        ]) {
          const p = join(cache, rev, sub, tail);
          if (existsSync(p)) return p;
        }
      }
    }
  }
  for (const p of [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
  ]) if (existsSync(p)) return p;
  throw new Error("No Chromium found. Set CHROME_BIN to a Chrome/Chromium binary.");
}

const CHROME = findChrome();

const [input, output] = process.argv.slice(2);
if (!input || !output) {
  console.error("usage: node render_pdf.mjs <input.html> <output.pdf>");
  process.exit(1);
}

const PORT = 9333 + Math.floor(Math.random() * 400);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const profile = await mkdtemp(join(tmpdir(), "iints-pdf-"));
const chrome = spawn(CHROME, [
  "--headless=new",
  "--disable-gpu",
  "--no-first-run",
  "--no-default-browser-check",
  "--hide-scrollbars",
  "--force-color-profile=srgb",
  "--font-render-hinting=none",
  `--user-data-dir=${profile}`,
  `--remote-debugging-port=${PORT}`,
  "about:blank",
], { stdio: ["ignore", "ignore", "pipe"] });

let chromeErr = "";
chrome.stderr.on("data", (d) => { chromeErr += d.toString(); });

async function devtools(path) {
  for (let i = 0; i < 80; i++) {
    try {
      const res = await fetch(`http://127.0.0.1:${PORT}${path}`);
      if (res.ok) return await res.json();
    } catch { /* not up yet */ }
    await sleep(150);
  }
  throw new Error(`Chrome DevTools never came up on ${PORT}\n${chromeErr}`);
}

const version = await devtools("/json/version");
const ws = new WebSocket(version.webSocketDebuggerUrl, {
  maxPayload: 512 * 1024 * 1024,
  perMessageDeflate: false,
});
await new Promise((res, rej) => { ws.once("open", res); ws.once("error", rej); });

let nextId = 1;
const pending = new Map();
const events = [];
ws.on("message", (raw) => {
  const msg = JSON.parse(raw.toString());
  if (msg.id && pending.has(msg.id)) {
    const { resolve, reject } = pending.get(msg.id);
    pending.delete(msg.id);
    msg.error ? reject(new Error(`${msg.error.message} (${JSON.stringify(msg.error.data ?? "")})`)) : resolve(msg.result);
  } else if (msg.method) {
    events.push(msg.method);
  }
});

function send(method, params = {}, sessionId) {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    ws.send(JSON.stringify({ id, method, params, sessionId }));
  });
}

// One tab, driven through a flat session.
const { targetId } = await send("Target.createTarget", { url: "about:blank" });
const { sessionId } = await send("Target.attachToTarget", { targetId, flatten: true });
const call = (m, p) => send(m, p, sessionId);

await call("Page.enable");
await call("Runtime.enable");
await call("Emulation.setEmulatedMedia", { media: "print" });

const url = pathToFileURL(input).href;
await call("Page.navigate", { url });

async function evaluate(expression) {
  const { result } = await call("Runtime.evaluate", {
    expression, returnByValue: true, awaitPromise: true,
  });
  return result.value;
}

// Load, then MathJax, then fonts. Each is a separate failure mode, so each
// gets its own wait and its own reported state.
for (let i = 0; i < 200; i++) {
  if (await evaluate("document.readyState === 'complete'")) break;
  await sleep(150);
}

let mathReady = false;
for (let i = 0; i < 300; i++) {
  mathReady = await evaluate("document.documentElement.classList.contains('math-ready')");
  if (mathReady) break;
  await sleep(200);
}

// Images must actually be decoded before printing. A figure that is still
// loading prints as an empty plate, and nothing else reports that.
let imagesReady = false;
for (let i = 0; i < 300; i++) {
  imagesReady = await evaluate(
    "[...document.images].every(im => im.complete && im.naturalWidth > 0)"
  );
  if (imagesReady) break;
  await sleep(200);
}

await evaluate("document.fonts.ready.then(() => true)");
await sleep(700);

// A silently-unavailable webfont renders the whole document in fallback faces
// and changes nothing else that a checker would notice, so assert it.
const fontReport = await evaluate(`JSON.stringify({
  loaded: document.fonts.size,
  Archivo: document.fonts.check('700 16px Archivo'),
  SourceSerif: document.fonts.check('400 16px "Source Serif 4"'),
  JetBrainsMono: document.fonts.check('400 16px "JetBrains Mono"'),
  status: [...document.fonts].slice(0, 3).map(f => f.family + ':' + f.status)
})`);
const fr = JSON.parse(fontReport);
console.log(`  fonts: ${fr.loaded} faces  Archivo=${fr.Archivo} SourceSerif=${fr.SourceSerif} JetBrainsMono=${fr.JetBrainsMono}`);
if (fr.status.length) console.log(`  face status sample: ${fr.status.join(", ")}`);
if (!fr.Archivo || !fr.SourceSerif || !fr.JetBrainsMono) {
  console.error("  ERROR: intended typefaces unavailable; PDF would use fallbacks");
  process.exitCode = 3;
}

const stats = await evaluate(`JSON.stringify({
  mjx: document.querySelectorAll('mjx-container').length,
  imgs: document.querySelectorAll('.plate img').length,
  toc: document.querySelectorAll('.pdftoc__row').length,
  chapters: document.querySelectorAll('.chapter').length
})`);
console.log(`  page ready: math=${mathReady} images=${imagesReady} ${stats}`);
if (!mathReady) console.warn("  WARNING: MathJax never signalled ready");
if (!imagesReady) {
  const broken = await evaluate(
    "JSON.stringify([...document.images].filter(im => !im.complete || !im.naturalWidth).map(im => im.alt))"
  );
  console.error(`  ERROR: images not decoded: ${broken}`);
  process.exitCode = 2;
}

const footer = `
<div style="width:100%;font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;
     font-size:7.5px;color:#6d808c;padding:0 15mm;display:flex;
     justify-content:space-between;align-items:center;">
  <span style="letter-spacing:.06em;">IINTS-AF MODEL COMPANION &middot; SDK v1.5.36 &middot; NOT A MEDICAL DEVICE</span>
  <span style="font-variant-numeric:tabular-nums;">
    <span class="pageNumber"></span> / <span class="totalPages"></span>
  </span>
</div>`;

const figSizes = await evaluate(
  "JSON.stringify([...document.querySelectorAll('.plate img')]" +
  ".map(im => Math.round(im.getBoundingClientRect().width) + 'x' + " +
  "Math.round(im.getBoundingClientRect().height)))"
);
const zeroSized = JSON.parse(figSizes).filter((s) => s.startsWith("0x") || s.endsWith("x0"));
console.log(`  figures laid out: ${JSON.parse(figSizes).length}, zero-sized: ${zeroSized.length}`);
if (zeroSized.length) { console.error("  ERROR: zero-sized figures present"); process.exitCode = 2; }

const printParams = {
  landscape: false,
  printBackground: true,
  paperWidth: 8.27,
  paperHeight: 11.69,
  marginTop: 0,
  marginBottom: 0,
  marginLeft: 0,
  marginRight: 0,
  preferCSSPageSize: true,
  displayHeaderFooter: true,
  headerTemplate: "<span></span>",
  footerTemplate: footer,
  transferMode: "ReturnAsStream",
  generateDocumentOutline: true,
};

let result;
try {
  result = await call("Page.printToPDF", printParams);
} catch (err) {
  console.warn(`  outline unsupported (${err.message}); retrying without it`);
  delete printParams.generateDocumentOutline;
  result = await call("Page.printToPDF", printParams);
}

// Pull the stream in chunks rather than one huge base64 blob.
const parts = [];
let eof = false;
while (!eof) {
  const chunk = await call("IO.read", { handle: result.stream, size: 4 * 1024 * 1024 });
  if (chunk.data) parts.push(Buffer.from(chunk.data, chunk.base64Encoded ? "base64" : "utf8"));
  eof = chunk.eof;
}
await call("IO.close", { handle: result.stream });

const pdf = Buffer.concat(parts);
await writeFile(output, pdf);
console.log(`  wrote ${output} (${(pdf.length / 1048576).toFixed(2)} MB)`);

ws.close();
chrome.kill("SIGTERM");
await sleep(300);
await rm(profile, { recursive: true, force: true });
process.exit(0);
