---
hide:
  - toc
---

<div class="home-shell">
  <section class="home-hero">
    <div class="home-brand-row">
      <img src="assets/iints-logo.png" alt="IINTS-AF logo" class="home-brand-mark">
      <span>IINTS-AF SDK</span>
    </div>
    <div class="home-hero-grid">
      <div class="home-hero-copy">
        <h1>IINTS-AF SDK Documentation</h1>
        <p class="home-lead">
          Research tooling for virtual-patient insulin simulations, data quality
          certification, edge studies, and reproducible review bundles.
        </p>
        <div class="home-action-row">
          <a class="home-button home-button-primary" href="./QUICKSTART/">Start with Quickstart</a>
          <a class="home-button" href="./USER_GUIDE_MAP/">Open Guide Map</a>
          <a class="home-button" href="./COMMAND_REFERENCE/">Browse CLI</a>
        </div>
      </div>
      <div class="home-summary-panel" aria-label="Documentation highlights">
        <div class="home-summary-item">
          <span class="home-summary-icon home-icon-workflow" aria-hidden="true"></span>
          <div>
            <span class="home-summary-label">Research workflow</span>
            <strong>Simulation to report</strong>
          </div>
        </div>
        <div class="home-summary-item">
          <span class="home-summary-icon home-icon-quality" aria-hidden="true"></span>
          <div>
            <span class="home-summary-label">Data quality</span>
            <strong>MDMP and realism checks</strong>
          </div>
        </div>
        <div class="home-summary-item">
          <span class="home-summary-icon home-icon-edge" aria-hidden="true"></span>
          <div>
            <span class="home-summary-label">Edge hardware</span>
            <strong>Pi, UNO Q, Jetson</strong>
          </div>
        </div>
      </div>
    </div>
  </section>

  <section class="home-section">
    <div class="home-section-head">
      <h2>Getting Started</h2>
      <p>Choose the fastest entry point for what you need right now.</p>
    </div>
    <div class="home-card-grid">
      <a class="home-card" href="./QUICKSTART/">
        <span class="home-card-icon home-icon-quickstart" aria-hidden="true"></span>
        <span class="home-card-title">5-Minute Quickstart</span>
        <span class="home-card-body">Install the SDK, run a first demo, and confirm your environment is ready.</span>
      </a>
      <a class="home-card" href="./USER_GUIDE_MAP/">
        <span class="home-card-icon home-icon-map" aria-hidden="true"></span>
        <span class="home-card-title">User Guide Map</span>
        <span class="home-card-body">A connected route through beginner, research, data, edge, and maintainer docs.</span>
      </a>
      <a class="home-card" href="./TROUBLESHOOTING/">
        <span class="home-card-icon home-icon-tools" aria-hidden="true"></span>
        <span class="home-card-title">Troubleshooting</span>
        <span class="home-card-body">Common failures, exact fixes, and recovery steps for local installs.</span>
      </a>
      <a class="home-card" href="./COMMAND_REFERENCE/">
        <span class="home-card-icon home-icon-cli" aria-hidden="true"></span>
        <span class="home-card-title">Command Reference</span>
        <span class="home-card-body">A compact map of the CLI surface when you already know what you want to run.</span>
      </a>
    </div>
  </section>

  <section class="home-section">
    <div class="home-section-head">
      <h2>Research Paths</h2>
      <p>Move from a first run to evidence you can inspect and share.</p>
    </div>
    <div class="home-path-grid">
      <a class="home-path" href="./SCIENTIFIC_WORKFLOW/">
        <span class="home-path-index">01</span>
        <span class="home-path-title">Scientific Workflow</span>
        <span class="home-path-body">Study design, benchmark runs, comparisons, and reproducible outputs.</span>
      </a>
      <a class="home-path" href="./MDMP_QUICKSTART/">
        <span class="home-path-index">02</span>
        <span class="home-path-title">Data Quality</span>
        <span class="home-path-body">Certification, trust grading, realism checks, and dataset review.</span>
      </a>
      <a class="home-path" href="./EDGE_HARDWARE/">
        <span class="home-path-index">03</span>
        <span class="home-path-title">Edge Deployment</span>
        <span class="home-path-body">Raspberry Pi, Arduino UNO Q, Jetson endurance, and booth setup paths.</span>
      </a>
    </div>
  </section>

  <section class="home-section home-command-section">
    <div class="home-section-head">
      <h2>First Commands</h2>
      <p>The shortest useful sequence for a fresh environment.</p>
    </div>
    <pre class="home-command"><code>python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -U "iints-sdk-python35[full,mdmp]"

iints doctor --suggest
iints start
iints demo</code></pre>
  </section>

  <p class="home-scope"><strong>Scope:</strong> research use only. IINTS-AF is not a medical device and does not provide clinical dosing advice.</p>
</div>
