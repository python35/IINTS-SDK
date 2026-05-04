---
hide:
  - toc
---

<div class="home-shell">
  <section class="home-hero">
    <div class="home-hero-copy">
      <div class="home-kicker">IINTS-AF SDK</div>
      <h1>Welcome to the IINTS-AF Documentation</h1>
      <p class="home-lead">
        Safety-first simulation, data certification, and reproducible insulin
        algorithm research in one open research platform.
      </p>
      <div class="home-pill-row">
        <a class="home-pill home-pill-primary" href="./QUICKSTART/">Getting Started</a>
        <a class="home-pill" href="./USER_GUIDE_MAP/">Docs Map</a>
        <a class="home-pill" href="./COMMAND_REFERENCE/">CLI Guide</a>
      </div>
      <div class="home-meta-row">
        <div class="home-meta-card">
          <span class="home-meta-label">Best for</span>
          <strong>Researchers, evaluators, edge builders</strong>
        </div>
        <div class="home-meta-card">
          <span class="home-meta-label">Focus</span>
          <strong>Simulation, trust grading, publication-ready review</strong>
        </div>
      </div>
    </div>
    <div class="home-hero-art" aria-hidden="true">
      <div class="home-logo-wrap">
        <img src="assets/iints-logo.png" alt="IINTS logo" class="home-logo-mark">
      </div>
      <div class="home-orbit home-orbit-one"></div>
      <div class="home-orbit home-orbit-two"></div>
      <div class="home-orbit home-orbit-three"></div>
    </div>
  </section>

  <section class="home-section">
    <div class="home-section-head">
      <h2>Start Here</h2>
      <p>Pick the fastest route based on what you want to achieve today.</p>
    </div>
    <div class="home-card-grid">
      <a class="home-card" href="./QUICKSTART/">
        <span class="home-card-title">5-Minute Quickstart</span>
        <span class="home-card-body">The shortest path to a successful first run with `iints demo`.</span>
      </a>
      <a class="home-card" href="./USER_GUIDE_MAP/">
        <span class="home-card-title">User Guide Map</span>
        <span class="home-card-body">A connected route through quickstart, research, edge, data, and maintenance docs.</span>
      </a>
      <a class="home-card" href="./TROUBLESHOOTING/">
        <span class="home-card-title">Troubleshooting</span>
        <span class="home-card-body">Exact fixes for broken installs, missing dependencies, and failing commands.</span>
      </a>
      <a class="home-card" href="./COMMAND_REFERENCE/">
        <span class="home-card-title">Command Reference</span>
        <span class="home-card-body">A concise map of the CLI surface when you want to move fast without digging through internals.</span>
      </a>
    </div>
  </section>

  <section class="home-section">
    <div class="home-section-head">
      <h2>Main Workflows</h2>
      <p>The SDK revolves around three practical tracks.</p>
    </div>
    <div class="home-workflow-grid">
      <div class="home-workflow-card">
        <span class="home-workflow-index">01</span>
        <h3>Simulate</h3>
        <p>Run virtual-patient studies, compare algorithms, and inspect safety behavior under controlled scenarios.</p>
        <a href="./SCIENTIFIC_WORKFLOW/">Open scientific workflow</a>
      </div>
      <div class="home-workflow-card">
        <span class="home-workflow-index">02</span>
        <h3>Certify</h3>
        <p>Check realism, trust grading, and output quality with MDMP-oriented validation and dataset tooling.</p>
        <a href="./MDMP_QUICKSTART/">Open MDMP quickstart</a>
      </div>
      <div class="home-workflow-card">
        <span class="home-workflow-index">03</span>
        <h3>Deploy</h3>
        <p>Prepare Raspberry Pi, Jetson, and booth-ready edge flows for demos or long-running stress studies.</p>
        <a href="./EDGE_HARDWARE/">Open edge guide</a>
      </div>
    </div>
  </section>

  <section class="home-section">
    <div class="home-section-head">
      <h2>Quick Commands</h2>
      <p>Good defaults when you want the CLI to lead.</p>
    </div>
    <div class="home-command-panel">
      <pre><code>python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -U "iints-sdk-python35[full,mdmp]"

iints doctor --suggest
iints start
iints demo</code></pre>
    </div>
  </section>

  <section class="home-footer-note">
    <p><strong>Scope:</strong> research use only, not a medical device, and not clinical dosing advice.</p>
  </section>
</div>
