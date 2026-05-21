---
hide:
  - toc
---

<div class="home-shell">
  <section class="home-hero">
    <div class="home-brand-row">
      <img src="assets/iints-mark.png" alt="IINTS-AF logo" class="home-brand-mark">
      <span>IINTS-AF SDK</span>
    </div>
    <div class="home-hero-grid">
      <div class="home-hero-copy">
        <p class="home-kicker">Safety-first insulin algorithm research</p>
        <h1>Documentation that starts with what you need to do.</h1>
        <p class="home-lead">
          IINTS-AF helps you simulate virtual patients, validate glucose data,
          study insulin algorithms, and package evidence that other people can inspect.
        </p>
        <div class="home-action-row">
          <a class="home-button home-button-primary" href="./QUICKSTART/">Start in 5 minutes</a>
          <a class="home-button" href="./USER_GUIDE_MAP/">Choose your path</a>
          <a class="home-button" href="./DEVELOPER_PORTAL/">Developer portal</a>
        </div>
      </div>
      <div class="home-summary-panel" aria-label="What the SDK covers">
        <div class="home-summary-item">
          <span class="home-summary-label">Simulate</span>
          <strong>Virtual-patient runs</strong>
        </div>
        <div class="home-summary-item">
          <span class="home-summary-label">Validate</span>
          <strong>MDMP + realism checks</strong>
        </div>
        <div class="home-summary-item">
          <span class="home-summary-label">Deploy</span>
          <strong>Pi, UNO Q, Jetson</strong>
        </div>
        <div class="home-summary-item">
          <span class="home-summary-label">Explain</span>
          <strong>Reports + audit trails</strong>
        </div>
      </div>
    </div>
  </section>

  <section class="home-section">
    <div class="home-section-head">
      <h2>Pick The Route That Matches You</h2>
      <p>Do not read the whole site from top to bottom. Start from the job you have today.</p>
    </div>
    <div class="home-route-grid">
      <a class="home-route-card" href="./QUICKSTART/">
        <span class="home-route-label">New here</span>
        <span class="home-route-title">Get one successful run</span>
        <span class="home-route-body">Install the SDK, verify the environment, and run the first demo without learning every subsystem first.</span>
      </a>
      <a class="home-route-card" href="./WORKFLOWS/">
        <span class="home-route-label">Researcher</span>
        <span class="home-route-title">Run a defensible study</span>
        <span class="home-route-body">Move from scenario design to comparisons, plots, reports, and evidence you can hand to a reviewer.</span>
      </a>
      <a class="home-route-card" href="./MDMP_QUICKSTART/">
        <span class="home-route-label">Data quality</span>
        <span class="home-route-title">Check whether data is trustworthy</span>
        <span class="home-route-body">Certify datasets, inspect realism, and keep claims linked to provenance instead of pretty-looking curves.</span>
      </a>
      <a class="home-route-card" href="./HARDWARE/">
        <span class="home-route-label">Hardware</span>
        <span class="home-route-title">Deploy on an edge device</span>
        <span class="home-route-body">Choose between Raspberry Pi, Arduino UNO Q, or Jetson and follow the simplest path for that board.</span>
      </a>
      <a class="home-route-card" href="./DEVELOPER_PORTAL/">
        <span class="home-route-label">Developer</span>
        <span class="home-route-title">Change the SDK safely</span>
        <span class="home-route-body">Find architecture, generated API docs, contribution rules, and the checks required before you merge code.</span>
      </a>
    </div>
  </section>

  <section class="home-section">
    <div class="home-section-head">
      <h2>The Core Workflow</h2>
      <p>Most SDK work is the same four-step story.</p>
    </div>
    <div class="home-flow-grid">
      <div class="home-flow-step">
        <span>01</span>
        <strong>Configure</strong>
        <p>Choose a patient, scenario, algorithm, and seed.</p>
      </div>
      <div class="home-flow-step">
        <span>02</span>
        <strong>Run</strong>
        <p>Generate a reproducible simulation bundle.</p>
      </div>
      <div class="home-flow-step">
        <span>03</span>
        <strong>Validate</strong>
        <p>Check data quality, safety, and realism.</p>
      </div>
      <div class="home-flow-step">
        <span>04</span>
        <strong>Review</strong>
        <p>Inspect reports, manifests, plots, and audit trails.</p>
      </div>
    </div>
  </section>

  <section class="home-section">
    <div class="home-section-head">
      <h2>Three Useful Starting Points</h2>
      <p>If you are unsure where to begin, these are the safest bets.</p>
    </div>
    <div class="home-card-grid">
      <a class="home-card" href="./GETTING_STARTED/">
        <span class="home-card-title">Getting Started</span>
        <span class="home-card-body">The shortest reliable route from installation to a full run bundle.</span>
      </a>
      <a class="home-card" href="./COMMAND_REFERENCE/">
        <span class="home-card-title">Command Reference</span>
        <span class="home-card-body">A compact CLI map when you already know the result you want.</span>
      </a>
      <a class="home-card" href="./REFERENCE_OVERVIEW/">
        <span class="home-card-title">Reference Hub</span>
        <span class="home-card-body">Manuals, evidence, API docs, commands, and release history in one place.</span>
      </a>
    </div>
  </section>

  <section class="home-section home-command-section">
    <div class="home-section-head">
      <h2>First Commands</h2>
      <p>The smallest useful sequence for a fresh machine.</p>
    </div>
    <pre class="home-command"><code>python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -U "iints-sdk-python35[full,mdmp]"

iints doctor --smoke-run
iints demo</code></pre>
  </section>

  <p class="home-scope"><strong>Scope:</strong> research use only. IINTS-AF is not a medical device and does not provide clinical dosing advice.</p>
</div>
