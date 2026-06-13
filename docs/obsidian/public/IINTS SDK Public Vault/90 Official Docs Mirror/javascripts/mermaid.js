function initMermaid() {
  if (typeof window.mermaid === "undefined") {
    return;
  }
  const diagrams = document.querySelectorAll(".mermaid");
  if (!diagrams.length) {
    return;
  }
  window.mermaid.initialize({
    startOnLoad: false,
    securityLevel: "loose",
    theme: "default",
  });
  window.mermaid.run({
    nodes: Array.from(diagrams),
  });
}

window.addEventListener("load", initMermaid);

if (typeof document$ !== "undefined") {
  document$.subscribe(initMermaid);
}
