window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document$.subscribe(() => {
  if (!window.MathJax || typeof MathJax.typesetPromise !== "function") {
    return;
  }
  if (typeof MathJax.typesetClear === "function") {
    MathJax.typesetClear();
  }
  MathJax.typesetPromise().catch((error) => {
    console.error("IINTS documentation formula rendering failed", error);
  });
});
