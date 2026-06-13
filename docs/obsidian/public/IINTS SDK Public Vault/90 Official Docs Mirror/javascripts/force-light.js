(function forceLightPalette() {
  function applyLightPalette() {
    try {
      window.localStorage.removeItem("__palette");
    } catch (error) {
      // Private browsing can block localStorage; the attributes still fix it.
    }

    if (!document.body) {
      return;
    }

    document.body.setAttribute("data-md-color-scheme", "default");
    document.body.setAttribute("data-md-color-primary", "white");
    document.body.setAttribute("data-md-color-accent", "teal");
  }

  applyLightPalette();
  window.addEventListener("DOMContentLoaded", applyLightPalette);
  window.addEventListener("load", applyLightPalette);

  if (typeof document$ !== "undefined") {
    document$.subscribe(applyLightPalette);
  }
})();
