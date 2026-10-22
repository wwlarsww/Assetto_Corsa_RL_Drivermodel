/* Arithmatex protects LaTeX from Markdown; MathJax renders its wrappers. */
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  },
  startup: {
    pageReady() {
      return MathJax.startup.defaultPageReady().then(() => {
        if (typeof document$ === "undefined") return;
        let initialPage = true;
        document$.subscribe(() => {
          if (initialPage) {
            initialPage = false;
            return;
          }
          MathJax.startup.promise = MathJax.startup.promise.then(() => {
            MathJax.typesetClear();
            MathJax.texReset();
            return MathJax.typesetPromise();
          });
        });
      });
    }
  }
};
