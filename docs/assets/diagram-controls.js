/* Render generated diagrams independently of the theme's closed shadow DOM. */
(() => {
  let renderer;
  let sequence = 0;

  function mount(diagram, svg) {
    const { width, height } = svg.viewBox.baseVal;
    if (!width || !height) return;
    const section = document.createElement("section");
    section.className = "diagram-view";
    section.setAttribute("aria-label", "Class dependency diagram");
    const toolbar = document.createElement("div");
    toolbar.className = "diagram-toolbar";
    const viewport = document.createElement("div");
    viewport.className = "diagram-viewport";
    viewport.tabIndex = 0;
    viewport.setAttribute("aria-label", "Scrollable class diagram");
    diagram.before(section);
    section.append(toolbar, viewport);
    viewport.append(diagram);
    const status = document.createElement("span");
    status.setAttribute("aria-live", "polite");
    let scale = 1;
    const update = (value) => {
      scale = Math.min(4, Math.max(0.05, value));
      svg.style.maxWidth = "none";
      svg.style.width = `${width * scale}px`;
      svg.style.height = `${height * scale}px`;
      status.textContent = `${Math.round(scale * 100)}%`;
    };
    const button = (label, handler) => {
      const element = document.createElement("button");
      element.type = "button";
      element.textContent = label;
      element.addEventListener("click", handler);
      toolbar.append(element);
    };
    const fit = () => {
      const availableHeight = parseFloat(getComputedStyle(viewport).maxHeight);
      update(Math.min(1, (viewport.clientWidth - 16) / width, (availableHeight - 16) / height));
    };
    button("Zoom in", () => update(scale * 1.3));
    button("Zoom out", () => update(scale / 1.3));
    button("Fit", fit);
    button("100%", () => update(1));
    button("Export SVG", () => {
      const copy = svg.cloneNode(true);
      copy.style.width = `${width}px`;
      copy.style.height = `${height}px`;
      const blob = new Blob([new XMLSerializer().serializeToString(copy)], { type: "image/svg+xml" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "class-diagram.svg";
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    });
    toolbar.append(status);
    fit();
  }

  function enhance() {
    document.querySelectorAll("pre.class-diagram-source").forEach(async (source) => {
      if (source.dataset.renderStarted) return;
      source.dataset.renderStarted = "true";
      try {
        renderer ||= import("https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs").then((module) => {
          module.default.initialize({ startOnLoad: false, securityLevel: "strict", theme: "default" });
          return module.default;
        });
        const mermaid = await renderer;
        const rendered = await mermaid.render("classDiagram" + sequence++, source.textContent);
        if (!source.isConnected) return;
        const diagram = document.createElement("div");
        diagram.className = "class-diagram";
        diagram.innerHTML = rendered.svg;
        source.replaceWith(diagram);
        if (rendered.bindFunctions) rendered.bindFunctions(diagram);
        mount(diagram, diagram.querySelector("svg"));
      } catch (error) {
        const message = document.createElement("p");
        message.textContent = "The diagram renderer could not load. The diagram source remains available below.";
        message.setAttribute("role", "status");
        source.before(message);
        console.error("Class diagram rendering failed:", error);
      }
    });
  }

  const observer = new MutationObserver(enhance);
  observer.observe(document.body, { childList: true, subtree: true });
  if (typeof document$ !== "undefined") document$.subscribe(enhance);
  enhance();
})();
