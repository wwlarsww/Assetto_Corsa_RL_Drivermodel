# Documentation and publishing

The documentation has two complementary layers:

- **Project guides** explain architecture, workflows, units and design decisions.
  They are maintained as Markdown files under `docs/`.
- **API reference pages** are generated from Python modules, signatures, type
  annotations and docstrings. They are rebuilt whenever the documentation builds.

All maintained documentation and docstrings should be written in English.

## Build and preview

From an activated project environment:

```powershell
python -m pip install -e ".[docs]"
python -m mkdocs serve
```

Open <http://127.0.0.1:8000>. To validate and build the static site:

```powershell
python -m mkdocs build --strict
```

The output lives under `site/`, which is ignored by Git. API Markdown pages
are generated in a virtual filesystem; no generated reference files need to
be committed. File-based URLs allow the built HTML pages to be opened locally.
Search works best through `mkdocs serve` or the hosted site. Mermaid diagrams
use the theme's browser integration and may require network access to load
the renderer.

The build disables dynamic inspection of application modules. It can run on
Linux without Windows shared memory, Qt, Torch or a running simulator.

## Where to make changes

| Change | Edit |
| --- | --- |
| Explain a workflow or design decision | The corresponding Markdown guide in `docs/` |
| Add a guide to the main navigation | `mkdocs.yml` |
| Document a method's behavior | Its Python docstring and type annotations |
| Add a Python module to the reference | Nothing extra: modules in `src/` and `tools/` are discovered automatically |
| Change reference generation or repository-link handling | `scripts/gen_reference.py` |
| Change dependencies | The `docs` extra in `pyproject.toml` |
| Change validation or deployment | `.github/workflows/docs.yml` |

Keep conceptual explanations in the guides. In docstrings describe the
component's purpose, input/output units, shapes and important failure cases.
Signatures and defaults are extracted from code; avoid copying them into prose
unless a default needs explanation.

Link guides to relevant repository files using relative links, for example
`../src/assetto_corsa_ai/reward.py`. The generator resolves these links to API
pages in the built site. Other linked files, such as the TOML template, receive
generated source views. The original Markdown links continue to work on GitHub.
The strict build rejects missing local links and invalid anchors.

## Mathematical notation

Inline formulas use `$...$`; display equations use `$$` delimiters on
separate lines, with blank lines before and after the block:

```text
The reward at step $t$ is:

$$
r_t = w_p r_{\text{progress},t} + w_l r_{\text{line},t}.
$$
```

The Arithmatex Markdown extension preserves LaTeX expressions for MathJax,
which renders them in the browser. Fractions, Greek symbols and
`\begin{cases} ... \end{cases}` are supported. The browser loads MathJax
from a CDN, so rendering requires network access.

## GitHub Pages

The repository uses a custom GitHub Actions workflow rather than committing
generated HTML to a `gh-pages` branch.

1. In the repository, open **Settings → Pages**.
2. Under **Build and deployment**, set **Source** to **GitHub Actions**.
3. Push documentation changes to the default branch, or manually run the
   **Documentation** workflow from the **Actions** tab.

Pull requests build the site without deploying. Pushes to the default branch
build it in strict mode and deploy the resulting artifact. Only the deployment
job receives `pages: write` and `id-token: write` permissions. No personal token
is stored in the repository.

The configured site address is
<https://wwlarsww.github.io/Assetto_Corsa_RL_Drivermodel/>. The repository's
Actions and Pages settings determine whether deployment is available.

## Tool documentation

- [MkDocs configuration](https://www.mkdocs.org/user-guide/configuration/)
- [Automatic reference generation with mkdocstrings](https://mkdocstrings.github.io/recipes/)
- [Material diagram support](https://squidfunk.github.io/mkdocs-material/reference/diagrams/)
- [GitHub Pages custom workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)
