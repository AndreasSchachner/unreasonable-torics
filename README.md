# Toric methods for Calabi–Yau compactifications and the string landscape

Lecture notes for the ESI 2026 programme
*The Unreasonable Effectiveness of Toric Geometry: Bridging Mathematics, Computation, and String Theory*.

**Author:** Andreas Schachner
**Lectures:** 22–24 June 2026

> ⚠️ **Work in progress.** These notes are an actively maintained draft. Errors, imprecisions, missing references, and gaps remain throughout. **Feedback, corrections, and suggestions are very welcome** — please open an issue or send a note. The notes will continue to evolve after the workshop.

## Scope

Three 75-minute lectures aimed at mathematicians fluent in algebraic and toric geometry, with physics background introduced inline as needed:

- **Lecture 1.** String compactifications, Calabi–Yau threefolds, and why toric hypersurfaces give a computable class of examples.
- **Lecture 2.** Non-perturbative physics from toric data: Type IIB orientifold EFT, fluxes, tadpoles, periods, Gopakumar–Vafa invariants, rigid divisors, and perturbatively flat vacua.
- **Lecture 3.** Vacuum energies from toric geometry: candidate leading-order de Sitter vacua, control conditions, and the vacuum search algorithm.

Lecture 4 (`fanroots`, root-finding over secondary fans) is a separate standalone talk for a computational-algebraic-geometry audience and lives behind an opt-in TeX switch in `main.tex`. The appendices collect the algebraic-geometry, toric-combinatorial, mirror-symmetric, and orientifold reference material cited from the main lectures.

## Building

The default build (Lectures 1–3 + appendices, with Lecture 4 omitted) is:

```bash
pdflatex main.tex
bibtex   main
pdflatex main.tex
pdflatex main.tex
```

Build-time toggles in the preamble of `main.tex`:

- `\includelecturefourtrue` — include the standalone Lecture 4 (`fanroots`) talk.
- `\includeappendicesfalse` — exclude the appendix block.

## Repository layout

```
.
├── main.tex          # top-level document; \input{sections/...}
├── references.bib    # bibliography (arXiv- and DOI-centred)
├── utphys.bst        # local HEP bibliography style
└── sections/         # per-section source files
    ├── 00_scope.tex
    ├── 01_lecture1.tex
    ├── 02_lecture2.tex
    ├── 03_lecture3.tex
    ├── 04_lecture4.tex
    ├── 05_outlook.tex
    ├── 06_appendix_A_conventions.tex
    ├── 07_appendix_B_foundations.tex
    ├── 08_appendix_C_convex_toric.tex
    ├── 09_appendix_D_toric_varieties.tex
    ├── 10_appendix_E_mirror_GV.tex
    └── 11_appendix_F_orientifolds_glossary.tex
```

## Related code and data

The notes lean on a small ecosystem of open-source computational tools and public databases. The most directly relevant entry points are:

### Calabi–Yau / toric pipelines

- **[CYTools](https://github.com/LiamMcAllisterGroup/cytools)** — Python toolkit for Calabi–Yau hypersurfaces in toric varieties: triangulations, intersection numbers, divisor topology, Gopakumar–Vafa invariants, mirror periods. *Demirtas, Rios-Tascon, McAllister, [arXiv:2211.03823](https://arxiv.org/abs/2211.03823).* Docs: [cytools.liammcallistergroup.com](https://cytools.liammcallistergroup.com).
- **[`fanroots`](https://github.com/LiamMcAllisterGroup/fanroots)** — chamber-aware root-finding and optimisation over secondary fans (the focus of Lecture 4).

### The [StringJAX](https://github.com/StringJAX) ecosystem

JAX-native, differentiable infrastructure for string compactifications. Exact gradients through geometry → effective theory → moduli stabilisation. Each member package is `pip`-installable and has its own ReadTheDocs site; an umbrella `stringjax` metapackage installs the lot.

- **[`stringjax`](https://github.com/StringJAX/stringjax)** — umbrella metapackage. `pip install stringjax` (or `pip install "stringjax[all]"` for the full stack). Docs: [stringjax.readthedocs.io](https://stringjax.readthedocs.io).
- **[`jaxvacua`](https://github.com/StringJAX/jaxvacua)** — Type IIB flux-vacuum engine: differentiable supergravity potentials, complex-structure + axio-dilaton stabilisation via GVW at large complex structure, batched vacuum search. *Dubey, Krippendorf, Schachner, [arXiv:2306.06160](https://arxiv.org/abs/2306.06160).* Docs: [jaxvacua.readthedocs.io](https://jaxvacua.readthedocs.io). [PyPI](https://pypi.org/project/jaxvacua/).
- **[`jaxpolylog`](https://github.com/StringJAX/jaxpolylog)** — differentiable polylogarithms in JAX; JIT-compile, vectorise, survive arbitrary-order autodiff even at extreme arguments ($|z| \lesssim 10^{-30}$). Docs: [jaxpolylog.readthedocs.io](https://jaxpolylog.readthedocs.io). [PyPI](https://pypi.org/project/jaxpolylog/).
- **[`stringforge`](https://github.com/StringJAX/stringforge)** — shared database, model-loading, and vacua-vault infrastructure for string-compactification workflows. Bridges Calabi–Yau geometry databases to JAXVacua physics. Docs: [stringforge.readthedocs.io](https://stringforge.readthedocs.io).

### Public Calabi–Yau and vacua data

- **[Kreuzer–Skarke database](http://hep.itp.tuwien.ac.at/~kreuzer/CY/)** — the canonical online list of 4-dimensional reflexive polytopes. *Kreuzer, Skarke, [arXiv:hep-th/0002240](https://arxiv.org/abs/hep-th/0002240).* A Hugging Face mirror of the full list (~473.8M rows, 15.8 GB, Parquet; vertex normal form, facet/lattice-point counts, Hodge numbers, Euler characteristic) is available at [`calabi-yau-data/polytopes-4d`](https://huggingface.co/datasets/calabi-yau-data/polytopes-4d).
- **[`aschachner/KS_Cornell`](https://huggingface.co/datasets/aschachner/KS_Cornell)** (Hugging Face, ~474M rows, Parquet) — Kreuzer–Skarke extension with per-polytope precomputed quantities useful for string compactifications: face counts by dimension, lattice-point and interior-point counts on faces and on dual-polytope faces, rigid-structure counts, favourability flags (`fav_N`, `fav_M`), trilayer classification, and ID mappings. Streamable via `datasets`, `dask`, or `polars`.
- **[`aschachner/cy-database`](https://huggingface.co/datasets/aschachner/cy-database)** (Hugging Face, 18.4 GB) — precomputed Calabi–Yau threefold data: TDF (trilayer, double-favourable Kreuzer–Skarke hypersurfaces), CICY (complete-intersection Calabi–Yau threefolds), and a curated KKLT index over TDF. Accessed via `stringforge`.
- **[`aschachner/vacua_vault`](https://huggingface.co/datasets/aschachner/vacua_vault)** (Hugging Face) — community-curated flux-sector vacua for Type IIB on Calabi–Yau threefolds: integer flux vectors, stabilised complex-structure moduli, axio-dilaton values, tadpole and classification metadata, cross-referenced against `cy-database`.

### General algebraic and polyhedral geometry

- **[Macaulay2](http://www2.macaulay2.com/Macaulay2/)** — system for computation in commutative algebra and algebraic geometry; useful for divisor cohomology and homological algebra of toric and Calabi–Yau spaces.
- **[polymake](https://polymake.org/)** — polyhedral geometry: fans, cones, polytopes, secondary fans, mixed volumes.
- **[TOPCOM](http://www.rambau.wm.uni-bayreuth.de/TOPCOM/)** — triangulations of point configurations and oriented matroids.
- **[Sage](https://www.sagemath.org/)** — open-source mathematics system; has built-in toric and polyhedral functionality.

### Anchor references

- McAllister, Moritz, Nally, Schachner, *Candidate de Sitter vacua*, [arXiv:2406.13751](https://arxiv.org/abs/2406.13751) — the worked-example construction the lectures build towards.
- McAllister, Schachner, *TASI 2024 lectures: de Sitter Vacua in String Theory*, [arXiv:2512.17095](https://arxiv.org/abs/2512.17095) — the companion physics-side exposition.
- Demirtas, Kim, McAllister, Moritz, Rios-Tascon, *Computational Mirror Symmetry*, [arXiv:2303.00757](https://arxiv.org/abs/2303.00757) — the CMS pipeline used throughout Lecture 2 and Appendix E.
- Cox, Little, Schenck, *Toric Varieties* (AMS GSM 124, 2011) — the standard textbook reference; the toric appendices follow its notation.

## Feedback

These notes are not yet a finished product. Comments — from mathematicians, string theorists, and computational scientists alike — are explicitly welcome:

- Open a GitHub issue for line-level corrections or precision concerns.
- For pedagogical or scope-level suggestions, an email is fine; my address is on the title page of the compiled PDF.

## Licence and citation

Distribution licence to be fixed before public release. A BibTeX entry for citing the notes will be provided once (of if) they are first uploaded to arXiv.
