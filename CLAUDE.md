# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Static AI learning hub (ailearnings.in) — React SPA pre-rendered to static HTML via Puppeteer, deployed to GitHub Pages. All commands run from `source/`.

## Development Commands

```bash
cd source/
npm run build              # Build JS (Babel+Terser) + CSS (Tailwind) + inline into index.html
npm run build:inline       # Inline CSS only (skips JS/CSS recompile)
npm run dev                # Watch mode: rebuild + deploy on file changes (500ms debounce)
```

**Regenerate static HTML after any content or component change:**
```bash
node scripts/generate-static.js   # Landing pages (app.jsx components) → docs/[slug]/index.html
npm run generate:blog              # Blog posts (markdown) → docs/blog/[slug]/index.html
npm run generate:projects          # Project posts → docs/projects/[slug]/index.html
npm run generate:paths             # Path pages → docs/paths/[slug]/index.html
npm run generate:all               # Full site rebuild (slow — runs build first)
npm run generate:content-data      # Regenerate content-data.js from markdown frontmatter only
```

**SEO workflow (edit titles/descriptions via seo-master.json):**
```bash
npm run seo:master           # Refresh seo-master.json with live GSC data (requires service-account.json)
npm run seo:master:fast      # Same but skip GSC API call — keep existing performance numbers
npm run seo:apply:dry        # Preview what seo-apply would change
npm run seo:apply            # Push title/description from seo-master.json back to source files
```

**Deploy:**
```bash
npm run deploy   # git add -A + commit "build: auto-deploy" + push → GitHub Pages
```

There is no test suite. Validation is done by running static generation and visually inspecting the output HTML.

## Architecture

### Two Rendering Pipelines

This is the most important architectural fact. There are **two separate pipelines** that produce different output types:

**Pipeline 1 — React/Puppeteer SSG** (for standalone landing pages):
- Source: Components defined in `source/app.jsx`
- Metadata: Title/description defined in the `PAGES` array in `source/scripts/generate-static.js`
- Script: `node scripts/generate-static.js` — spins up a local server, visits each route with Puppeteer, saves pre-rendered HTML
- Output: `docs/[slug]/index.html` (e.g. `docs/langchain-tutorial/`, `docs/rag-tutorial/`)
- Examples: all landing pages like `/langchain-tutorial/`, `/ollama-local-llm-guide/`, `/ai-roadmap-for-developers/`

**Pipeline 2 — Markdown → HTML** (for blog posts, projects, paths):
- Source: Markdown files in `source/blog/posts/`, `source/blog/roadmap-guides/`, `source/projects/posts/`, `source/paths/`
- Scripts: `generate-blog.js`, `generate-projects.js`, `generate-paths.js`
- Uses `marked` + `highlight.js` for code highlighting; no Puppeteer
- Output: `docs/blog/[slug]/index.html`, `docs/projects/[slug]/index.html`, `docs/paths/[slug]/index.html`
- Title/description come from markdown frontmatter (`title:`, `description:`)

### Adding a New Landing Page

Requires changes in **three places**:

1. **`source/app.jsx`** — add a new React component (e.g. `function MyNewPage() { ... }`)
2. **`source/app.jsx` TABS or routing** — register the component so the SPA routes to it
3. **`source/scripts/generate-static.js` `PAGES` array** — add an entry with `slug`, `outDir`, `url`, `title`, `description`, `canonical`, `ogUrl`, and optionally `schema`

Then run `node scripts/generate-static.js` to produce the HTML.

### Adding a New Blog Post

1. Create `source/blog/posts/[filename].md` with frontmatter: `slug`, `title`, `description`, `date`, `level`, `time`, `stack`
2. Run `npm run generate:content-data` (updates the blog listing)
3. Run `npm run generate:blog` (produces `docs/blog/[slug]/index.html`)
4. Add an entry to `source/seo-master.json` for the new page

### SEO Master File (`source/seo-master.json`)

Single source of truth for all ~177 pages' SEO metadata. Fields per page: `url`, `slug`, `type`, `source_file`, `title`, `description`, `impressions`, `clicks`, `ctr_pct`, `position`, `seo` (intent/hook/priority).

**`seo-apply` regex limitation — critical:** `apply-seo-pages.js` uses `([^']+)` to match description values in single-quoted JS strings inside `generate-static.js`. **Never use contractions** (apostrophes) in `seo-master.json` titles or descriptions. Use "do not" not "don't", "it is" not "it's", etc. Violating this corrupts the string in `generate-static.js` on the next `seo:apply` run.

The `seo-apply` script targets two source types based on `source_file`:
- Markdown files → updates `title:` and `description:` frontmatter fields
- `source/scripts/generate-static.js` → updates values in the `PAGES` array via regex

### Blog Listing Sort Order

`source/scripts/generate-content-data.js` loads `seo-master.json` and sorts blog posts by:
1. `seo.priority` — `high` → `medium` → `low`
2. `impressions` DESC (GSC data)
3. `ctr_pct` ASC (low CTR = needs visibility most)

This means the blog listing at `/blog/` surfaces high-priority, high-traffic pages first — not newest-first.

### `source/app.jsx` Structure

~8,500 LOC monolith. All standalone page components live here. Key sections:
- `useSeo(title, description)` — sets `<title>` and meta description at runtime (browser-side only; the canonical metadata is in `generate-static.js` PAGES array)
- Roadmap component with 7 phases (inline data)
- All standalone tutorial/guide pages (e.g. `LangChainTutorialPage`, `OllamaGuidePage`)
- `TABS` array and routing logic at the bottom

### Key Files

| File | Purpose |
|---|---|
| `source/app.jsx` | All React components and routing |
| `source/scripts/generate-static.js` | Puppeteer SSG + canonical SEO metadata for landing pages |
| `source/scripts/generate-blog.js` | Markdown → blog HTML (uses marked + highlight.js) |
| `source/scripts/generate-content-data.js` | Parses markdown frontmatter → `src/content-data.js` |
| `source/seo-master.json` | SEO metadata for all pages (edit here, then seo:apply) |
| `source/src/content-data.js` | Auto-generated — never edit manually |
| `source/tailwind.config.js` | Includes large safelist for dynamic gradient classes |

## Blog Content Improvement (Ongoing Project)

Blog posts are being improved systematically by topic phase. When working on any blog post, apply all 6 steps of the content strategy:

1. **Analyze intent** — what is the reader really trying to learn?
2. **Create outline** — logical H1/H2/H3 hierarchy, beginner → advanced
3. **Write the content** — strong hook (problem/insight, not generic opener), first-principles explanation, working copy-paste code, common mistakes, best practices, expert-level insights
4. **Optimize for AI + SEO** — clear headings, semantic keywords, FAQ section (5–7 Q&A), Key Takeaways
5. **Readability** — short paragraphs, bullet points, no fluff
6. **When updating** — expand thin sections, add missing code, remove redundancy

**Every blog post must have:** hook → explanation → working code → common mistakes → FAQ (5–7) → Key Takeaways → "What to Learn Next" links

**Technical standards for code in posts:**
- Use LCEL syntax for LangChain (not deprecated `RetrievalQA`, `LLMChain`, `ConversationalRetrievalChain`)
- Include cost/performance tables where relevant
- Test both happy path and failure path in examples
- No contractions in text (apostrophes corrupt `apply-seo-pages.js` regex)

**Priority order within each phase:** process pages by GSC impressions DESC. High-impression pages first, zero-impression pages last.

**Phase plan (10 topic phases):**
- Phase 1: RAG (15 posts) — `build-rag-app` ✓ done
- Phase 2: AI Agents (14 posts)
- Phase 3: Prompt Engineering (13 posts)
- Phase 4: Fine-tuning & Training (13 posts)
- Phase 5: LLMs & Models (12 posts)
- Phase 6: APIs & SDKs (12 posts)
- Phase 7: Vector DBs & Embeddings (13 posts)
- Phase 8: Local LLMs & Ollama (4 posts)
- Phase 9: ML Fundamentals (9 posts)
- Phase 10: Production & Apps (6 posts)

After updating any blog post markdown: run `npm run generate:blog` from `source/`, then `npm run deploy`.

### Static Output

All pre-rendered HTML lands in `docs/`. GitHub Pages serves `docs/` as the site root. The repo root `index.html` and `404.html` are the SPA shell — `docs/` contains the pre-rendered per-route files.
