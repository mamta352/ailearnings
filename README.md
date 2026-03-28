# AI Learnings — ailearnings.in

A static AI learning hub covering prompt engineering, LLMs, RAG, fine-tuning, AI agents, and career roadmaps. Built as a React SPA with server-side pre-rendering via Puppeteer, deployed to GitHub Pages at [ailearnings.in](https://ailearnings.in).

---

## Table of Contents

- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Local Setup](#local-setup)
- [Development Workflow](#development-workflow)
- [Content Authoring](#content-authoring)
- [Build Pipeline](#build-pipeline)
- [Static Generation](#static-generation)
- [Deployment](#deployment)
- [SEO & Indexing](#seo--indexing)
- [SEO Audit & Metadata Tools](#seo-audit--metadata-tools)
- [Scripts Reference](#scripts-reference)

---

## Project Structure

```
AI_Learning/                   ← repo root (also the GitHub Pages serving root)
├── index.html                 ← SPA entry point
├── 404.html                   ← GitHub Pages 404 fallback
├── sitemap.xml                ← sitemap index
├── sitemap-blog.xml
├── sitemap-pages.xml
├── sitemap-projects.xml
├── sitemap-guides.xml
├── robots.txt
├── CNAME                      ← custom domain: ailearnings.in
├── dist/                      ← compiled JS + CSS (auto-generated, do not edit)
│   ├── app.js
│   └── app.css
├── blog/                      ← pre-rendered blog pages (Puppeteer output)
├── projects/                  ← pre-rendered project pages
├── paths/                     ← pre-rendered learning path pages
├── docs/                      ← legacy pre-render output (may overlap)
└── source/                    ← all source code lives here
    ├── app.jsx                ← main React app (~8,000 LOC, all routing + components)
    ├── ai_roadmap.tsx         ← AI roadmap feature component
    ├── prompt_eng.tsx         ← Prompt engineering guide component
    ├── genai_guide.tsx        ← GenAI guide component
    ├── knowledge_assessment.tsx
    ├── knowledge_gaps.tsx
    ├── readiness_check.tsx
    ├── tailwind.css           ← Tailwind entry file
    ├── tailwind.config.js     ← Tailwind config (large safelist for dynamic classes)
    ├── package.json
    ├── src/
    │   ├── content-data.js    ← AUTO-GENERATED — never edit manually
    │   └── icons.jsx          ← custom React icon components
    ├── blog/posts/            ← 49 markdown blog posts (frontmatter + content)
    ├── projects/posts/        ← 21 markdown project tutorials
    ├── paths/                 ← 5 markdown learning path files
    ├── templates/             ← HTML templates for pre-rendering
    │   ├── blog-post.html
    │   ├── blog-index.html
    │   ├── project-post.html
    │   ├── projects-index.html
    │   ├── path-page.html
    │   └── paths-index.html
    ├── reports/                   ← SEO audit + metadata exports (auto-generated)
    └── scripts/
        ├── generate-content-data.js  ← parses markdown frontmatter → content-data.js
        ├── generate-static.js        ← Puppeteer SSG for static pages
        ├── generate-blog.js          ← Puppeteer SSG for blog posts
        ├── generate-projects.js      ← Puppeteer SSG for project pages
        ├── generate-paths.js         ← Puppeteer SSG for learning paths
        ├── generate-og-image.js      ← OG social image generator
        ├── generate-sitemap.js       ← sitemap generator
        ├── inline.js                 ← inlines CSS into HTML templates
        ├── create-noindex-pages.js   ← creates noindex redirect pages for legacy URLs
        ├── request-indexing.js       ← Google Indexing API submission
        ├── audit-pages.js            ← GSC + title/description audit report
        └── export-page-metadata.js   ← exports full page metadata to JSON + CSV
```

---

## Prerequisites

- **Node.js 20+** — required for Puppeteer and build tooling
- **npm 9+** — comes with Node 20
- **Git** — for version control and deployment

Install Node via [nvm](https://github.com/nvm-sh/nvm) (recommended):

```bash
nvm install 20
nvm use 20
```

---

## Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/<org>/AI_Learning.git
cd AI_Learning

# 2. Install dependencies (from source/ directory — always)
cd source
npm install
```

> All `npm` commands must be run from the `source/` directory.

---

## Development Workflow

### Watch mode (auto-rebuild on save)

```bash
cd source
npm run dev
```

Watches `**/*.tsx`, `**/*.jsx`, `tailwind.css`, `tailwind.config.js`, and `scripts/inline.js`. Rebuilds JS + CSS + inlines on any change (500ms debounce). Also auto-deploys on each rebuild — **only use this when you intend to push changes live.**

### Manual build (no deploy)

```bash
cd source
npm run build
```

Runs: `generate-content-data.js` → Babel + Terser (JS) → Tailwind (CSS) → `inline.js` (CSS injection into templates).

---

## Content Authoring

### Adding a blog post

1. Create a new file in `source/blog/posts/your-slug.md`
2. Add frontmatter:

```markdown
---
title: "Your Post Title"
description: "One-sentence meta description (150–160 chars)."
date: "2026-03-13"
slug: "your-slug"
keywords: "keyword1, keyword2, keyword3"
---

## Introduction
...body content...
```

3. Regenerate the content index:

```bash
cd source
npm run generate:content-data
```

4. Pre-render the blog pages:

```bash
npm run generate:blog
```

> `source/src/content-data.js` is **auto-generated** — never edit it manually. Always run `npm run generate:content-data` after adding or modifying content files.

### Content directories

| Directory | Type | Count |
|---|---|---|
| `source/blog/posts/` | Blog guides | 115 posts |
| `source/blog/roadmap-guides/` | Roadmap guide posts | 14 guides |
| `source/projects/posts/` | Project tutorials | 21 posts |
| `source/paths/` | Career learning paths | 5 paths |

### Frontmatter fields

| Field | Required | Description |
|---|---|---|
| `title` | Yes | Page `<title>` and H1 |
| `description` | Yes | Meta description |
| `date` | Yes | ISO date (`YYYY-MM-DD`) |
| `slug` | Yes | URL path segment (must be unique) |
| `keywords` | No | Comma-separated SEO keywords |
| `level` | No | `beginner` / `intermediate` / `advanced` |
| `time` | No | Estimated read time (e.g. `15 min`) |
| `stack` | No | Tech stack tags |

---

## Build Pipeline

```
Markdown files
    ↓
generate-content-data.js   → source/src/content-data.js
    ↓
app.jsx + icons.jsx + content-data.js
    ↓  (cat → combined.jsx)
Babel (JSX → JS)
    ↓
Terser (minify)
    ↓
dist/app.js
    ↓
Tailwind CSS
    ↓
dist/app.css
    ↓
inline.js                  → CSS injected into templates/*.html
```

---

## Static Generation

Pre-rendering uses Puppeteer to load the React SPA in a headless browser and dump the resulting HTML.

### Generate everything (full site rebuild)

```bash
cd source
npm run generate:all
```

This runs: `build` → `generate:blog` → `generate:projects` → `generate:paths` → `generate-static.js` (static landing pages).

### Generate selectively

```bash
npm run generate:blog       # blog posts → blog/*/index.html
npm run generate:projects   # project pages → projects/*/index.html
npm run generate:paths      # learning paths → paths/*/index.html
npm run generate            # build + static landing pages only
```

### Generate OG images

```bash
npm run generate:og
```

Output: `og-image.png` / `og-image.jpg` at repo root.

---

## Deployment

Deployment pushes to `main`, which is served directly by GitHub Pages (no build step in CI for static files — the pre-rendered HTML is committed to the repo).

```bash
cd source
npm run deploy
```

This runs from the repo root:
```bash
git add -A
git commit -m "build: auto-deploy [skip ci]"
git push
```

> **Always run `npm run generate:all` (or the relevant generate command) before deploying** to ensure pre-rendered HTML is up to date.

### CI — GitHub Actions

`.github/workflows/build.yml` triggers on push to `main` for changes under `source/`. It runs `npm ci` and `npm run build` (JS + CSS only — does not run Puppeteer SSG). The pre-rendered HTML must be committed locally before pushing.

---

## SEO & Indexing

### Sitemaps

Four sitemaps are maintained manually at the repo root:

- `sitemap-blog.xml` — all blog post URLs
- `sitemap-pages.xml` — static landing pages
- `sitemap-projects.xml` — project tutorial pages
- `sitemap-guides.xml` — learning path pages
- `sitemap.xml` — sitemap index referencing all four

After adding new pages, update the relevant sitemap and bump all `<lastmod>` dates in `sitemap.xml` to today's date.

### Google Indexing API (programmatic submission)

For faster indexing after publishing new content:

#### One-time setup

1. Google Cloud Console → create or select a project
2. **APIs & Services** → Enable **Web Search Indexing API**
3. **IAM & Admin** → Service Accounts → Create service account → Keys → Add Key → JSON → download
4. Save the JSON as `source/service-account.json` (already in `.gitignore`)
5. **Google Search Console** → Settings → Users and permissions → Add user → paste the service account email → Permission: **Owner**

#### Submit URLs

```bash
cd source

# Preview URLs without submitting
node scripts/request-indexing.js --dry-run

# Submit all sitemaps (200 URL/day limit)
node scripts/request-indexing.js

# Submit blog URLs only
node scripts/request-indexing.js --blog
```

Google processes indexing requests within 24–72 hours. Verify in **Search Console → URL Inspection**.

#### Submit specific URLs

```bash
node scripts/request-indexing.js --urls=https://ailearnings.in/blog/my-post/,https://ailearnings.in/blog/other-post/
```

### Legacy URL suppression (noindex pages)

Creates `noindex` + redirect HTML pages at old WordPress-era URL paths so Google deindexes them:

```bash
cd source
npm run seo:noindex
# or
make seo-noindex
```

Paths covered: `/lander`, `/comments/feed`, `/search`, `/category`, `/author`, `/feed`, `/wp-admin`, `/wp-login.php`.

### HTTPS redirect

All HTML templates include an inline JS redirect to enforce HTTPS:

```html
<script>if(location.protocol!=='https:')location.replace('https:'+location.href.substring(location.protocol.length));</script>
```

This resolves GSC "Alternate page with proper canonical tag" and "Page with redirect" issues caused by HTTP access.

---

## SEO Audit & Metadata Tools

Three scripts help evaluate and track page SEO health. All require `source/service-account.json` with GSC **Owner** access (see [Google Indexing API setup](#google-indexing-api-programmatic-submission)).

### Page audit (`audit-pages.js`)

Pulls GSC performance data (impressions, clicks, CTR, position) and cross-references with current titles and descriptions. Flags pages with low CTR, titles over 60 chars, descriptions over 155 chars, and missing years.

```bash
cd source

# 90-day audit (default)
npm run seo:audit
make seo-audit

# 30-day audit
npm run seo:audit:30
make seo-audit-30
```

Output: a prioritised report in the terminal — ranked pages with issues at the top, then pages with no GSC data, then clean pages.

**Priority scoring:** `impressions × (1 - CTR) × position_multiplier` — high-impression, low-CTR pages on page 1 rank highest.

### Metadata export (`export-page-metadata.js`)

Exports a full snapshot of every page — combining GSC data with source titles/descriptions and rendered HTML validation — to JSON and CSV.

```bash
cd source

# 90-day range (default)
npm run seo:export
make seo-export

# Custom range
node scripts/export-page-metadata.js --days=30
```

Output files written to `source/reports/`:
- `page-metadata-<date>.json` — full structured data
- `page-metadata-<date>.csv` — spreadsheet-friendly format

Each record includes:

| Field | Description |
|---|---|
| `url` | Full page URL |
| `title` | Title from source (markdown or generate-static.js) |
| `title_length` | Character count |
| `description` | Meta description from source |
| `desc_length` | Character count |
| `rendered_title` | Actual `<title>` in pre-rendered HTML |
| `html_exists` | Whether pre-rendered HTML file exists on disk |
| `title_mismatch` | Source title ≠ rendered title (regeneration needed) |
| `robots` | robots meta value in rendered HTML |
| `canonical` | canonical URL in rendered HTML |
| `impressions` | GSC impressions in date range |
| `clicks` | GSC clicks |
| `ctr_pct` | Click-through rate % |
| `position` | Average search position |
| `issues` | Pipe-separated issue flags |
| `priority` | Priority score (higher = fix first) |

**When to run:** after bulk title/description changes to validate all rendered HTML matches source, and to catch pages missing pre-rendered output.

### One-time GSC setup for SEO tools

1. Google Cloud Console → Enable **Google Search Console API** for your project
2. IAM → Service Accounts → your account → Keys → download JSON → save as `source/service-account.json`
3. GSC → Settings → Users and permissions → set service account to **Owner**

---

## Scripts Reference

All scripts run from `source/` with `node scripts/<name>.js` or via npm.

### Content pipeline

| Script | npm / make command | Description |
|---|---|---|
| `generate-content-data.js` | `npm run generate:content-data` | Parses markdown frontmatter → `src/content-data.js` |
| `generate-static.js` | `npm run generate` / `make generate` | Puppeteer SSG for static landing pages |
| `generate-blog.js` | `npm run generate:blog` / `make generate-blog` | Puppeteer SSG for all blog posts |
| `generate-projects.js` | `npm run generate:projects` / `make generate-projects` | Puppeteer SSG for project pages |
| `generate-paths.js` | `npm run generate:paths` / `make generate-paths` | Puppeteer SSG for learning path pages |
| `generate-sitemap.js` | `npm run generate:sitemap` / `make generate-sitemap` | Regenerates all XML sitemaps |
| `generate-og-image.js` | `npm run generate:og` / `make generate-og` | Generates OG social images |
| `inline.js` | `npm run build:inline` | Inlines `dist/app.css` into HTML templates |

### SEO tools

| Script | npm / make command | Description |
|---|---|---|
| `request-indexing.js` | `npm run seo:index` / `make seo-index` | Submits all sitemap URLs to Google Indexing API |
| `request-indexing.js` | `npm run seo:index:blog` / `make seo-index-blog` | Submits blog URLs only |
| `request-indexing.js` | `npm run seo:index:dry` / `make seo-index-dry` | Preview URLs without submitting |
| `audit-pages.js` | `npm run seo:audit` / `make seo-audit` | GSC + title/description audit (90 days) |
| `audit-pages.js` | `npm run seo:audit:30` / `make seo-audit-30` | Same audit, 30-day window |
| `export-page-metadata.js` | `npm run seo:export` / `make seo-export` | Export full metadata to JSON + CSV |
| `create-noindex-pages.js` | `npm run seo:noindex` / `make seo-noindex` | Create noindex pages for legacy URLs |

---

## Common Issues

**`npm run generate:content-data` fails with "Cannot find package.json"**
→ Make sure you're running from `source/`, not the repo root.

**Pre-rendered pages look unstyled**
→ Run `npm run build` first (generates `dist/app.css`), then re-run the generate command.

**`request-indexing.js` fails with "Credentials file not found"**
→ Download the service account JSON from Google Cloud Console and save as `source/service-account.json`. See [Google Indexing API setup](#google-indexing-api-programmatic-submission).

**Changes to `app.jsx` not reflected in pre-rendered HTML**
→ Run `npm run build` to recompile JS/CSS, then run the appropriate generate command.

**New blog post not appearing in the content index**
→ Run `npm run generate:content-data` after adding or editing any markdown file in `blog/posts/`, `projects/posts/`, or `paths/`.

**`audit-pages.js` or `export-page-metadata.js` fails with "User does not have sufficient permission"**
→ The service account must be set to **Owner** (not Full User) in GSC → Settings → Users and permissions.

**`audit-pages.js` shows title mismatches**
→ The source title was updated but the pre-rendered HTML wasn't regenerated. Run the relevant generate command (`npm run generate:blog`, `generate:projects`, `generate:paths`, or `generate`) then re-run the export to confirm zero mismatches.

**`export-page-metadata.js` shows `html_exists: false` for a page**
→ The page has a sitemap entry and source metadata but no pre-rendered HTML on disk. Run the appropriate generate command to produce the HTML output.
