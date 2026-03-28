.PHONY: help install build clean generate deploy \
        generate-all generate-blog generate-projects generate-paths generate-sitemap generate-og \
        seo-generate seo-generate-fast seo-apply seo-apply-dry \
        seo-audit seo-audit-30 seo-export seo-index seo-index-blog seo-index-dry seo-noindex \
        seo-ctr seo-ctr-30

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  AI Learnings — make commands"
	@echo ""
	@echo "  SETUP & BUILD"
	@echo "    install              Install npm dependencies"
	@echo "    build                Compile JS (Babel+Terser) + CSS (Tailwind) + inline CSS into templates"
	@echo "    clean                Delete dist/app.js and dist/app.css"
	@echo ""
	@echo "  STATIC GENERATION  (run after build to pre-render HTML)"
	@echo "    generate             Build + generate static landing pages only"
	@echo "    generate-all         Full site rebuild — blog + projects + paths + static pages (slow)"
	@echo "    generate-blog        Pre-render all blog posts → blog/*/index.html"
	@echo "    generate-projects    Pre-render all project pages → projects/*/index.html"
	@echo "    generate-paths       Pre-render all learning path pages → paths/*/index.html"
	@echo "    generate-sitemap     Regenerate all XML sitemaps"
	@echo "    generate-og          Generate OG social media images"
	@echo ""
	@echo "  SEO MASTER FILE  (source/seo-master.json is the source of truth)"
	@echo "    seo-master           Regenerate seo-master.json with live GSC data (edit this file to change SEO)"
	@echo "    seo-master-fast      Same but skips GSC API call — keeps existing GSC numbers"
	@echo "    seo-apply            Push title/description changes from seo-master.json back to source files"
	@echo "    seo-apply-dry        Preview what seo-apply would change without writing files"
	@echo ""
	@echo "  SEO ANALYSIS  (read-only, requires source/service-account.json)"
	@echo "    seo-audit            Page audit: GSC data + title/desc checks, ranked by priority (90 days)"
	@echo "    seo-audit-30         Same audit over last 30 days"
	@echo "    seo-export           Export full metadata snapshot to reports/page-metadata-<date>.json + .csv"
	@echo "    seo-ctr              CTR trend analysis — validates title/desc alignment with search queries (90 days)"
	@echo "    seo-ctr-30           Same CTR analysis over last 30 days"
	@echo ""
	@echo "  SEO ACTIONS  (requires source/service-account.json)"
	@echo "    seo-index            Submit all sitemap URLs to Google Indexing API (200 URL/day limit)"
	@echo "    seo-index-blog       Submit blog URLs only"
	@echo "    seo-index-dry        Preview URLs that would be submitted (no actual submission)"
	@echo "    seo-noindex          Create noindex+redirect pages for legacy WordPress URLs"
	@echo ""
	@echo "  DEPLOY"
	@echo "    deploy               git add -A + commit + push to main (auto-deploys to GitHub Pages)"
	@echo ""
	@echo "  Typical workflow:"
	@echo "    make seo-master             # refresh seo-master.json with latest GSC data"
	@echo "    # edit source/seo-master.json (title/description fields)"
	@echo "    make seo-apply-dry          # preview changes"
	@echo "    make seo-apply              # apply changes to source files"
	@echo "    make generate-blog          # regenerate HTML"
	@echo "    make deploy                 # push live"
	@echo ""

# ── Setup ────────────────────────────────────────────────────────────────────
install:
	npm install

# ── Build ────────────────────────────────────────────────────────────────────
build: install
	npm run build

clean:
	rm -f dist/app.js dist/app.css

# ── Static generation ────────────────────────────────────────────────────────
generate: build
	npm run generate

generate-all: build
	npm run generate:all

generate-blog:
	npm run generate:blog

generate-projects:
	npm run generate:projects

generate-paths:
	npm run generate:paths

generate-sitemap:
	npm run generate:sitemap

generate-og:
	npm run generate:og

# ── SEO master file ───────────────────────────────────────────────────────────

## Regenerate seo-master.json with live GSC data — source of truth for all page SEO
## Edit title/description in seo-master.json, then run seo-apply
seo-master:
	npm run seo:master

## Same but skips GSC API call (fast, keeps existing GSC numbers)
seo-master-fast:
	npm run seo:master:fast

## Push title/description changes from seo-master.json back to source files
seo-apply:
	npm run seo:apply

## Preview what seo-apply would change without writing any files
seo-apply-dry:
	npm run seo:apply:dry

# ── SEO analysis ──────────────────────────────────────────────────────────────

## Page audit: GSC performance + title/description checks, ranked by priority (90 days)
## Requires: source/service-account.json with GSC Owner access
seo-audit:
	npm run seo:audit

## Same audit over last 30 days
seo-audit-30:
	npm run seo:audit:30

## Export full page metadata to reports/page-metadata-<date>.json + .csv
seo-export:
	npm run seo:export

## CTR trend analysis — validates title/desc alignment with actual search queries (90 days)
seo-ctr:
	npm run seo:ctr

## Same CTR analysis over last 30 days
seo-ctr-30:
	npm run seo:ctr:30

# ── SEO actions ───────────────────────────────────────────────────────────────

## Submit all sitemap URLs to Google Indexing API (200 URL/day limit)
seo-index:
	npm run seo:index

## Submit blog URLs only
seo-index-blog:
	npm run seo:index:blog

## Submit only ranked pages (impressions > 0 in seo-master.json)
seo-index-ranked:
	npm run seo:index:ranked

## Preview URLs that would be submitted (no actual submission)
seo-index-dry:
	npm run seo:index:dry

## Preview ranked pages that would be submitted
seo-index-ranked-dry:
	npm run seo:index:ranked:dry

## Create noindex redirect pages for legacy WordPress URLs
seo-noindex:
	npm run seo:noindex

# ── Deploy ───────────────────────────────────────────────────────────────────
deploy:
	npm run deploy
