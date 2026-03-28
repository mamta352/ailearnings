#!/usr/bin/env node
/**
 * generate-sitemap.js
 * Regenerates all 4 sitemaps + sitemap index from live content.
 *
 * Usage:  node scripts/generate-sitemap.js
 */

const fs   = require('fs');
const path = require('path');

const ROOT    = path.resolve(__dirname, '../..');
const TODAY   = new Date().toISOString().slice(0, 10);
const BASE    = 'https://ailearnings.in';

// ── helpers ──────────────────────────────────────────────────────────────────
function urlEntry(loc, lastmod, changefreq = 'monthly', priority = '0.8') {
  return `  <url><loc>${loc}</loc><lastmod>${lastmod}</lastmod><changefreq>${changefreq}</changefreq><priority>${priority}</priority></url>`;
}

function wrap(urls) {
  return `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${urls.join('\n')}\n</urlset>\n`;
}

function getLastmod(mdPath) {
  try {
    const content = fs.readFileSync(mdPath, 'utf8');
    const m = content.match(/updatedAt:\s*["']?(\d{4}-\d{2}-\d{2})/);
    if (m) return m[1];
    const d = content.match(/date:\s*["']?(\d{4}-\d{2}-\d{2})/);
    if (d) return d[1];
  } catch {}
  return TODAY;
}

// ── 1. sitemap-blog.xml ───────────────────────────────────────────────────────
const blogDir   = path.join(__dirname, '../blog/posts');
const blogFiles = fs.readdirSync(blogDir).filter(f => f.endsWith('.md')).sort();

const blogUrls = [urlEntry(`${BASE}/blog/`, TODAY, 'weekly', '0.9')];
for (const f of blogFiles) {
  const slug    = f.replace('.md', '');
  const lastmod = getLastmod(path.join(blogDir, f));
  blogUrls.push(urlEntry(`${BASE}/blog/${slug}/`, lastmod));
}
fs.writeFileSync(path.join(ROOT, 'sitemap-blog.xml'), wrap(blogUrls));
console.log(`✓ sitemap-blog.xml — ${blogUrls.length} URLs`);

// ── 2. sitemap-guides.xml ────────────────────────────────────────────────────
const guidesDir   = path.join(__dirname, '../blog/roadmap-guides');
const guidesFiles = fs.readdirSync(guidesDir).filter(f => f.endsWith('.md')).sort();

const guidesUrls = [urlEntry(`${BASE}/blog/roadmap-guides/`, TODAY, 'weekly', '0.9')];
for (const f of guidesFiles) {
  const slug    = f.replace('.md', '');
  const lastmod = getLastmod(path.join(guidesDir, f));
  guidesUrls.push(urlEntry(`${BASE}/blog/roadmap-guides/${slug}/`, lastmod));
}
fs.writeFileSync(path.join(ROOT, 'sitemap-guides.xml'), wrap(guidesUrls));
console.log(`✓ sitemap-guides.xml — ${guidesUrls.length} URLs`);

// ── 3. sitemap-projects.xml ──────────────────────────────────────────────────
const projDir   = path.join(__dirname, '../projects/posts');
const projFiles = fs.readdirSync(projDir).filter(f => f.endsWith('.md')).sort();

const projUrls = [urlEntry(`${BASE}/projects/`, TODAY, 'weekly', '0.9')];
for (const f of projFiles) {
  const slug    = f.replace('.md', '');
  const lastmod = getLastmod(path.join(projDir, f));
  projUrls.push(urlEntry(`${BASE}/projects/${slug}/`, lastmod));
}
fs.writeFileSync(path.join(ROOT, 'sitemap-projects.xml'), wrap(projUrls));
console.log(`✓ sitemap-projects.xml — ${projUrls.length} URLs`);

// ── 4. sitemap-pages.xml ─────────────────────────────────────────────────────
const pathsDir   = path.join(__dirname, '../paths');
const pathsFiles = fs.existsSync(pathsDir)
  ? fs.readdirSync(pathsDir).filter(f => f.endsWith('.md')).sort()
  : [];

const staticPages = [
  urlEntry(`${BASE}/`,                      TODAY, 'weekly',  '1.0'),
  urlEntry(`${BASE}/blog/`,                 TODAY, 'weekly',  '0.9'),
  urlEntry(`${BASE}/projects/`,             TODAY, 'weekly',  '0.9'),
  urlEntry(`${BASE}/paths/`,                TODAY, 'monthly', '0.8'),
  urlEntry(`${BASE}/roadmap/`,              TODAY, 'monthly', '0.8'),
  urlEntry(`${BASE}/ai-engineering-roadmap/`, TODAY, 'monthly', '0.7'),
];
for (const f of pathsFiles) {
  const slug    = f.replace('.md', '');
  const lastmod = getLastmod(path.join(pathsDir, f));
  staticPages.push(urlEntry(`${BASE}/paths/${slug}/`, lastmod, 'monthly', '0.7'));
}
fs.writeFileSync(path.join(ROOT, 'sitemap-pages.xml'), wrap(staticPages));
console.log(`✓ sitemap-pages.xml — ${staticPages.length} URLs`);

// ── 5. sitemap index ─────────────────────────────────────────────────────────
const index = `<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap>
    <loc>${BASE}/sitemap-pages.xml</loc>
    <lastmod>${TODAY}</lastmod>
  </sitemap>
  <sitemap>
    <loc>${BASE}/sitemap-blog.xml</loc>
    <lastmod>${TODAY}</lastmod>
  </sitemap>
  <sitemap>
    <loc>${BASE}/sitemap-projects.xml</loc>
    <lastmod>${TODAY}</lastmod>
  </sitemap>
  <sitemap>
    <loc>${BASE}/sitemap-guides.xml</loc>
    <lastmod>${TODAY}</lastmod>
  </sitemap>
</sitemapindex>
`;
fs.writeFileSync(path.join(ROOT, 'sitemap.xml'), index);
console.log(`✓ sitemap.xml (index) updated — lastmod ${TODAY}`);

const total = blogUrls.length + guidesUrls.length + projUrls.length + staticPages.length;
console.log(`\n✅ Done — ${total} total URLs across 4 sitemaps`);
