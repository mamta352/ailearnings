#!/usr/bin/env node
/**
 * generate-seo-master.js
 * Generates source/seo-master.json — single source of truth for all page SEO.
 * Combines live GSC data with current titles/descriptions from source files.
 *
 * Workflow:
 *   1. npm run seo:master          → refresh seo-master.json with latest GSC data
 *   2. Edit title/description in seo-master.json
 *   3. npm run seo:apply           → push changes back to source files
 *   4. npm run generate:blog (etc) → regenerate HTML
 *   5. npm run deploy
 *
 * Usage:
 *   node scripts/generate-seo-master.js           # pull fresh GSC data
 *   node scripts/generate-seo-master.js --no-gsc  # skip GSC, keep zeros
 */

const { GoogleAuth } = require('google-auth-library');
const https = require('https');
const fs    = require('fs');
const path  = require('path');

const SOURCE     = path.resolve(__dirname, '..');
const CREDS_FILE = path.join(SOURCE, 'service-account.json');
const SITE_URL   = 'sc-domain:ailearnings.in';
const OUT        = path.join(SOURCE, 'seo-master.json');

const args   = process.argv.slice(2);
const NO_GSC = args.includes('--no-gsc') || !fs.existsSync(CREDS_FILE);

function toDateStr(d) { return d.toISOString().split('T')[0]; }
const endDate   = toDateStr(new Date());
const startDate = toDateStr(new Date(Date.now() - 90 * 86400000));

// ── GSC ───────────────────────────────────────────────────────────────────────
async function fetchGscData() {
  if (NO_GSC) { console.log('  Skipping GSC (--no-gsc or no credentials)'); return {}; }
  const auth = new GoogleAuth({ keyFile: CREDS_FILE, scopes: ['https://www.googleapis.com/auth/webmasters.readonly'] });
  const client = await auth.getClient();
  const { token } = await client.getAccessToken();

  const res = await gscQuery(token, {
    startDate, endDate, dimensions: ['page'], rowLimit: 500,
    orderBy: [{ fieldName: 'impressions', sortOrder: 'DESCENDING' }],
  });

  const byUrl = {};
  for (const row of res.rows || []) {
    byUrl[row.keys[0]] = {
      impressions: row.impressions,
      clicks:      row.clicks,
      ctr_pct:     parseFloat((row.ctr * 100).toFixed(2)),
      position:    parseFloat(row.position.toFixed(1)),
    };
  }

  const topPages = Object.entries(byUrl)
    .filter(([, v]) => v.impressions > 0)
    .sort((a, b) => b[1].impressions - a[1].impressions)
    .slice(0, 40).map(([url]) => url);

  console.log(`  ${Object.keys(byUrl).length} pages with GSC data. Fetching queries for top ${topPages.length}...`);

  for (const url of topPages) {
    const qRes = await gscQuery(token, {
      startDate, endDate, dimensions: ['query'],
      dimensionFilterGroups: [{ filters: [{ dimension: 'page', operator: 'equals', expression: url }] }],
      rowLimit: 5,
      orderBy: [{ fieldName: 'impressions', sortOrder: 'DESCENDING' }],
    });
    byUrl[url].top_queries = (qRes.rows || []).map(r => r.keys[0]);
    await new Promise(r => setTimeout(r, 150));
  }
  return byUrl;
}

function gscQuery(token, body) {
  return new Promise((resolve, reject) => {
    const payload = JSON.stringify(body);
    const req = https.request({
      hostname: 'www.googleapis.com',
      path:     `/webmasters/v3/sites/${encodeURIComponent(SITE_URL)}/searchAnalytics/query`,
      method:   'POST',
      headers:  { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload) },
    }, res => {
      let d = ''; res.on('data', c => (d += c));
      res.on('end', () => { try { resolve(JSON.parse(d)); } catch(e) { reject(new Error(d)); } });
    });
    req.on('error', reject); req.write(payload); req.end();
  });
}

// ── Preserve existing notes from current seo-master.json ─────────────────────
function readExistingNotes() {
  if (!fs.existsSync(OUT)) return {};
  try {
    const data = JSON.parse(fs.readFileSync(OUT, 'utf8'));
    const notes = {};
    for (const p of data.pages || []) {
      if (p.notes) notes[p.slug] = p.notes;
    }
    return notes;
  } catch(e) { return {}; }
}

// ── Parse markdown frontmatter ────────────────────────────────────────────────
function parseFrontmatter(content) {
  const fm = {};
  const block = content.match(/^---\n([\s\S]*?)\n---/);
  if (!block) return fm;
  for (const line of block[1].split('\n')) {
    const m = line.match(/^(\w+):\s*(.+)/);
    if (m) fm[m[1]] = m[2].trim().replace(/^["']|["']$/g, '');
  }
  return fm;
}

// ── Read standalone pages from generate-static.js ────────────────────────────
function readStaticPages() {
  const content = fs.readFileSync(path.join(SOURCE, 'scripts/generate-static.js'), 'utf8');
  const pages = [];
  const blocks = [...content.matchAll(/\{\s*\n\s*slug:\s+'([^']+)'[\s\S]*?title:\s*['"]([^'"]+)['"][\s\S]*?description:\s*'([^']+)'/g)];
  for (const [, slug, title, description] of blocks) {
    pages.push({ slug, title, description, url: `https://ailearnings.in/${slug}/` });
  }
  return pages;
}

// ── Evaluate issues ────────────────────────────────────────────────────────────
function getIssues(title, description, gsc) {
  const issues = [];
  if (title.length > 60)  issues.push('title_too_long');
  if (title.length < 20)  issues.push('title_too_short');
  if (!title.match(/202[5-9]/)) issues.push('no_year');
  if (description.length > 155) issues.push('desc_too_long');
  if (description.length < 50 && description.length > 0) issues.push('desc_too_short');
  if (gsc && gsc.impressions >= 10) {
    const exp = gsc.position <= 3 ? 0.10 : gsc.position <= 5 ? 0.05 : gsc.position <= 10 ? 0.02 : 0.005;
    if (gsc.ctr_pct / 100 < exp * 0.5) issues.push('low_ctr');
  }
  return issues;
}

// ── Main ──────────────────────────────────────────────────────────────────────
async function main() {
  console.log(`\nGenerating seo-master.json (${startDate} → ${endDate})...\n`);

  const gscByUrl      = await fetchGscData();
  const existingNotes = readExistingNotes();
  const pages         = [];

  const dirs = [
    { dir: path.join(SOURCE, 'blog/posts'),          type: 'blog',          urlPrefix: 'blog',                srcDir: 'source/blog/posts' },
    { dir: path.join(SOURCE, 'blog/roadmap-guides'), type: 'roadmap-guide', urlPrefix: 'blog/roadmap-guides', srcDir: 'source/blog/roadmap-guides' },
    { dir: path.join(SOURCE, 'projects/posts'),      type: 'project',       urlPrefix: 'projects',            srcDir: 'source/projects/posts' },
    { dir: path.join(SOURCE, 'paths'),               type: 'path',          urlPrefix: 'paths',               srcDir: 'source/paths' },
  ];

  for (const { dir, type, urlPrefix, srcDir } of dirs) {
    if (!fs.existsSync(dir)) continue;
    for (const file of fs.readdirSync(dir).filter(f => f.endsWith('.md'))) {
      const content = fs.readFileSync(path.join(dir, file), 'utf8');
      const fm      = parseFrontmatter(content);
      const slug    = fm.slug || file.replace('.md', '');
      const url     = `https://ailearnings.in/${urlPrefix}/${slug}/`;
      const gsc     = gscByUrl[url] || {};

      let keywords = fm.keywords || '';
      if (keywords.startsWith('[')) {
        try { keywords = JSON.parse(keywords).join(', '); }
        catch(e) { keywords = keywords.replace(/[\[\]"]/g, '').replace(/,\s*/g, ', '); }
      }

      pages.push({
        url,
        slug,
        type,
        source_file:  `${srcDir}/${file}`,
        title:        fm.title || '',
        title_length: (fm.title || '').length,
        description:  fm.description || '',
        desc_length:  (fm.description || '').length,
        keywords,
        date:         fm.date  || '',
        level:        fm.level || '',
        impressions:  gsc.impressions  || 0,
        clicks:       gsc.clicks       || 0,
        ctr_pct:      gsc.ctr_pct      || 0,
        position:     gsc.position     || 0,
        top_queries:  gsc.top_queries  || [],
        issues:       getIssues(fm.title || '', fm.description || '', gsc),
        notes:        existingNotes[slug] || '',
      });
    }
  }

  // Standalone pages
  for (const p of readStaticPages()) {
    const gsc = gscByUrl[p.url] || {};
    pages.push({
      url:          p.url,
      slug:         p.slug,
      type:         'standalone',
      source_file:  'source/scripts/generate-static.js',
      title:        p.title,
      title_length: p.title.length,
      description:  p.description,
      desc_length:  p.description.length,
      keywords:     '',
      date:         '',
      level:        '',
      impressions:  gsc.impressions  || 0,
      clicks:       gsc.clicks       || 0,
      ctr_pct:      gsc.ctr_pct      || 0,
      position:     gsc.position     || 0,
      top_queries:  gsc.top_queries  || [],
      issues:       getIssues(p.title, p.description, gsc),
      notes:        existingNotes[p.slug] || '',
    });
  }

  // Sort: impressions desc, then url
  pages.sort((a, b) => b.impressions - a.impressions || a.url.localeCompare(b.url));

  const master = {
    generated:   new Date().toISOString(),
    gsc_range:   `${startDate} to ${endDate}`,
    total:       pages.length,
    with_gsc:    pages.filter(p => p.impressions > 0).length,
    with_issues: pages.filter(p => p.issues.length > 0).length,
    pages,
  };

  fs.writeFileSync(OUT, JSON.stringify(master, null, 2), 'utf8');

  const kb = (fs.statSync(OUT).size / 1024).toFixed(0);
  console.log(`✓ seo-master.json — ${pages.length} pages (${kb} KB)`);
  console.log(`  Path:           ${OUT}`);
  console.log(`  With GSC data:  ${master.with_gsc}`);
  console.log(`  With issues:    ${master.with_issues}`);
  console.log(`\nEdit title/description in seo-master.json, then run: npm run seo:apply\n`);
}

main().catch(err => { console.error('Fatal:', err.message); process.exit(1); });
