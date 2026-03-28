#!/usr/bin/env node
/**
 * export-page-metadata.js
 * Builds a metadata file for every page — combining GSC performance data
 * with current title, description, length checks, and issue flags.
 *
 * Usage:
 *   node scripts/export-page-metadata.js              # last 90 days, outputs JSON + CSV
 *   node scripts/export-page-metadata.js --days=30    # custom range
 */

const { GoogleAuth } = require('google-auth-library');
const https          = require('https');
const fs             = require('fs');
const path           = require('path');

const ROOT       = path.resolve(__dirname, '../..');
const SOURCE     = path.resolve(__dirname, '..');
const CREDS_FILE = path.join(SOURCE, 'service-account.json');
const SITE_URL   = 'sc-domain:ailearnings.in';
const OUT_DIR    = path.join(SOURCE, 'reports');

const args  = process.argv.slice(2);
const DAYS  = parseInt((args.find(a => a.startsWith('--days=')) || '--days=90').split('=')[1]);

function toDateStr(d) { return d.toISOString().split('T')[0]; }
const endDate   = toDateStr(new Date());
const startDate = toDateStr(new Date(Date.now() - DAYS * 86400000));

// ── GSC API ───────────────────────────────────────────────────────────────────
function gscQuery(token, body) {
  return new Promise((resolve, reject) => {
    const payload = JSON.stringify(body);
    const req = https.request({
      hostname: 'www.googleapis.com',
      path:     `/webmasters/v3/sites/${encodeURIComponent(SITE_URL)}/searchAnalytics/query`,
      method:   'POST',
      headers:  { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload) },
    }, res => {
      let d = ''; res.on('data', c => (d += c)); res.on('end', () => { try { resolve(JSON.parse(d)); } catch(e) { reject(new Error(d)); } });
    });
    req.on('error', reject);
    req.write(payload);
    req.end();
  });
}

// ── Read metadata from generate-static.js ────────────────────────────────────
function readStaticMeta() {
  const content = fs.readFileSync(path.join(SOURCE, 'scripts/generate-static.js'), 'utf8');
  const results = {};
  const blocks = [...content.matchAll(/slug:\s+'([^']+)'[\s\S]*?(?:title:\s+'([^']*)'|title:\s+"([^"]*)")[\s\S]*?description:\s+'([^']*)'/g)];
  for (const [, slug, t1, t2, description] of blocks) {
    const title = t1 || t2 || '';
    const url = `https://ailearnings.in/${slug}/`;
    results[url] = { title, description, source: 'standalone' };
  }
  return results;
}

// ── Read metadata from markdown frontmatter ───────────────────────────────────
function readMarkdownMeta() {
  const results = {};
  const dirs = [
    { dir: path.join(SOURCE, 'blog/posts'),              prefix: 'blog' },
    { dir: path.join(SOURCE, 'blog/roadmap-guides'),     prefix: 'blog/roadmap-guides' },
    { dir: path.join(SOURCE, 'projects/posts'),          prefix: 'projects' },
    { dir: path.join(SOURCE, 'paths'),                   prefix: 'paths' },
  ];
  for (const { dir, prefix } of dirs) {
    if (!fs.existsSync(dir)) continue;
    for (const file of fs.readdirSync(dir).filter(f => f.endsWith('.md'))) {
      const content = fs.readFileSync(path.join(dir, file), 'utf8');
      const slug  = (content.match(/^slug:\s+"?([^"\n]+)"?/m) || [])[1]?.trim();
      const title = (content.match(/^title:\s+"([^"]+)"/m) || [])[1]?.trim();
      const desc  = (content.match(/^description:\s+"([^"]+)"/m) || [])[1]?.trim();
      if (!slug || !title) continue;
      const url = `https://ailearnings.in/${prefix}/${slug}/`;
      results[url] = { title, description: desc || '', source: `${prefix}/${file}` };
    }
  }
  return results;
}

// ── Validate HTML output ───────────────────────────────────────────────────────
function readRenderedMeta(url) {
  const slug = url.replace('https://ailearnings.in/', '').replace(/\/$/, '');
  const htmlPath = path.join(ROOT, slug, 'index.html');
  if (!fs.existsSync(htmlPath)) return { rendered_title: null, rendered_desc: null, html_exists: false };
  const html = fs.readFileSync(htmlPath, 'utf8');
  const rendered_title = (html.match(/<title>([^<]+)<\/title>/) || [])[1]?.trim() || null;
  const rendered_desc  = (html.match(/<meta name="description" content="([^"]+)"/) || [])[1]?.trim() || null;
  const robots         = (html.match(/<meta name="robots" content="([^"]+)"/) || [])[1]?.trim() || null;
  const canonical      = (html.match(/<link rel="canonical" href="([^"]+)"/) || [])[1]?.trim() || null;
  return { rendered_title, rendered_desc, robots, canonical, html_exists: true };
}

// ── Evaluate ──────────────────────────────────────────────────────────────────
function evaluate(title, description, gsc) {
  const issues = [];
  const { clicks = 0, impressions = 0, ctr = 0, position = 0 } = gsc || {};
  if (title.length > 65)   issues.push('title_too_long');
  if (title.length < 20)   issues.push('title_too_short');
  if (description.length > 155) issues.push('desc_too_long');
  if (description.length < 50 && description.length > 0) issues.push('desc_too_short');
  if (!title.match(/202[5-6]/)) issues.push('no_year');
  if (impressions >= 10) {
    const expected = position <= 3 ? 0.10 : position <= 5 ? 0.05 : position <= 10 ? 0.02 : 0.005;
    if (ctr < expected * 0.5) issues.push('low_ctr');
  }
  const priority = impressions > 0
    ? Math.round(impressions * (1 - ctr) * (position <= 10 ? 2 : 1))
    : 0;
  return { issues, priority };
}

// ── CSV helper ────────────────────────────────────────────────────────────────
function csvEscape(val) {
  if (val === null || val === undefined) return '';
  const s = String(val);
  return s.includes(',') || s.includes('"') || s.includes('\n') ? `"${s.replace(/"/g, '""')}"` : s;
}

// ── Main ──────────────────────────────────────────────────────────────────────
async function main() {
  if (!fs.existsSync(CREDS_FILE)) { console.error('❌  No credentials file'); process.exit(1); }
  if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR, { recursive: true });

  const auth = new GoogleAuth({ keyFile: CREDS_FILE, scopes: ['https://www.googleapis.com/auth/webmasters.readonly'] });
  const client = await auth.getClient();
  const { token } = await client.getAccessToken();

  console.log(`\nFetching GSC data (${startDate} → ${endDate})...`);
  const gscData = await gscQuery(token, { startDate, endDate, dimensions: ['page'], rowLimit: 500 });
  if (gscData.error) { console.error('GSC error:', gscData.error.message); process.exit(1); }

  const gscByUrl = {};
  for (const row of gscData.rows || []) {
    gscByUrl[row.keys[0]] = { clicks: row.clicks, impressions: row.impressions, ctr: row.ctr, position: row.position };
  }
  console.log(`  ${Object.keys(gscByUrl).length} pages with GSC data\n`);

  const staticMeta   = readStaticMeta();
  const markdownMeta = readMarkdownMeta();
  const allMeta      = { ...staticMeta, ...markdownMeta };

  const allUrls = new Set([...Object.keys(allMeta), ...Object.keys(gscByUrl)]);
  const records = [];

  for (const url of allUrls) {
    if (url.endsWith('.png') || url.endsWith('.jpg')) continue; // skip assets

    const meta   = allMeta[url] || {};
    const gsc    = gscByUrl[url] || {};
    const rendered = readRenderedMeta(url);
    const { issues, priority } = evaluate(meta.title || '', meta.description || '', gsc);

    // Check if rendered title matches source title (normalize HTML entities and strip suffix)
    const normalizeTitle = t => (t || '').replace(/&amp;/g, '&').replace(/ \| AI Learning Hub$/, '').trim();
    const title_mismatch = rendered.rendered_title && meta.title &&
      normalizeTitle(rendered.rendered_title) !== normalizeTitle(meta.title);

    records.push({
      url,
      source:           meta.source || 'gsc_only',
      title:            meta.title || '',
      title_length:     (meta.title || '').length,
      description:      meta.description || '',
      desc_length:      (meta.description || '').length,
      rendered_title:   rendered.rendered_title || '',
      rendered_desc:    rendered.rendered_desc || '',
      html_exists:      rendered.html_exists,
      robots:           rendered.robots || '',
      canonical:        rendered.canonical || '',
      title_mismatch:   title_mismatch || false,
      impressions:      gsc.impressions || 0,
      clicks:           gsc.clicks || 0,
      ctr_pct:          gsc.ctr ? (gsc.ctr * 100).toFixed(2) : '0.00',
      position:         gsc.position ? gsc.position.toFixed(1) : '0.0',
      issues:           issues.join(' | '),
      priority,
    });
  }

  // Sort by priority desc
  records.sort((a, b) => b.priority - a.priority);

  // ── Write JSON ────────────────────────────────────────────────────────────
  const jsonPath = path.join(OUT_DIR, `page-metadata-${endDate}.json`);
  fs.writeFileSync(jsonPath, JSON.stringify({ generated: new Date().toISOString(), date_range: `${startDate} to ${endDate}`, total: records.length, records }, null, 2));
  console.log(`✓ JSON: ${path.relative(SOURCE, jsonPath)}`);

  // ── Write CSV ─────────────────────────────────────────────────────────────
  const csvHeaders = ['url','source','title','title_length','description','desc_length','rendered_title','html_exists','title_mismatch','robots','canonical','impressions','clicks','ctr_pct','position','issues','priority'];
  const csvRows = [csvHeaders.join(','), ...records.map(r => csvHeaders.map(h => csvEscape(r[h])).join(','))];
  const csvPath = path.join(OUT_DIR, `page-metadata-${endDate}.csv`);
  fs.writeFileSync(csvPath, csvRows.join('\n'));
  console.log(`✓ CSV: ${path.relative(SOURCE, csvPath)}`);

  // ── Summary ───────────────────────────────────────────────────────────────
  const withGsc      = records.filter(r => r.impressions > 0);
  const missingHtml  = records.filter(r => !r.html_exists && r.source !== 'gsc_only');
  const mismatched   = records.filter(r => r.title_mismatch);
  const hasIssues    = records.filter(r => r.issues.length > 0);

  console.log(`
── Summary ─────────────────────────────────
  Total pages:          ${records.length}
  With GSC data:        ${withGsc.length}
  Missing HTML output:  ${missingHtml.length}
  Title mismatches:     ${mismatched.length}
  Pages with issues:    ${hasIssues.length}
────────────────────────────────────────────`);

  if (missingHtml.length > 0) {
    console.log('\n⚠  Pages missing pre-rendered HTML:');
    missingHtml.forEach(r => console.log(`  ${r.url}`));
  }
  if (mismatched.length > 0) {
    console.log('\n⚠  Title mismatches (source vs rendered HTML):');
    mismatched.forEach(r => console.log(`  ${r.url}\n    Source:   "${r.title}"\n    Rendered: "${r.rendered_title}"`));
  }
}

main().catch(err => { console.error('Fatal:', err.message); process.exit(1); });
