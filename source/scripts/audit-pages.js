#!/usr/bin/env node
/**
 * audit-pages.js
 * Pulls page-level data from GSC Search Analytics API, reads current
 * titles + descriptions from generate-static.js and markdown frontmatter,
 * then outputs a prioritised audit report.
 *
 * Usage:
 *   node scripts/audit-pages.js              # last 90 days
 *   node scripts/audit-pages.js --days=30    # custom date range
 *   node scripts/audit-pages.js --json       # output raw JSON
 */

const { GoogleAuth } = require('google-auth-library');
const https          = require('https');
const fs             = require('fs');
const path           = require('path');

const ROOT       = path.resolve(__dirname, '../..');
const SOURCE     = path.resolve(__dirname, '..');
const CREDS_FILE = path.join(SOURCE, 'service-account.json');
const SITE_URL   = 'sc-domain:ailearnings.in';

const args    = process.argv.slice(2);
const DAYS    = parseInt((args.find(a => a.startsWith('--days=')) || '--days=90').split('=')[1]);
const JSON_OUT = args.includes('--json');

// ── Date helpers ──────────────────────────────────────────────────────────────
function toDateStr(date) {
  return date.toISOString().split('T')[0];
}
const endDate   = toDateStr(new Date());
const startDate = toDateStr(new Date(Date.now() - DAYS * 86400000));

// ── GSC API call ──────────────────────────────────────────────────────────────
function gscQuery(token, body) {
  return new Promise((resolve, reject) => {
    const payload = JSON.stringify(body);
    const req = https.request({
      hostname: 'www.googleapis.com',
      path:     `/webmasters/v3/sites/${encodeURIComponent(SITE_URL)}/searchAnalytics/query`,
      method:   'POST',
      headers:  {
        'Authorization':  `Bearer ${token}`,
        'Content-Type':   'application/json',
        'Content-Length': Buffer.byteLength(payload),
      },
    }, res => {
      let data = '';
      res.on('data', c => (data += c));
      res.on('end', () => {
        try { resolve(JSON.parse(data)); }
        catch (e) { reject(new Error(`Parse error: ${data}`)); }
      });
    });
    req.on('error', reject);
    req.write(payload);
    req.end();
  });
}

// ── Read titles + descriptions from generate-static.js ───────────────────────
function readStaticPageMeta() {
  const content = fs.readFileSync(path.join(SOURCE, 'scripts/generate-static.js'), 'utf8');
  const results = {};

  // Match page objects: slug + title + description
  const pageBlocks = [...content.matchAll(/slug:\s+'([^']+)'[\s\S]*?title:\s+'([^']+)'[\s\S]*?description:\s+'([^']+)'/g)];
  for (const [, slug, title, description] of pageBlocks) {
    const url = `https://ailearnings.in/${slug}/`;
    results[url] = { title, description, source: 'generate-static.js' };
  }
  return results;
}

// ── Read titles + descriptions from markdown frontmatter ─────────────────────
function readMarkdownMeta() {
  const results = {};
  const dirs = [
    path.join(SOURCE, 'blog/posts'),
    path.join(SOURCE, 'blog/roadmap-guides'),
    path.join(SOURCE, 'projects/posts'),
    path.join(SOURCE, 'paths'),
  ];

  for (const dir of dirs) {
    if (!fs.existsSync(dir)) continue;
    const files = fs.readdirSync(dir).filter(f => f.endsWith('.md'));
    for (const file of files) {
      const content = fs.readFileSync(path.join(dir, file), 'utf8');
      const slugMatch  = content.match(/^slug:\s+"?([^"\n]+)"?/m);
      const titleMatch = content.match(/^title:\s+"?([^"\n]+)"?/m);
      const descMatch  = content.match(/^description:\s+"?([^"\n]+)"?/m);

      if (!slugMatch || !titleMatch) continue;
      const slug = slugMatch[1].trim();
      const title = titleMatch[1].trim();
      const description = descMatch ? descMatch[1].trim() : '';

      // Determine URL prefix by directory
      let prefix = 'blog';
      if (dir.includes('projects')) prefix = 'projects';
      else if (dir.includes('paths')) prefix = 'paths';
      else if (dir.includes('roadmap-guides')) prefix = 'blog/roadmap-guides';

      const url = `https://ailearnings.in/${prefix}/${slug}/`;
      results[url] = { title, description, source: file };
    }
  }
  return results;
}

// ── Evaluation criteria ───────────────────────────────────────────────────────
function evaluatePage(url, title, description, gsc) {
  const issues = [];
  const { clicks = 0, impressions = 0, ctr = 0, position = 0 } = gsc || {};

  // Title length (Google truncates at ~60 chars)
  if (title.length > 65) issues.push(`title too long (${title.length} chars, max ~60)`);
  if (title.length < 20) issues.push(`title too short (${title.length} chars)`);

  // Description length (Google truncates at ~155 chars)
  if (description.length > 160) issues.push(`description too long (${description.length} chars)`);
  if (description.length < 50 && description.length > 0) issues.push(`description too short (${description.length} chars)`);

  // CTR vs position benchmarks (industry averages)
  if (impressions >= 10) {
    const expectedCtr = position <= 3 ? 0.10 : position <= 5 ? 0.05 : position <= 10 ? 0.02 : 0.005;
    if (ctr < expectedCtr * 0.5) {
      issues.push(`low CTR (${(ctr * 100).toFixed(2)}% at position ${position.toFixed(1)}, expected ~${(expectedCtr * 100).toFixed(0)}%)`);
    }
  }

  // Generic title signals
  const genericWords = [/\bguide\b/i, /\btutorial\b$/i, /\bexplained\b$/i, /\boverview\b/i];
  if (genericWords.some(r => r.test(title)) && !title.includes('2026') && !title.match(/\d+ /)) {
    issues.push('title may be generic — no year, number, or strong differentiator');
  }

  // Missing year for evergreen content
  if (!title.includes('2026') && !title.includes('2025') && impressions > 0) {
    issues.push('no year in title (hurts freshness signal)');
  }

  // Priority score: high impressions + low CTR = most urgent
  const priority = impressions * (1 - ctr) * (position <= 10 ? 2 : 1);

  return { issues, priority, clicks, impressions, ctr, position };
}

// ── Main ──────────────────────────────────────────────────────────────────────
async function main() {
  if (!fs.existsSync(CREDS_FILE)) {
    console.error(`❌  Credentials not found: ${CREDS_FILE}`);
    process.exit(1);
  }

  const auth = new GoogleAuth({
    keyFile: CREDS_FILE,
    scopes:  ['https://www.googleapis.com/auth/webmasters.readonly'],
  });
  const client = await auth.getClient();
  const { token } = await client.getAccessToken();

  console.log(`\nFetching GSC data (${startDate} → ${endDate})...\n`);

  // Pull up to 500 pages
  const gscData = await gscQuery(token, {
    startDate,
    endDate,
    dimensions: ['page'],
    rowLimit:   500,
  });

  if (gscData.error) {
    console.error('GSC API error:', gscData.error.message);
    process.exit(1);
  }

  // Index GSC rows by URL
  const gscByUrl = {};
  for (const row of gscData.rows || []) {
    const url = row.keys[0];
    gscByUrl[url] = {
      clicks:      row.clicks,
      impressions: row.impressions,
      ctr:         row.ctr,
      position:    row.position,
    };
  }

  // Read page metadata
  const staticMeta   = readStaticPageMeta();
  const markdownMeta = readMarkdownMeta();
  const allMeta      = { ...staticMeta, ...markdownMeta };

  if (JSON_OUT) {
    console.log(JSON.stringify({ gscByUrl, allMeta }, null, 2));
    return;
  }

  // Evaluate each page
  const report = [];
  const allUrls = new Set([...Object.keys(allMeta), ...Object.keys(gscByUrl)]);

  for (const url of allUrls) {
    const meta = allMeta[url] || {};
    const { title = '(no title found)', description = '' } = meta;
    const evaluation = evaluatePage(url, title, description, gscByUrl[url]);
    report.push({ url, title, description, ...evaluation, source: meta.source });
  }

  // Sort by priority (most urgent first)
  report.sort((a, b) => b.priority - a.priority);

  // ── Output ────────────────────────────────────────────────────────────────
  console.log(`${'─'.repeat(100)}`);
  console.log(`PAGE AUDIT REPORT  |  ${startDate} → ${endDate}  |  ${report.length} pages`);
  console.log(`${'─'.repeat(100)}\n`);

  // Pages with GSC data and issues
  const withIssues    = report.filter(p => p.issues.length > 0 && p.impressions > 0);
  const noDataIssues  = report.filter(p => p.issues.length > 0 && p.impressions === 0);
  const clean         = report.filter(p => p.issues.length === 0);

  console.log(`🔴  PRIORITY FIXES (ranked pages with issues)  —  ${withIssues.length} pages\n`);
  for (const p of withIssues) {
    console.log(`  ${p.url}`);
    console.log(`  Title: "${p.title}"`);
    console.log(`  GSC: ${p.impressions} impressions · ${p.clicks} clicks · ${(p.ctr * 100).toFixed(2)}% CTR · pos ${p.position.toFixed(1)}`);
    for (const issue of p.issues) console.log(`    ⚠  ${issue}`);
    console.log();
  }

  console.log(`${'─'.repeat(100)}`);
  console.log(`🟡  NO GSC DATA YET (title/description issues)  —  ${noDataIssues.length} pages\n`);
  for (const p of noDataIssues.slice(0, 20)) {
    console.log(`  ${p.url}`);
    console.log(`  Title: "${p.title}"`);
    for (const issue of p.issues) console.log(`    ⚠  ${issue}`);
    console.log();
  }
  if (noDataIssues.length > 20) console.log(`  ... and ${noDataIssues.length - 20} more\n`);

  console.log(`${'─'.repeat(100)}`);
  console.log(`✅  CLEAN  —  ${clean.length} pages with no issues detected\n`);

  // Summary
  console.log(`${'─'.repeat(100)}`);
  console.log(`SUMMARY`);
  console.log(`  Total pages audited:  ${report.length}`);
  console.log(`  Pages with GSC data:  ${report.filter(p => p.impressions > 0).length}`);
  console.log(`  Priority fixes:       ${withIssues.length}`);
  console.log(`  No-data issues:       ${noDataIssues.length}`);
  console.log(`  Clean:                ${clean.length}`);
  console.log(`${'─'.repeat(100)}\n`);
}

main().catch(err => {
  console.error('Fatal error:', err.message);
  process.exit(1);
});
