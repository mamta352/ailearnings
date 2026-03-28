#!/usr/bin/env node
/**
 * analyze-ctr-trends.js
 * Pulls query-level GSC data per page and validates that current
 * titles/descriptions align with the queries driving impressions.
 * Flags misalignments that could hurt CTR.
 *
 * Usage:
 *   node scripts/analyze-ctr-trends.js              # top 30 pages by impression, last 90 days
 *   node scripts/analyze-ctr-trends.js --days=30    # shorter window
 *   node scripts/analyze-ctr-trends.js --all        # all pages, not just top 30
 *   node scripts/analyze-ctr-trends.js --json       # output raw JSON
 */

const { GoogleAuth } = require('google-auth-library');
const https          = require('https');
const fs             = require('fs');
const path           = require('path');

const SOURCE     = path.resolve(__dirname, '..');
const CREDS_FILE = path.join(SOURCE, 'service-account.json');
const SITE_URL   = 'sc-domain:ailearnings.in';
const OUT_DIR    = path.join(SOURCE, 'reports');

const args    = process.argv.slice(2);
const DAYS    = parseInt((args.find(a => a.startsWith('--days=')) || '--days=90').split('=')[1]);
const JSON_OUT = args.includes('--json');
const ALL_PAGES = args.includes('--all');

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
      headers:  {
        'Authorization':  `Bearer ${token}`,
        'Content-Type':   'application/json',
        'Content-Length': Buffer.byteLength(payload),
      },
    }, res => {
      let d = ''; res.on('data', c => (d += c));
      res.on('end', () => { try { resolve(JSON.parse(d)); } catch(e) { reject(new Error(d)); } });
    });
    req.on('error', reject);
    req.write(payload);
    req.end();
  });
}

// ── Fetch top queries for a specific page ─────────────────────────────────────
async function fetchPageQueries(token, pageUrl) {
  const result = await gscQuery(token, {
    startDate,
    endDate,
    dimensions: ['query'],
    dimensionFilterGroups: [{
      filters: [{
        dimension: 'page',
        operator:  'equals',
        expression: pageUrl,
      }],
    }],
    rowLimit: 20,
    orderBy: [{ fieldName: 'impressions', sortOrder: 'DESCENDING' }],
  });
  return (result.rows || []).map(r => ({
    query:       r.keys[0],
    impressions: r.impressions,
    clicks:      r.clicks,
    ctr:         r.ctr,
    position:    r.position,
  }));
}

// ── Read metadata from generate-static.js ─────────────────────────────────────
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

// ── Read metadata from markdown frontmatter ────────────────────────────────────
function readMarkdownMeta() {
  const results = {};
  const dirs = [
    { dir: path.join(SOURCE, 'blog/posts'),          prefix: 'blog' },
    { dir: path.join(SOURCE, 'blog/roadmap-guides'), prefix: 'blog/roadmap-guides' },
    { dir: path.join(SOURCE, 'projects/posts'),      prefix: 'projects' },
    { dir: path.join(SOURCE, 'paths'),               prefix: 'paths' },
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

// ── Check if query terms appear in title/description ──────────────────────────
function tokenize(str) {
  return str.toLowerCase().replace(/[^a-z0-9\s]/g, ' ').split(/\s+/).filter(Boolean);
}

function queryAlignment(query, title, description) {
  const queryTokens = tokenize(query);
  const titleTokens = new Set(tokenize(title));
  const descTokens  = new Set(tokenize(description));
  const stopwords   = new Set(['a','an','the','and','or','for','to','in','on','of','with','how','is','are','best','top','what','vs']);

  const meaningful = queryTokens.filter(t => !stopwords.has(t) && t.length > 2);
  if (meaningful.length === 0) return { score: 1, matched: [], missing: [] };

  const matchedInTitle = meaningful.filter(t => titleTokens.has(t));
  const matchedInDesc  = meaningful.filter(t => descTokens.has(t) && !titleTokens.has(t));
  const missing        = meaningful.filter(t => !titleTokens.has(t) && !descTokens.has(t));

  const score = (matchedInTitle.length * 2 + matchedInDesc.length) / (meaningful.length * 2);
  return { score, matched: [...matchedInTitle, ...matchedInDesc], missing };
}

// ── Format helpers ─────────────────────────────────────────────────────────────
function bar(ctr, width = 20) {
  const filled = Math.round(ctr * width * 10); // scale: 10% = full bar
  return '█'.repeat(Math.min(filled, width)) + '░'.repeat(Math.max(width - filled, 0));
}

function pct(n) { return (n * 100).toFixed(1) + '%'; }
function pos(n) { return n.toFixed(1); }

// ── Main ──────────────────────────────────────────────────────────────────────
async function main() {
  if (!fs.existsSync(CREDS_FILE)) { console.error('❌  No credentials file'); process.exit(1); }
  if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR, { recursive: true });

  const auth = new GoogleAuth({ keyFile: CREDS_FILE, scopes: ['https://www.googleapis.com/auth/webmasters.readonly'] });
  const client = await auth.getClient();
  const { token } = await client.getAccessToken();

  console.log(`\nFetching page-level GSC data (${startDate} → ${endDate})...`);

  // Step 1: Get all pages with impressions
  const pageData = await gscQuery(token, {
    startDate, endDate, dimensions: ['page'], rowLimit: 500,
    orderBy: [{ fieldName: 'impressions', sortOrder: 'DESCENDING' }],
  });
  if (pageData.error) { console.error('GSC error:', pageData.error.message); process.exit(1); }

  const pages = (pageData.rows || [])
    .filter(r => !r.keys[0].match(/\.(png|jpg|xml|txt)$/))
    .map(r => ({
      url:         r.keys[0],
      impressions: r.impressions,
      clicks:      r.clicks,
      ctr:         r.ctr,
      position:    r.position,
    }));

  const topPages = ALL_PAGES ? pages : pages.slice(0, 30);
  console.log(`  ${pages.length} pages with impressions. Analyzing top ${topPages.length}...\n`);

  // Step 2: Read current metadata
  const allMeta = { ...readStaticMeta(), ...readMarkdownMeta() };

  // Step 3: Fetch queries per page (rate-limited, sequential)
  const results = [];
  for (const page of topPages) {
    process.stdout.write(`  Fetching queries: ${page.url.replace('https://ailearnings.in', '')}...`);
    const queries = await fetchPageQueries(token, page.url);
    const meta    = allMeta[page.url] || {};
    const { title = '', description = '' } = meta;

    // Analyze alignment of each query with current title/desc
    const queryAnalysis = queries.map(q => ({
      ...q,
      alignment: queryAlignment(q.query, title, description),
    }));

    // Overall alignment score (weighted by impressions)
    const totalImpressions = queries.reduce((s, q) => s + q.impressions, 0);
    const weightedScore = totalImpressions > 0
      ? queries.reduce((s, q) => s + q.impressions * queryAlignment(q.query, title, description).score, 0) / totalImpressions
      : null;

    // Collect missing terms across top queries (weighted by impressions)
    const missingTermFreq = {};
    for (const q of queries) {
      const { missing } = queryAlignment(q.query, title, description);
      for (const term of missing) {
        missingTermFreq[term] = (missingTermFreq[term] || 0) + q.impressions;
      }
    }
    const topMissingTerms = Object.entries(missingTermFreq)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([term]) => term);

    // CTR gap: expected vs actual
    const expectedCtr = page.position <= 3 ? 0.10 : page.position <= 5 ? 0.05 : page.position <= 10 ? 0.02 : 0.005;
    const ctrGap = expectedCtr - page.ctr;
    const ctrOpportunity = Math.round(page.impressions * ctrGap);

    results.push({
      ...page,
      title,
      description,
      source: meta.source || 'unknown',
      queries: queryAnalysis,
      alignmentScore:  weightedScore,
      topMissingTerms,
      expectedCtr,
      ctrGap,
      ctrOpportunity,
    });

    console.log(` ${queries.length} queries`);
    // Small delay to avoid rate limiting
    await new Promise(r => setTimeout(r, 200));
  }

  if (JSON_OUT) {
    const jsonPath = path.join(OUT_DIR, `ctr-trend-analysis-${endDate}.json`);
    fs.writeFileSync(jsonPath, JSON.stringify({ generated: new Date().toISOString(), date_range: `${startDate} to ${endDate}`, results }, null, 2));
    console.log(`\n✓ JSON: reports/ctr-trend-analysis-${endDate}.json`);
    return;
  }

  // ── Report ───────────────────────────────────────────────────────────────────
  console.log(`\n${'═'.repeat(100)}`);
  console.log(`CTR TREND ANALYSIS  |  ${startDate} → ${endDate}`);
  console.log(`${'═'.repeat(100)}\n`);

  // Sort by CTR opportunity (missed clicks)
  results.sort((a, b) => b.ctrOpportunity - a.ctrOpportunity);

  // Group into: poor alignment, ok alignment, good alignment
  const poorAlignment = results.filter(r => r.alignmentScore !== null && r.alignmentScore < 0.4);
  const okAlignment   = results.filter(r => r.alignmentScore !== null && r.alignmentScore >= 0.4 && r.alignmentScore < 0.7);
  const goodAlignment = results.filter(r => r.alignmentScore !== null && r.alignmentScore >= 0.7);
  const noData        = results.filter(r => r.alignmentScore === null);

  function printPage(p) {
    const align = p.alignmentScore !== null ? `${(p.alignmentScore * 100).toFixed(0)}%` : 'n/a';
    const ctrStr = pct(p.ctr);
    const expStr = pct(p.expectedCtr);
    const gap    = p.ctrOpportunity > 0 ? `+${p.ctrOpportunity} missed clicks` : 'at benchmark';

    console.log(`  ${p.url}`);
    console.log(`  Title:    "${p.title}"`);
    console.log(`  GSC:      ${p.impressions} impr · ${p.clicks} clicks · CTR ${ctrStr} (expected ${expStr}) · pos ${pos(p.position)}`);
    console.log(`  CTR gap:  ${bar(p.ctr)} ${ctrStr}  →  goal ${expStr}  [${gap}]`);
    console.log(`  Alignment: ${align} (how well title/desc matches search queries)`);

    if (p.topMissingTerms.length > 0) {
      console.log(`  Missing query terms in title/desc: ${p.topMissingTerms.map(t => `"${t}"`).join(', ')}`);
    }

    if (p.queries.length > 0) {
      console.log(`  Top queries:`);
      for (const q of p.queries.slice(0, 5)) {
        const a = q.alignment;
        const flag = a.score < 0.4 ? '⚠ ' : a.score < 0.7 ? '~  ' : '✓  ';
        const missing = a.missing.length > 0 ? `  [missing: ${a.missing.slice(0,3).join(', ')}]` : '';
        console.log(`    ${flag}"${q.query}"  ${q.impressions} impr · ${pct(q.ctr)} CTR · pos ${pos(q.position)}${missing}`);
      }
    }
    console.log();
  }

  if (poorAlignment.length > 0) {
    console.log(`🔴  POOR ALIGNMENT (title/desc doesn't match search queries)  —  ${poorAlignment.length} pages\n`);
    console.log(`    These pages rank for queries that don't appear in the title. Users see a mismatch\n    between what they searched for and what the result says — causing low CTR.\n`);
    for (const p of poorAlignment) printPage(p);
  }

  if (okAlignment.length > 0) {
    console.log(`${'─'.repeat(100)}`);
    console.log(`🟡  PARTIAL ALIGNMENT (some query terms missing)  —  ${okAlignment.length} pages\n`);
    for (const p of okAlignment) printPage(p);
  }

  if (goodAlignment.length > 0) {
    console.log(`${'─'.repeat(100)}`);
    console.log(`✅  GOOD ALIGNMENT  —  ${goodAlignment.length} pages\n`);
    for (const p of goodAlignment.slice(0, 10)) {
      console.log(`  ${p.url.replace('https://ailearnings.in','')}`);
      console.log(`  Title: "${p.title}"`);
      console.log(`  ${p.impressions} impr · CTR ${pct(p.ctr)} · pos ${pos(p.position)} · alignment ${(p.alignmentScore * 100).toFixed(0)}%`);
      console.log();
    }
    if (goodAlignment.length > 10) console.log(`  ... and ${goodAlignment.length - 10} more\n`);
  }

  // ── Summary ───────────────────────────────────────────────────────────────
  const totalMissed  = results.reduce((s, r) => s + Math.max(r.ctrOpportunity, 0), 0);
  const avgAlignment = results.filter(r => r.alignmentScore !== null).reduce((s, r) => s + r.alignmentScore, 0) /
                       Math.max(results.filter(r => r.alignmentScore !== null).length, 1);

  console.log(`${'═'.repeat(100)}`);
  console.log(`SUMMARY`);
  console.log(`  Pages analyzed:         ${results.length}`);
  console.log(`  Poor alignment (<40%):  ${poorAlignment.length} pages  ← rewrite titles to include query terms`);
  console.log(`  Partial alignment:      ${okAlignment.length} pages  ← add missing terms to description`);
  console.log(`  Good alignment (≥70%):  ${goodAlignment.length} pages`);
  console.log(`  Avg alignment score:    ${(avgAlignment * 100).toFixed(0)}%`);
  console.log(`  Estimated missed clicks: ~${totalMissed} (if CTR reached position benchmark)`);
  console.log(`${'═'.repeat(100)}\n`);

  // ── Recommendations ───────────────────────────────────────────────────────
  if (poorAlignment.length > 0) {
    console.log(`RECOMMENDATIONS:\n`);
    console.log(`  For poor-alignment pages, rewrite titles to include the exact query terms`);
    console.log(`  users are searching. Example patterns that improve CTR:\n`);
    console.log(`    ✓ Include the exact query phrase in the title`);
    console.log(`    ✓ Add specificity: numbers, year (2026), "step-by-step", "with examples"`);
    console.log(`    ✓ Add benefit: "in 30 min", "without X", "the right way"`);
    console.log(`    ✗ Avoid: vague "guide to", "understanding", "introduction to"`);
    console.log(`    ✗ Avoid: keyword stuffing — one clear value prop is better than 5 terms\n`);

    console.log(`  Top missing terms across all poor-alignment pages:`);
    const allMissing = {};
    for (const p of poorAlignment) {
      for (const term of p.topMissingTerms) {
        allMissing[term] = (allMissing[term] || 0) + p.impressions;
      }
    }
    const topMissing = Object.entries(allMissing).sort((a,b) => b[1]-a[1]).slice(0, 15);
    for (const [term, impr] of topMissing) {
      console.log(`    "${term}"  (${impr} impressions on pages that don't use this term in title/desc)`);
    }
    console.log();
  }

  // Save JSON report
  const jsonPath = path.join(OUT_DIR, `ctr-trend-analysis-${endDate}.json`);
  fs.writeFileSync(jsonPath, JSON.stringify({
    generated: new Date().toISOString(),
    date_range: `${startDate} to ${endDate}`,
    summary: {
      total: results.length,
      poor_alignment: poorAlignment.length,
      ok_alignment:   okAlignment.length,
      good_alignment: goodAlignment.length,
      avg_alignment_pct: Math.round(avgAlignment * 100),
      estimated_missed_clicks: totalMissed,
    },
    results,
  }, null, 2));
  console.log(`✓ Full report saved: reports/ctr-trend-analysis-${endDate}.json\n`);
}

main().catch(err => { console.error('Fatal:', err.message); process.exit(1); });
