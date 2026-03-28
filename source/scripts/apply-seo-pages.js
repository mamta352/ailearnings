#!/usr/bin/env node
/**
 * apply-seo-pages.js
 * Reads source/seo-master.json and applies title/description changes
 * back to source files. Only writes when a value has actually changed.
 *
 * Usage:
 *   node scripts/apply-seo-pages.js            # apply all changes
 *   node scripts/apply-seo-pages.js --dry-run  # preview without writing
 */

const fs   = require('fs');
const path = require('path');

const SOURCE    = path.resolve(__dirname, '..');
const ROOT      = path.resolve(SOURCE, '..');
const MASTER    = path.join(SOURCE, 'seo-master.json');
const STATIC_JS = path.join(SOURCE, 'scripts/generate-static.js');
const DRY_RUN   = process.argv.includes('--dry-run');

if (DRY_RUN) console.log('\n[DRY RUN] No files will be written.\n');

// ── Apply to markdown frontmatter ─────────────────────────────────────────────
function applyToMarkdown(sourceFile, newTitle, newDesc) {
  if (!fs.existsSync(sourceFile)) { console.warn(`  ⚠  Not found: ${sourceFile}`); return false; }
  let content = fs.readFileSync(sourceFile, 'utf8');
  let changed  = false;

  const curTitle = (content.match(/^title:\s*"([^"]+)"/m) || [])[1]?.trim() || '';
  const curDesc  = (content.match(/^description:\s*"([^"]+)"/m) || [])[1]?.trim() || '';

  if (curTitle !== newTitle) {
    content = content.replace(/^(title:\s*)"([^"]+)"/m, `$1"${newTitle}"`);
    changed = true;
    console.log(`    title:  "${curTitle}"`);
    console.log(`         →  "${newTitle}"`);
  }
  if (curDesc !== newDesc) {
    content = content.replace(/^(description:\s*)"([^"]+)"/m, `$1"${newDesc}"`);
    changed = true;
    console.log(`    desc:   "${curDesc.slice(0, 65)}..."`);
    console.log(`         →  "${newDesc.slice(0, 65)}..."`);
  }

  if (changed && !DRY_RUN) fs.writeFileSync(sourceFile, content, 'utf8');
  return changed;
}

// ── Apply to generate-static.js ───────────────────────────────────────────────
function applyToStaticJs(slug, newTitle, newDesc) {
  let content = fs.readFileSync(STATIC_JS, 'utf8');
  let changed  = false;

  const pattern = new RegExp(
    `(slug:\\s*'${slug}'[\\s\\S]{0,400}?title:\\s*['"])([^'"]+)(['"][\\s\\S]{0,400}?description:\\s*')([^']+)(')`,
    'g'
  );
  const updated = content.replace(pattern, (match, pre, oldTitle, mid, oldDesc, end) => {
    if (oldTitle === newTitle && oldDesc === newDesc) return match;
    changed = true;
    if (oldTitle !== newTitle) { console.log(`    title:  "${oldTitle}"`); console.log(`         →  "${newTitle}"`); }
    if (oldDesc  !== newDesc)  { console.log(`    desc:   "${oldDesc.slice(0,65)}..."`); console.log(`         →  "${newDesc.slice(0,65)}..."`); }
    return `${pre}${newTitle}${mid}${newDesc}${end}`;
  });

  if (!changed) {
    // double-quoted title variant
    const dqPattern = new RegExp(
      `(slug:\\s*'${slug}'[\\s\\S]{0,400}?title:\\s*")([^"]+)("[\\s\\S]{0,400}?description:\\s*')([^']+)(')`,
      'g'
    );
    const updated2 = content.replace(dqPattern, (match, pre, oldTitle, mid, oldDesc, end) => {
      if (oldTitle === newTitle && oldDesc === newDesc) return match;
      changed = true;
      if (oldTitle !== newTitle) { console.log(`    title:  "${oldTitle}"`); console.log(`         →  "${newTitle}"`); }
      return `${pre}${newTitle}${mid}${newDesc}${end}`;
    });
    if (changed && !DRY_RUN) fs.writeFileSync(STATIC_JS, updated2, 'utf8');
    return changed;
  }

  if (changed && !DRY_RUN) fs.writeFileSync(STATIC_JS, updated, 'utf8');
  return changed;
}

// ── Main ──────────────────────────────────────────────────────────────────────
function main() {
  if (!fs.existsSync(MASTER)) {
    console.error(`seo-master.json not found. Run: npm run seo:master`);
    process.exit(1);
  }

  const { pages } = JSON.parse(fs.readFileSync(MASTER, 'utf8'));
  console.log(`\nApplying changes from seo-master.json (${pages.length} pages)...\n`);

  let changed = 0, unchanged = 0;

  for (const page of pages) {
    const isStatic   = page.source_file === 'source/scripts/generate-static.js';
    const sourceFile = isStatic ? STATIC_JS : path.join(ROOT, page.source_file);

    const didChange = isStatic
      ? applyToStaticJs(page.slug, page.title, page.description)
      : applyToMarkdown(sourceFile, page.title, page.description);

    if (didChange) { changed++; console.log(`  ✓ [${page.type}] ${page.slug}\n`); }
    else unchanged++;
  }

  console.log(`${'─'.repeat(60)}`);
  console.log(`  Changed:   ${changed}`);
  console.log(`  Unchanged: ${unchanged}`);
  console.log(`${'─'.repeat(60)}`);

  if (changed > 0 && !DRY_RUN) {
    console.log('\nNext steps:');
    console.log('  npm run generate:content-data');
    console.log('  npm run generate:blog        (if blog posts changed)');
    console.log('  npm run generate:projects    (if project posts changed)');
    console.log('  npm run generate             (if standalone pages changed)');
    console.log('  npm run deploy\n');
  } else if (DRY_RUN && changed > 0) {
    console.log('\nRe-run without --dry-run to apply.\n');
  } else {
    console.log('\nNo changes to apply.\n');
  }
}

main();
