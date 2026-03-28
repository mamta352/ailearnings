#!/usr/bin/env node
/**
 * generate-excalidraw.js
 * Exports all source/diagrams/*.excalidraw files to PNG via Puppeteer + Excalidraw.
 *
 * Usage:
 *   node scripts/generate-excalidraw.js              # export all
 *   node scripts/generate-excalidraw.js foo.excalidraw  # export one file
 *   node scripts/generate-excalidraw.js --dry-run    # preview only
 */

const fs       = require('fs');
const path     = require('path');
const puppeteer = require('puppeteer');

const ROOT        = path.resolve(__dirname, '../..');
const DIAGRAMS_SRC = path.join(__dirname, '../diagrams');
const DIAGRAMS_OUT = path.join(ROOT, 'assets/diagrams');
const TEMPLATE    = path.join(__dirname, '../templates/excalidraw-export.html');
const DRY_RUN     = process.argv.includes('--dry-run');
const TARGET      = process.argv.find(a => a.endsWith('.excalidraw') && !a.includes('/'));

async function main() {
  if (!fs.existsSync(DIAGRAMS_SRC)) {
    console.error('No source/diagrams/ directory found. Create it and add .excalidraw files.');
    process.exit(1);
  }

  let files = fs.readdirSync(DIAGRAMS_SRC).filter(f => f.endsWith('.excalidraw'));
  if (TARGET) files = files.filter(f => f === TARGET);

  if (files.length === 0) {
    console.log('No .excalidraw files found' + (TARGET ? ` matching "${TARGET}"` : ''));
    return;
  }

  if (DRY_RUN) {
    console.log('[dry-run] Would export:');
    files.forEach(f => console.log(`  ${f} → assets/diagrams/${f.replace('.excalidraw', '.png')}`));
    return;
  }

  fs.mkdirSync(DIAGRAMS_OUT, { recursive: true });

  console.log('Launching Puppeteer...');
  const browser = await puppeteer.launch({ args: ['--no-sandbox', '--disable-setuid-sandbox'] });
  const page    = await browser.newPage();
  await page.setViewport({ width: 1600, height: 900 });

  console.log('Loading Excalidraw template (fetching CDN scripts)...');
  await page.goto(`file://${TEMPLATE}`, { waitUntil: 'networkidle0', timeout: 60000 });

  // Verify Excalidraw loaded
  const hasLib = await page.evaluate(() => typeof window.ExcalidrawLib !== 'undefined');
  if (!hasLib) {
    console.error('ExcalidrawLib not available — check CDN connectivity or version.');
    await browser.close();
    process.exit(1);
  }
  console.log('Excalidraw ready.\n');

  let exported = 0;
  let failed   = 0;

  for (const file of files) {
    const name    = file.replace('.excalidraw', '');
    const outPath = path.join(DIAGRAMS_OUT, `${name}.png`);
    const srcPath = path.join(DIAGRAMS_SRC, file);

    const sceneJSON = fs.readFileSync(srcPath, 'utf8');

    try {
      // Trigger export
      await page.evaluate((json) => window.exportExcalidraw(json), sceneJSON);

      // Wait for result (success or error)
      await page.waitForFunction(
        () => {
          const r = document.getElementById('result');
          const s = r.getAttribute('data-ready');
          return s === 'true' || s === 'error';
        },
        { timeout: 30000 }
      );

      const status = await page.$eval('#result', el => el.getAttribute('data-ready'));

      if (status === 'error') {
        const err = await page.$eval('#result', el => el.getAttribute('data-error'));
        console.error(`  ✗ Failed: ${file} — ${err}`);
        failed++;
        continue;
      }

      const dataUrl  = await page.$eval('#result', el => el.getAttribute('data-png'));
      const base64   = dataUrl.replace(/^data:image\/png;base64,/, '');
      fs.writeFileSync(outPath, Buffer.from(base64, 'base64'));

      const kb = Math.round(fs.statSync(outPath).size / 1024);
      console.log(`  ✓ ${name}.png  (${kb} KB)`);
      exported++;

    } catch (e) {
      console.error(`  ✗ Timeout/error: ${file} — ${e.message}`);
      failed++;
    }
  }

  await browser.close();
  console.log(`\nDone — ${exported} exported, ${failed} failed.`);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
