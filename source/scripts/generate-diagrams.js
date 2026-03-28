#!/usr/bin/env node
/**
 * generate-diagrams.js
 * Finds all ```mermaid blocks in blog/posts/*.md and roadmap-guides/*.md,
 * renders them to PNG via mmdc, saves to /assets/diagrams/{slug}-{n}.png,
 * and replaces the mermaid block in the markdown with an <img> tag.
 *
 * Usage:
 *   node scripts/generate-diagrams.js           # process all posts
 *   node scripts/generate-diagrams.js --dry-run # preview without writing
 *
 * Requires: @mermaid-js/mermaid-cli (npm install --save-dev @mermaid-js/mermaid-cli)
 */

const fs     = require('fs');
const path   = require('path');
const { execSync } = require('child_process');

const ROOT       = path.resolve(__dirname, '../..');
const MMDC       = path.resolve(__dirname, '../node_modules/.bin/mmdc');
const DIAGRAMS   = path.join(ROOT, 'assets/diagrams');
const DRY_RUN    = process.argv.includes('--dry-run');

const DIRS = [
  path.join(__dirname, '../blog/posts'),
  path.join(__dirname, '../blog/roadmap-guides'),
];

const MERMAID_RE = /```mermaid\n([\s\S]*?)```/g;

if (!DRY_RUN) fs.mkdirSync(DIAGRAMS, { recursive: true });

let total = 0;
let skipped = 0;

for (const dir of DIRS) {
  if (!fs.existsSync(dir)) continue;

  for (const file of fs.readdirSync(dir).filter(f => f.endsWith('.md'))) {
    const filePath = path.join(dir, file);
    const slug     = file.replace('.md', '');
    let content    = fs.readFileSync(filePath, 'utf8');

    let n = 0;
    let modified = false;

    content = content.replace(MERMAID_RE, (match, diagram) => {
      n++;
      const imgName = `${slug}-diagram-${n}.png`;
      const imgPath = path.join(DIAGRAMS, imgName);
      const imgSrc  = `/assets/diagrams/${imgName}`;

      if (fs.existsSync(imgPath)) {
        console.log(`  ⏭  Exists: ${imgName}`);
        skipped++;
        return `![Architecture diagram](${imgSrc})`;
      }

      if (DRY_RUN) {
        console.log(`  [dry] Would render: ${imgName}`);
        total++;
        return match;
      }

      // Write temp .mmd file
      const tmpFile = path.join('/tmp', `${slug}-${n}.mmd`);
      fs.writeFileSync(tmpFile, diagram.trim());

      try {
        execSync(
          `${MMDC} -i "${tmpFile}" -o "${imgPath}" -b transparent -t dark --width 1400 --scale 2`,
          { stdio: 'pipe' }
        );
        console.log(`  ✓  Rendered: ${imgName}`);
        total++;
        modified = true;
        return `![Architecture diagram](${imgSrc})`;
      } catch (e) {
        console.warn(`  ⚠  Failed to render ${imgName}: ${e.message}`);
        return match; // leave original if render fails
      } finally {
        fs.rmSync(tmpFile, { force: true });
      }
    });

    if (modified && !DRY_RUN) {
      fs.writeFileSync(filePath, content, 'utf8');
      console.log(`  ✎  Updated markdown: ${file}`);
    }
  }
}

console.log(`\n${DRY_RUN ? '[dry-run] ' : ''}Done — ${total} rendered, ${skipped} skipped (already exist).`);
