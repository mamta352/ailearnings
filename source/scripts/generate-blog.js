#!/usr/bin/env node
/**
 * generate-blog.js
 * Reads .md files from source/blog/posts/, converts to static HTML using
 * the blog-post.html template, and outputs to /blog/{slug}/index.html.
 * Also generates /blog/index.html from aggregated post metadata.
 *
 * Usage: node scripts/generate-blog.js
 */

const fs   = require('fs');
const path = require('path');
const { marked } = require('marked');
const hljs = require('highlight.js');

// Language display names and dot colors
const LANG_META = {
  python: { label: 'Python', dot: '#3b82f6' },
  py: { label: 'Python', dot: '#3b82f6' },
  javascript: { label: 'JavaScript', dot: '#f59e0b' },
  js: { label: 'JavaScript', dot: '#f59e0b' },
  typescript: { label: 'TypeScript', dot: '#60a5fa' },
  ts: { label: 'TypeScript', dot: '#60a5fa' },
  bash: { label: 'Bash', dot: '#10b981' },
  sh: { label: 'Shell', dot: '#10b981' },
  shell: { label: 'Shell', dot: '#10b981' },
  json: { label: 'JSON', dot: '#a78bfa' },
  yaml: { label: 'YAML', dot: '#fb923c' },
  html: { label: 'HTML', dot: '#f87171' },
  css: { label: 'CSS', dot: '#38bdf8' },
  sql: { label: 'SQL', dot: '#34d399' },
  rust: { label: 'Rust', dot: '#f97316' },
  go: { label: 'Go', dot: '#22d3ee' },
  java: { label: 'Java', dot: '#f59e0b' },
  cpp: { label: 'C++', dot: '#60a5fa' },
  c: { label: 'C', dot: '#60a5fa' },
};

// Configure marked to use highlight.js for code blocks
marked.use({
  renderer: {
    code(token) {
      const lang = (token.lang || '').toLowerCase();
      const code = token.text;
      let highlighted;
      if (lang && hljs.getLanguage(lang)) {
        highlighted = hljs.highlight(code, { language: lang, ignoreIllegals: true }).value;
      } else {
        highlighted = hljs.highlightAuto(code).value;
      }
      const meta = LANG_META[lang];
      const label = meta ? meta.label : (lang || 'plaintext');
      const dot = meta ? meta.dot : '#6b7280';
      return `<div class="code-viewer"><div class="code-viewer-header"><span class="code-viewer-dot" style="background:${dot}"></span><span class="code-viewer-lang">${label}</span><button class="code-copy-btn" onclick="(function(b){var p=b.closest('.code-viewer').querySelector('code');navigator.clipboard.writeText(p.innerText).then(function(){b.textContent='Copied!';b.classList.add('copied');setTimeout(function(){b.textContent='Copy';b.classList.remove('copied');},2000);});})(this)">Copy</button></div><pre><code class="hljs language-${lang || 'plaintext'}">${highlighted}</code></pre></div>`;
    }
  }
});

const ROOT              = path.resolve(__dirname, '../..');
const POSTS_DIR         = path.join(__dirname, '../blog/posts');
const ROADMAP_GUIDES_DIR = path.join(__dirname, '../blog/roadmap-guides');
const TEMPLATES_DIR     = path.join(__dirname, '../templates');
const BLOG_OUT          = path.join(ROOT, 'blog');
const CSS_FILE          = path.join(ROOT, 'dist/app.css');

// ── Helpers ──────────────────────────────────────────────────────────────────

function parseFrontmatter(raw) {
  const match = raw.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n([\s\S]*)$/);
  if (!match) return { meta: {}, content: raw };

  const meta = {};
  match[1].split('\n').forEach(line => {
    const colonIdx = line.indexOf(':');
    if (colonIdx === -1) return;
    const key = line.slice(0, colonIdx).trim();
    let val   = line.slice(colonIdx + 1).trim();
    // Strip surrounding quotes
    if ((val.startsWith('"') && val.endsWith('"')) ||
        (val.startsWith("'") && val.endsWith("'"))) {
      val = val.slice(1, -1);
    }
    // Parse YAML arrays: ["a", "b"]
    if (val.startsWith('[')) {
      try { meta[key] = JSON.parse(val.replace(/'/g, '"')); }
      catch { meta[key] = val; }
    } else {
      meta[key] = val;
    }
  });

  return { meta, content: match[2] };
}

function readTime(markdown) {
  const words = markdown.replace(/[#*`>\[\]()-]/g, '').split(/\s+/).length;
  return Math.max(1, Math.round(words / 220));
}

function formatDate(iso) {
  const d = new Date(iso + 'T00:00:00Z');
  return d.toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric', timeZone: 'UTC' });
}

function esc(s) {
  return String(s).replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ── Main ─────────────────────────────────────────────────────────────────────

function main() {
  const postTemplate  = fs.readFileSync(path.join(TEMPLATES_DIR, 'blog-post.html'), 'utf8');
  const indexTemplate = fs.readFileSync(path.join(TEMPLATES_DIR, 'blog-index.html'), 'utf8');
  const inlineCss     = fs.readFileSync(CSS_FILE, 'utf8');

  const mdFiles = fs.readdirSync(POSTS_DIR)
    .filter(f => f.endsWith('.md'))
    .sort();

  if (mdFiles.length === 0) {
    console.log('No .md files found in', POSTS_DIR);
    return;
  }

  const posts = [];

  for (const file of mdFiles) {
    const raw  = fs.readFileSync(path.join(POSTS_DIR, file), 'utf8');
    const { meta, content } = parseFrontmatter(raw);

    const slug        = meta.slug || file.replace('.md', '');
    const title       = meta.title || slug;
    const description = meta.description || '';
    const date        = meta.date || '2026-03-09';
    const updatedAt   = meta.updatedAt || meta.updated_at || date;

    // Auto-assign Mamta as author for weekday posts
    const postDay = new Date(date + 'T12:00:00Z').getUTCDay(); // 0=Sun,6=Sat
    const isWeekday = postDay >= 1 && postDay <= 5;
    const defaultAuthor = isWeekday ? 'Mamta Chauhan' : 'AI Learning Hub';
    const defaultAuthorTitle = isWeekday ? 'Content Creator and AI Enthusiast' : 'AI Learning Hub';
    const defaultAuthorBio = isWeekday
      ? 'Mamta Chauhan is an AI enthusiast and content creator behind ailearnings.in. She writes practical guides on LLMs, RAG, and AI engineering to help developers navigate the fast-moving world of artificial intelligence. Passionate about bridging the gap between cutting-edge research and real-world application.'
      : 'Software engineer building practical AI systems — RAG pipelines, LLM applications, and developer AI tools. Created AI Learning Hub to provide a structured, no-fluff roadmap for developers entering AI.';

    // Weekday rule overrides frontmatter author
    const authorName  = isWeekday ? 'Mamta Chauhan' : (meta.author || defaultAuthor);
    const authorTitle = isWeekday ? 'Content Creator and AI Enthusiast' : (meta.authorTitle || meta.author_title || defaultAuthorTitle);
    const canonical   = `https://ailearnings.in/blog/${slug}/`;
    const mins        = readTime(content);
    const htmlContent = marked.parse(content);

    // Author initials (up to 2 chars)
    const authorInitials = authorName.split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase();
    const authorBio = meta.authorBio || meta.author_bio || defaultAuthorBio;

    // Updated meta snippet — only shown if updatedAt differs from date
    const updatedMeta = updatedAt !== date
      ? `<span class="sep">·</span><span>Updated <time datetime="${updatedAt}">${formatDate(updatedAt)}</time></span>`
      : '';

    const articleSchema = JSON.stringify({
      '@context': 'https://schema.org',
      '@type': 'Article',
      headline: title,
      description: description,
      image: 'https://ailearnings.in/og-image.jpg',
      url: canonical,
      datePublished: date,
      dateModified: updatedAt,
      author: { '@type': 'Person', name: authorName, url: 'https://ailearnings.in/' },
      publisher: { '@type': 'Organization', name: 'AI Learning Hub', url: 'https://ailearnings.in/' },
      mainEntityOfPage: { '@type': 'WebPage', '@id': canonical },
    }, null, 2);

    const breadcrumbSchema = JSON.stringify({
      '@context': 'https://schema.org',
      '@type': 'BreadcrumbList',
      itemListElement: [
        { '@type': 'ListItem', position: 1, name: 'Home', item: 'https://ailearnings.in/' },
        { '@type': 'ListItem', position: 2, name: 'Blog', item: 'https://ailearnings.in/blog/' },
        { '@type': 'ListItem', position: 3, name: title, item: canonical },
      ],
    }, null, 2);

    // Fill template
    let html = postTemplate
      .replace(/\{\{TITLE\}\}/g,           esc(title))
      .replace(/\{\{TITLE_SHORT\}\}/g,      esc(title.length > 50 ? title.slice(0, 47) + '…' : title))
      .replace(/\{\{DESCRIPTION\}\}/g,      esc(description))
      .replace(/\{\{CANONICAL\}\}/g,        canonical)
      .replace(/\{\{DATE\}\}/g,             date)
      .replace(/\{\{DATE_DISPLAY\}\}/g,     formatDate(date))
      .replace(/\{\{READ_TIME\}\}/g,        String(mins))
      .replace(/\{\{AUTHOR_NAME\}\}/g,      esc(authorName))
      .replace(/\{\{AUTHOR_TITLE\}\}/g,     esc(authorTitle))
      .replace(/\{\{AUTHOR_INITIALS\}\}/g,  authorInitials)
      .replace(/\{\{UPDATED_META\}\}/g,     updatedMeta)
      .replace(/\{\{AUTHOR_BIO\}\}/g,       esc(authorBio))
      .replace(/\{\{CONTENT\}\}/g,          htmlContent)
      .replace(/\{\{INLINE_CSS\}\}/g,       inlineCss)
      .replace(/\{\{ARTICLE_SCHEMA\}\}/g,   articleSchema)
      .replace(/\{\{BREADCRUMB_SCHEMA\}\}/, breadcrumbSchema);

    // Restore non-blocking font (same fix as generate-static.js)
    html = html.replace(
      /(<link[^>]*fonts\.bunny\.net[^>]*)\bmedia="all"([^>]*>)/,
      '$1media="print"$2'
    );

    const outDir  = path.join(BLOG_OUT, slug);
    const outFile = path.join(outDir, 'index.html');
    fs.mkdirSync(outDir, { recursive: true });
    fs.writeFileSync(outFile, html, 'utf8');

    const kb = Math.round(Buffer.byteLength(html, 'utf8') / 1024);
    console.log(`   ✓ Wrote blog/${slug}/index.html (${kb} KB)`);

    posts.push({ slug, title, description, date, mins });
  }

  // Sort posts newest-first
  posts.sort((a, b) => b.date.localeCompare(a.date));

  // blog/index.html is generated by generate-static.js (Puppeteer) to match the React layout.
  // Do not write it here to avoid overwriting with a mismatched template layout.
  console.log(`   ℹ blog/index.html skipped — regenerate with: node scripts/generate-static.js`);

  // ── Roadmap Guides ───────────────────────────────────────────────────────────
  if (fs.existsSync(ROADMAP_GUIDES_DIR)) {
    const rgFiles = fs.readdirSync(ROADMAP_GUIDES_DIR).filter(f => f.endsWith('.md')).sort();
    const rgOut   = path.join(BLOG_OUT, 'roadmap-guides');
    const rgPosts = [];

    for (const file of rgFiles) {
      const raw = fs.readFileSync(path.join(ROADMAP_GUIDES_DIR, file), 'utf8');
      const { meta, content } = parseFrontmatter(raw);

      const slug        = meta.slug || file.replace('.md', '');
      const title       = meta.title || slug;
      const description = meta.description || '';
      const date        = meta.date || '2026-03-10';
      const updatedAt   = meta.updatedAt || meta.updated_at || date;

      const rgDay = new Date(date + 'T12:00:00Z').getUTCDay();
      const rgIsWeekday = rgDay >= 1 && rgDay <= 5;
      const rgDefaultAuthor = rgIsWeekday ? 'Mamta Chauhan' : 'AI Learning Hub';
      const rgDefaultTitle = rgIsWeekday ? 'Content Creator and AI Enthusiast' : 'AI Learning Hub';
      const rgDefaultBio = rgIsWeekday
        ? 'Mamta Chauhan is an AI enthusiast and content creator behind ailearnings.in. She writes practical guides on LLMs, RAG, and AI engineering to help developers navigate the fast-moving world of artificial intelligence. Passionate about bridging the gap between cutting-edge research and real-world application.'
        : 'Software engineer building practical AI systems — RAG pipelines, LLM applications, and developer AI tools. Created AI Learning Hub to provide a structured, no-fluff roadmap for developers entering AI.';

      const authorName  = rgIsWeekday ? 'Mamta Chauhan' : (meta.author || rgDefaultAuthor);
      const authorTitle = rgIsWeekday ? 'Content Creator and AI Enthusiast' : (meta.authorTitle || meta.author_title || rgDefaultTitle);
      const canonical   = `https://ailearnings.in/blog/roadmap-guides/${slug}/`;
      const mins        = readTime(content);
      const htmlContent = marked.parse(content);

      const authorInitials = authorName.split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase();
      const authorBio = meta.authorBio || meta.author_bio || rgDefaultBio;
      const updatedMeta = updatedAt !== date
        ? `<span class="sep">·</span><span>Updated <time datetime="${updatedAt}">${formatDate(updatedAt)}</time></span>`
        : '';

      const articleSchema = JSON.stringify({
        '@context': 'https://schema.org',
        '@type': 'Article',
        headline: title,
        description: description,
        url: canonical,
        datePublished: date,
        dateModified: updatedAt,
        author: { '@type': 'Person', name: authorName, url: 'https://ailearnings.in/' },
        publisher: { '@type': 'Organization', name: 'AI Learning Hub', url: 'https://ailearnings.in/' },
        mainEntityOfPage: { '@type': 'WebPage', '@id': canonical },
      }, null, 2);

      const breadcrumbSchema = JSON.stringify({
        '@context': 'https://schema.org',
        '@type': 'BreadcrumbList',
        itemListElement: [
          { '@type': 'ListItem', position: 1, name: 'Home',           item: 'https://ailearnings.in/' },
          { '@type': 'ListItem', position: 2, name: 'Blog',           item: 'https://ailearnings.in/blog/' },
          { '@type': 'ListItem', position: 3, name: 'Roadmap Guides', item: 'https://ailearnings.in/blog/roadmap-guides/' },
          { '@type': 'ListItem', position: 4, name: title,            item: canonical },
        ],
      }, null, 2);

      let html = postTemplate
        .replace(/\{\{TITLE\}\}/g,           esc(title))
        .replace(/\{\{TITLE_SHORT\}\}/g,      esc(title.length > 50 ? title.slice(0, 47) + '…' : title))
        .replace(/\{\{DESCRIPTION\}\}/g,      esc(description))
        .replace(/\{\{CANONICAL\}\}/g,        canonical)
        .replace(/\{\{DATE\}\}/g,             date)
        .replace(/\{\{DATE_DISPLAY\}\}/g,     formatDate(date))
        .replace(/\{\{READ_TIME\}\}/g,        String(mins))
        .replace(/\{\{AUTHOR_NAME\}\}/g,      esc(authorName))
        .replace(/\{\{AUTHOR_TITLE\}\}/g,     esc(authorTitle))
        .replace(/\{\{AUTHOR_INITIALS\}\}/g,  authorInitials)
        .replace(/\{\{UPDATED_META\}\}/g,     updatedMeta)
        .replace(/\{\{AUTHOR_BIO\}\}/g,       esc(authorBio))
        .replace(/\{\{CONTENT\}\}/g,          htmlContent)
        .replace(/\{\{INLINE_CSS\}\}/g,       inlineCss)
        .replace(/\{\{ARTICLE_SCHEMA\}\}/g,   articleSchema)
        .replace(/\{\{BREADCRUMB_SCHEMA\}\}/, breadcrumbSchema);

      html = html.replace(
        /(<link[^>]*fonts\.bunny\.net[^>]*)\bmedia="all"([^>]*>)/,
        '$1media="print"$2'
      );

      const outDir = path.join(rgOut, slug);
      fs.mkdirSync(outDir, { recursive: true });
      fs.writeFileSync(path.join(outDir, 'index.html'), html, 'utf8');
      const kb = Math.round(Buffer.byteLength(html, 'utf8') / 1024);
      console.log(`   ✓ Wrote blog/roadmap-guides/${slug}/index.html (${kb} KB)`);

      rgPosts.push({ slug, title, description, date, mins });
    }

    // Generate blog/roadmap-guides/index.html
    rgPosts.sort((a, b) => b.date.localeCompare(a.date));
    const rgListHtml = rgPosts.map(({ slug, title, description, date, mins }) => `
    <a href="/blog/roadmap-guides/${slug}/" class="group block bg-gray-800/40 rounded-xl p-5 border border-white/8 hover:border-blue-500/30 hover:bg-gray-800/60 transition-all" style="text-decoration:none;">
      <div class="flex items-center justify-between mb-2">
        <span class="text-xs text-gray-500">${formatDate(date)}</span>
        <span class="text-xs text-gray-600">${mins} min read</span>
      </div>
      <h2 class="text-white font-semibold text-base mb-2 leading-snug group-hover:text-blue-300 transition-colors">${esc(title)}</h2>
      <p class="text-gray-400 text-sm leading-relaxed mb-3">${esc(description)}</p>
      <div class="flex items-center gap-1 text-xs text-blue-400">Read guide <span>→</span></div>
    </a>`).join('\n');

    let rgIndexHtml = indexTemplate
      .replace('{{POST_LIST}}', rgListHtml)
      .replace('{{INLINE_CSS}}', inlineCss);
    rgIndexHtml = rgIndexHtml.replace(
      /(<link[^>]*fonts\.bunny\.net[^>]*)\bmedia="all"([^>]*>)/,
      '$1media="print"$2'
    );

    fs.mkdirSync(rgOut, { recursive: true });
    fs.writeFileSync(path.join(rgOut, 'index.html'), rgIndexHtml, 'utf8');
    const rgIdxKb = Math.round(Buffer.byteLength(rgIndexHtml, 'utf8') / 1024);
    console.log(`   ✓ Wrote blog/roadmap-guides/index.html (${rgIdxKb} KB) — ${rgPosts.length} guides`);
  }

  console.log('\n✅ Blog generation complete');
}

main();
