#!/usr/bin/env node
// Inlines dist/app.css into src/index.html
// app.js is kept as an external deferred script for better browser parse performance
// Output: index.html

const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '../..');

let html = fs.readFileSync(path.join(root, 'source/src/index.html'), 'utf8');

const css = fs.readFileSync(path.join(root, 'dist/app.css'), 'utf8');

// Use a function replacer to avoid $ special characters in replacement strings
const r = (str, search, replacement) => str.replace(search, () => replacement);

html = r(html, '<link rel="stylesheet" href="dist/app.css">', `<style>${css}</style>`);

fs.writeFileSync(path.join(root, 'index.html'), html);

const kb = Math.round(fs.statSync(path.join(root, 'index.html')).size / 1024);
console.log(`✓ index.html inlined — ${kb} KB`);
