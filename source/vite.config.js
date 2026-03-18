import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [
    react(),
  ],
  build: {
    outDir: '../dist',
    emptyOutDir: false,
    cssCodeSplit: false,
    rolldownOptions: {
      input: './src/main.jsx',
      output: {
        entryFileNames: 'app.js',
        chunkFileNames: 'chunks/[name].js',
        assetFileNames: 'app[extname]',
        advancedChunks: {
          groups: [
            { name: 'vendor', test: /node_modules\/(react|react-dom|scheduler)/ },
          ],
        },
      },
    },
  },
})
