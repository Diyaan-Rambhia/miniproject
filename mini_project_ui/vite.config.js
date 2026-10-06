import { defineConfig } from 'vite'
import react, { reactCompilerPreset } from '@vitejs/plugin-react'
import babel from '@rolldown/plugin-babel'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    babel({ presets: [reactCompilerPreset()] })
  ],
  server: {
    port: 5173,
    proxy: {
      '/health': 'http://127.0.0.1:8000',
      '/events': 'http://127.0.0.1:8000',
      '/score': 'http://127.0.0.1:8000',
      '/explain': 'http://127.0.0.1:8000',
      '/robustness': 'http://127.0.0.1:8000',
    },
  },
})
