import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Allow dev from localhost, LAN, or production domain
    allowedHosts: ['app.bimba3d.com', 'localhost', '127.0.0.1'],
    host: true,
    hmr: {
      // Use env or fallback to current hostname for HMR websocket
      host: process.env.VITE_HMR_HOST || undefined,
      port: process.env.VITE_HMR_PORT ? Number(process.env.VITE_HMR_PORT) : undefined,
    },
  },
});
