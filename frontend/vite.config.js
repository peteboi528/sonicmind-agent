import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

// 构建产物输出到 app/web/dist，由 FastAPI StaticFiles 直接 serve。
// 开发时 /api 等后端路径代理到 8000，避免 CORS 与端口割裂。
export default defineConfig({
  plugins: [vue()],
  base: "/web/",
  build: {
    outDir: "../app/web/dist",
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/agent": "http://localhost:8000",
      "/chat": "http://localhost:8000",
      "/search": "http://localhost:8000",
      "/recommend": "http://localhost:8000",
      "/playlist": "http://localhost:8000",
      "/playlists": "http://localhost:8000",
      "/assets": "http://localhost:8000",
      "/rate": "http://localhost:8000",
      "/listen": "http://localhost:8000",
      "/memory": "http://localhost:8000",
      "/taste": "http://localhost:8000",
      "/feedback": "http://localhost:8000",
      "/library": "http://localhost:8000",
      "/auth": "http://localhost:8000",
      "/api": "http://localhost:8000",
    },
  },
});
