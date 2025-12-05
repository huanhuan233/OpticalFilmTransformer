import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// 说明：
// 1) 配好下面的 proxy 后，前端代码里的 fetch 保持相对路径：
//    fetch('/api/optogpt/infer/', ...)
// 2) 若你使用了我在 OptoTargets.vue 里加的 API_BASE 开关，请在 .env.local 里设置：
//    VITE_API_BASE=
//    （空字符串，表示“走代理”，最终发起请求就是 '/api/...'）

export default defineConfig({
  plugins: [vue()],
  server: {
    host: true,          // 允许外网访问（容器/局域网）
    port: 8173,          // 你当前前端端口
    strictPort: true,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8174', // ← 改成后端实际地址（IP 或主机名）
        changeOrigin: true,
        // 如果后端真实路径没有 '/api' 前缀，就放开下面一行做重写
        // rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
