# 前端（Vue 3 + Vite）开发与构建

SONICMIND 的 Web 主界面用 Vue 3 + Vite 构建。源码在 `frontend/`，构建产物输出到 `app/web/dist/`，由 FastAPI 直接 serve（部署无需 node）。

## 目录结构

```
frontend/
├── package.json          # vue + vite，无重组件库
├── vite.config.js        # 产物输出 app/web/dist，/api 等代理到 8000
├── index.html
└── src/
    ├── main.js
    ├── App.vue           # 侧边栏 + Tab 容器
    ├── api.js            # 统一 fetch 封装 + SSE 解析
    ├── store.js          # reactive 全局状态（userId / 网易云 / 播放器）
    ├── theme.css         # Spotify 深色 design tokens
    └── components/
        ├── Sidebar.vue            # 用户 / 扫码登录 / 导入歌单 / 训练偏好
        ├── ChatTab.vue            # SSE 流式对话 + 歌曲卡片
        ├── DiscoverTab.vue        # 搜索
        ├── DailyTab.vue           # 今日推荐
        ├── LibraryTab.vue         # 我的库（评分/删除/入库）
        ├── PlaylistTab.vue        # 歌单生成/自动分类/管理
        ├── SongCard.vue           # 歌曲卡片（播放/MV/不喜欢）
        ├── PlayerBar.vue          # 底部播放器
        ├── MvOverlay.vue          # MV 浮层
        └── NeteaseLogin.vue       # 扫码登录模态 + 2s 轮询
```

## 开发（热更新）

```bash
cd frontend
npm install          # 首次
npm run dev          # http://localhost:5173，/api 等自动代理到 :8000
```

同时另起后端：`uvicorn app.api.main:app --port 8000`

## 构建（生产）

```bash
cd frontend
npm run build        # 产物写入 app/web/dist/
```

构建后访问 `http://localhost:8000/web` 即为完整界面。`app/web/dist/` 已纳入版本控制，**部署时无需安装 node**。

## 后端配套端点（已实现）

Vue 前端依赖的网易云认证/导入端点在 `app/api/auth_routes.py`：

| 端点 | 用途 |
|------|------|
| `GET /auth/netease/qr/key` | 取扫码 unikey + 二维码 PNG（data URI） |
| `GET /auth/netease/qr/status` | 轮询登录状态，成功时落 cookie |
| `GET /auth/netease/account` | 读已绑定账号（不回传 cookie 明文） |
| `POST /auth/netease/unbind` | 解绑 |
| `POST /playlist/import/netease` | 导入网易云歌单 |
| `GET /playlist/netease/list` | 拉取「我的歌单」 |

## 改了前端后

任何 `frontend/src/` 改动都要 `npm run build` 重新生成 `dist/` 并提交，否则线上看到的还是旧产物。
