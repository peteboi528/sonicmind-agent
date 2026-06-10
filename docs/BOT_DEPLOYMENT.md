# 飞书 / 微信 Bot 联调与部署指南

> 适配器代码已就绪（`app/adapters/`），本文档是「让它真正跑起来」的步骤。

## 一、安全说明（先读）

飞书 Webhook 现在**强制验签**（`app/adapters/feishu_adapter.py:verify_request`）：

- 配了 `FEISHU_ENCRYPT_KEY` + 请求带 `X-Lark-Signature` → 走 `sha256(timestamp + nonce + encrypt_key + body)` 校验。
- 否则用 `FEISHU_VERIFICATION_TOKEN` 校验 body 内 token。
- **两者都没配** → 放行并打告警。**公网部署必须至少配一项**，否则任何人都能伪造请求打你的 agent。

微信侧验签在路由层 `wechat_verify` / `wechat_message` 已强制（`sha1(sort(token, timestamp, nonce))`）。

## 二、填凭证

在项目根目录 `.env` 填写（留空则对应适配器不会实例化，路由返回未配置）：

```bash
# ---- 飞书 ----
FEISHU_APP_ID=cli_xxxxxxxxxxxx
FEISHU_APP_SECRET=xxxxxxxxxxxxxxxxxxxxxx
FEISHU_VERIFICATION_TOKEN=xxxxxxxxxx     # 事件订阅页的 Verification Token
FEISHU_ENCRYPT_KEY=xxxxxxxxxx            # 事件订阅页的 Encrypt Key（强烈建议配）

# ---- 微信公众号 ----
WECHAT_TOKEN=your_custom_token            # 你在公众号后台自定义的 Token
WECHAT_APP_ID=wxxxxxxxxxxxxx
WECHAT_APP_SECRET=xxxxxxxxxxxxxxxxxxxxxx
```

凭证从这里拿：
- 飞书：[开放平台](https://open.feishu.cn) → 创建企业自建应用 → 凭证与基础信息（App ID/Secret）→ 事件订阅（Verification Token / Encrypt Key）。
- 微信：[公众平台](https://mp.weixin.qq.com) → 开发 → 基本配置（AppID/AppSecret）→ 服务器配置（Token 自定义）。

## 三、启动服务

```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

Bot 端点：
- `POST /webhook/feishu`
- `GET  /webhook/wechat`（签名验证）
- `POST /webhook/wechat`（消息回调）

## 四、内网穿透（本地联调必需）

飞书/微信要回调你的公网地址，本地开发用穿透工具暴露 8000 端口：

```bash
# 方案 A：cpolar（国内稳定，免费够用）
cpolar http 8000

# 方案 B：ngrok
ngrok http 8000
```

拿到形如 `https://xxxx.cpolar.io` 的公网地址，回调 URL 即：
- 飞书事件订阅请求地址：`https://xxxx.cpolar.io/webhook/feishu`
- 微信服务器配置 URL：`https://xxxx.cpolar.io/webhook/wechat`

## 五、联调验证步骤

### 飞书
1. 开放平台 → 事件订阅 → 填请求地址 `.../webhook/feishu` → 飞书会发 `url_verification` challenge，本服务自动应答（含加密解密）。页面显示「校验通过」即成功。
2. 添加事件 `im.message.receive_v1`，开通机器人能力，发布版本。
3. 在飞书里给机器人发消息「推荐几首适合跑步的歌」，应收到歌曲卡片回复。

### 微信
1. 公众平台 → 服务器配置 → 填 URL `.../webhook/wechat` + Token → 提交。微信发 GET 验签，`echostr` 原样返回即「配置成功」。
2. 启用服务器配置。
3. 给公众号发文本消息，5 秒内收到「success」ACK，随后客服消息推送歌曲图文。
   - 注意：客服消息要求公众号已认证，且用户 48 小时内与公众号有交互。

## 六、常见坑

| 现象 | 原因 | 解法 |
|------|------|------|
| 飞书 challenge 校验失败 | 配了 Encrypt Key 但 body 解密失败 | 确认 `.env` 的 `FEISHU_ENCRYPT_KEY` 与平台一致；装了 `cryptography` |
| 飞书返回 403 | 验签不通过 | 检查 timestamp/nonce header 是否被穿透工具透传 |
| 微信验签失败 | Token 不一致或参数顺序错 | `.env` 的 `WECHAT_TOKEN` 必须与公众号后台填的完全一致 |
| 微信收不到回复 | 客服 API 限制 | 公众号需认证；用户需 48h 内有交互 |
| agent 回复慢/超时 | LLM 推理耗时 | 微信已用「立即 ACK + 后台客服推送」规避 5s 超时 |

## 七、生产部署补充（上线前）

- CORS 当前 `allow_origins=["*"]`（`app/api/main.py:38`），生产应收紧到你的前端域名。
- 无用户认证，`user_id` 客户端自填——Bot 场景下用平台 open_id 前缀隔离（`feishu_` / `wechat_`），够用；Web 直连场景需另加鉴权。
