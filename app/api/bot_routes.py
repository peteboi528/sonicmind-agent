"""Bot Webhook 路由：飞书 + 微信公众号。

路由端点：
  POST /webhook/feishu   — 飞书事件回调
  GET  /webhook/wechat   — 微信签名验证
  POST /webhook/wechat   — 微信消息回调

凭证为空时适配器不会实例化，路由注册但不生效。
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Query, Request, Response

from app.adapters.base import answer_to_bot_response
from app.adapters.feishu_adapter import FeishuAdapter
from app.adapters.wechat_adapter import WeChatAdapter
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["bot"])

# ---- 适配器实例化（按配置） ----

feishu: FeishuAdapter | None = None
wechat: WeChatAdapter | None = None

if settings.feishu_app_id and settings.feishu_app_secret:
    feishu = FeishuAdapter(
        app_id=settings.feishu_app_id,
        app_secret=settings.feishu_app_secret,
        verification_token=settings.feishu_verification_token,
        encrypt_key=settings.feishu_encrypt_key,
    )
    logger.info("Feishu adapter enabled (app_id=%s...)", settings.feishu_app_id[:6])

if settings.wechat_token and settings.wechat_app_id:
    wechat = WeChatAdapter(
        token=settings.wechat_token,
        app_id=settings.wechat_app_id,
        app_secret=settings.wechat_app_secret,
    )
    logger.info("WeChat adapter enabled (app_id=%s...)", settings.wechat_app_id[:6])


# ---- 飞书 Webhook ----


@router.post("/webhook/feishu")
async def feishu_webhook(request: Request):
    """飞书事件回调。

    处理两种场景：
    1. URL verification challenge → 直接回复 {"challenge": "..."}
    2. 消息事件 → 异步处理 + 回复
    """
    if not feishu:
        return {"error": "Feishu not configured"}

    body = await request.body()
    headers = {k: v for k, v in request.headers.items()}

    # 验签：拒绝伪造请求（未配 encrypt_key/verification_token 时内部会放行并告警）
    if not feishu.verify_request(body, headers):
        logger.warning("Feishu webhook: signature verification failed")
        return Response(content="invalid signature", status_code=403)

    # 1) Challenge
    challenge_resp = feishu.handle_challenge(body)
    if challenge_resp is not None:
        return challenge_resp

    # 2) 消息事件
    msg = feishu.parse_request(body, headers)
    if not msg:
        return {"ok": True}

    # 异步处理（不阻塞飞书 webhook）
    asyncio.create_task(_handle_feishu_message(msg))
    return {"ok": True}


async def _handle_feishu_message(msg):
    """后台处理飞书消息：调 agent → 发卡片。"""
    try:
        from app.api.main import agent as _agent

        answer = await _agent.chat_async(msg.user_id, msg.content)
        response = answer_to_bot_response(answer)

        # 提取 open_id（去掉 feishu_ 前缀）
        open_id = msg.user_id.replace("feishu_", "", 1)
        feishu.send_reply(open_id, response)
    except Exception:
        logger.exception("Feishu message handling failed")


# ---- 微信公众号 Webhook ----


@router.get("/webhook/wechat")
def wechat_verify(
    signature: str = Query(""),
    timestamp: str = Query(""),
    nonce: str = Query(""),
    echostr: str = Query(""),
):
    """微信签名验证（GET 请求）。"""
    if not wechat:
        return Response(content="not configured", status_code=503)

    if wechat.verify_signature(timestamp, nonce, signature):
        return Response(content=echostr, media_type="text/plain")

    return Response(content="invalid signature", status_code=403)


@router.post("/webhook/wechat")
async def wechat_message(request: Request):
    """微信消息回调（POST 请求）。

    策略：立即返回 "success"（避免 5 秒超时），后台通过客服 API 回复。
    """
    if not wechat:
        return Response(content="not configured", status_code=503)

    # 验签
    params = dict(request.query_params)
    sig = params.get("signature", "")
    ts = params.get("timestamp", "")
    nonce = params.get("nonce", "")

    if not wechat.verify_signature(ts, nonce, sig):
        return Response(content="invalid signature", status_code=403)

    body = await request.body()
    headers = {k: v for k, v in request.headers.items()}

    msg = wechat.parse_request(body, headers)
    if not msg:
        return Response(content="success", media_type="text/plain")

    # 异步处理
    asyncio.create_task(_handle_wechat_message(msg))
    return Response(content="success", media_type="text/plain")


async def _handle_wechat_message(msg):
    """后台处理微信消息：调 agent → 客服 API 回复。"""
    try:
        from app.api.main import agent as _agent

        answer = await _agent.chat_async(msg.user_id, msg.content)
        response = answer_to_bot_response(answer)

        # 提取 openid（去掉 wechat_ 前缀）
        openid = msg.user_id.replace("wechat_", "", 1)
        await wechat.send_customer_service(openid, response)
    except Exception:
        logger.exception("WeChat message handling failed")
