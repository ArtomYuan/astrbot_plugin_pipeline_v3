"""astrbot_plugin_pipeline_v3 — 多角色路由演出插件。

单 Bot 私聊场景：LLM 路由判断消息涉及哪个角色（或旁白/无关），
按需用对应人格生成回复。复用 AstrBot 人格系统（persona_manager），
状态（关系/剧情/近期台词）脚本化持久化于 data 目录。

架构（对比 v1/v2 的改进）：
- 完全绕过管道调度：context.llm_generate 直调 + event.send 直发
- priority=100 高于 AngelHeart(-10)，stop_event 终止事件传播后
  AngelHeart 与默认 LLM 管道均不介入
- 持久化数据在 /AstrBot/data/plugin_data/astrbot_plugin_pipeline_v3/
  （防插件重装覆盖）
"""

import asyncio
import json
import os
import re
import time
from typing import Any

from astrbot.api.star import Star, Context, register
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Plain
from astrbot.core.message.message_event_result import MessageChain

# 路由判断提示词（一次 LLM 调用输出归属）
_ROUTER_PROMPT = """你是角色演出路由。判断用户消息涉及哪些角色（可多个）：
- 庄方宜 / 陈千语 / 佩丽卡：消息内容与谁相关（提及名字、问话对象、动作指向）
- 旁白：用户进行场景动作/环境描述，需要叙事回应时输出旁白
- 无关：闲聊/打招呼/与角色无关的内容

只输出 JSON 数组，如 ["庄方宜"] 或 ["庄方宜","陈千语"] 或 ["旁白"] 或 []。
不要输出任何其他内容。"""

# 角色 → 人格 ID（从 AstrBot persona_manager 取；配置在插件配置中）
_DEFAULT_PERSONA_IDS = [
    "庄方宜（RC+YAML满血版本）",
    "陈千语（RC+YAML满血版本）",
    "佩丽卡（RC+YAML满血版本）",
    "旁白",
]


@register(
    "astrbot_plugin_pipeline_v3",
    "ArtomYuan",
    "多角色路由演出（按需触发/静默/旁白）",
    "0.1.0",
    "https://github.com/ArtomYuan/astrbot_plugin_pipeline_v3",
)
class PipelineV3Plugin(Star):
    def __init__(self, context: Context, config: dict = None) -> None:
        super().__init__(context)
        self._config = config or {}
        # 持久化数据目录（官方要求：data 目录，防重装覆盖）
        self._data_dir = "/AstrBot/data/plugin_data/astrbot_plugin_pipeline_v3"
        os.makedirs(self._data_dir, exist_ok=True)
        self._persona_ids = self._config.get("persona_ids", _DEFAULT_PERSONA_IDS)

    # ---------- 状态持久化 ----------
    def _load_state(self, name: str, default: Any) -> Any:
        path = os.path.join(self._data_dir, name)
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default

    def _save_state(self, name: str, data: Any) -> None:
        path = os.path.join(self._data_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)

    # ---------- 人格获取 ----------
    def _get_persona_prompt(self, persona_id: str) -> str | None:
        try:
            p = self.context.persona_manager.get_persona_v3_by_id(persona_id)
            if p:
                return p.get("system_prompt") or p.get("prompt") or ""
        except Exception:
            pass
        return None

    # ---------- LLM 调用 ----------
    async def _llm(self, prompt: str, system_prompt: str = "") -> str:
        try:
            resp = await self.context.llm_generate(
                prompt=prompt, system_prompt=system_prompt or None
            )
            if resp and getattr(resp, "completion_text", None):
                return resp.completion_text.strip()
        except Exception as exc:  # 错误处理：不让插件崩溃
            self.context.logger.error(f"pipeline_v3 LLM 调用失败: {exc}")
        return ""

    # ---------- 路由判断 ----------
    async def _route(self, message_text: str) -> list[str]:
        """LLM 判断消息归属：返回角色名列表（[] = 无关静默）。"""
        raw = await self._llm(_ROUTER_PROMPT, system_prompt="你是一个严格的 JSON 输出器。")
        m = re.search(r"\[.*?\]", raw or "", re.S)
        if not m:
            return []
        try:
            names = json.loads(m.group(0))
            valid = {p.split("（")[0] for p in self._persona_ids}
            return [n for n in names if n in valid or n == "旁白"]
        except Exception:
            return []

    # ---------- 主处理 ----------
    @filter.event_message_type(
        filter.EventMessageType.PRIVATE_MESSAGE, priority=100
    )
    async def route_handler(self, event: AstrMessageEvent) -> None:
        """私聊消息：路由 → 按需生成角色回复 / 静默。"""
        message_text = event.get_message_outline().strip()
        if not message_text:
            return

        # 1. 路由判断
        try:
            targets = await self._route(message_text)
        except Exception:
            targets = []
        if not targets:
            # 无关 → 静默（终止事件传播，默认管道也不响应）
            event.stop_event()
            event.should_call_llm(False)
            return

        # 2. 终止默认管道（防 AngelHeart/默认 LLM 介入）
        event.stop_event()
        event.should_call_llm(False)

        # 3. 状态注入（关系/近期台词 → 上下文）
        relationships = self._load_state("relationships.json", {})
        recent = self._load_state("recent_lines.json", [])
        state_hint = ""
        if relationships:
            state_hint += f"\n[关系状态] {json.dumps(relationships, ensure_ascii=False)}"
        if recent:
            state_hint += f"\n[近期台词] {' / '.join(recent[-6:])}"

        # 4. 每个响应者生成并发送
        for persona_id in self._persona_ids:
            short = persona_id.split("（")[0]
            if short not in targets:
                continue
            system_prompt = self._get_persona_prompt(persona_id)
            if not system_prompt:
                continue
            reply = await self._llm(message_text + state_hint, system_prompt=system_prompt)
            if reply:
                await event.send(MessageChain([Plain(f"[{short}] {reply}")]))
                # 状态更新（脚本化——仅追加台词，关系更新后续版本）
                lines = self._load_state("recent_lines.json", [])
                lines.append(f"{short}: {reply[:120]}")
                self._save_state("recent_lines.json", lines[-60:])
            await asyncio.sleep(0.3)
