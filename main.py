"""astrbot_plugin_pipeline_v3 — 多角色路由演出插件。

单 Bot 私聊场景：轮流询问每个角色「这条消息是否与它有关」——
角色人设自带所有称呼/昵称，插件零维护别名表；
有关 → 该角色直接以身份回应；无关 → 输出 __SILENT__。
判断与生成合一（每角色一次 LLM 调用）。

架构要点：
- context.llm_generate 直调 + event.send 直发（完全绕过管道调度）
- priority=100 高于 AngelHeart(-10)，stop_event 终止事件传播
- 持久化数据在 /AstrBot/data/plugin_data/astrbot_plugin_pipeline_v3/
"""

import asyncio
import json
import os
from typing import Any

from astrbot.api.star import Star, Context, register
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Plain
from astrbot.core.message.message_event_result import MessageChain

# 静默标记：角色判断与己无关时输出（插件识别后丢弃）
_SILENT = "__SILENT__"

# 角色 → 人格 ID（从 AstrBot persona_manager 取；后续加角色只改这里）
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
        self._data_dir = "/AstrBot/data/plugin_data/astrbot_plugin_pipeline_v3"
        os.makedirs(self._data_dir, exist_ok=True)
        # 人格列表：优先插件配置（WebUI 可编辑），逗号分隔字符串兼容
        raw = self._config.get("persona_ids") or ""
        if isinstance(raw, str):
            self._persona_ids = [p.strip() for p in raw.split(",") if p.strip()] or _DEFAULT_PERSONA_IDS
        else:
            self._persona_ids = raw or _DEFAULT_PERSONA_IDS
        self.logger.info(f"pipeline_v3 角色列表: {self._persona_ids}")

    # ---------- 状态持久化 ----------
    def _load_state(self, name: str, default: Any) -> Any:
        try:
            with open(os.path.join(self._data_dir, name), encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default

    def _save_state(self, name: str, data: Any) -> None:
        with open(os.path.join(self._data_dir, name), "w", encoding="utf-8") as f:
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
    async def _llm(self, prompt: str, system_prompt: str = "", umo: str = "") -> str:
        try:
            chat_provider_id = await self.context.get_current_chat_provider_id(umo)
            resp = await self.context.llm_generate(
                chat_provider_id=chat_provider_id,
                prompt=prompt,
                system_prompt=system_prompt or None,
            )
            if resp and getattr(resp, "completion_text", None):
                return resp.completion_text.strip()
        except Exception as exc:
            self.logger.error(f"pipeline_v3 LLM 调用失败: {exc!r}", exc_info=True)
        return ""

    # ---------- 主处理 ----------
    @filter.event_message_type(
        filter.EventMessageType.PRIVATE_MESSAGE, priority=100
    )
    async def route_handler(self, event: AstrMessageEvent) -> None:
        """私聊消息：轮流问各角色是否与己有关——有关回应，无关静默。"""
        message_text = event.get_message_outline().strip()
        if not message_text:
            return
        self.logger.info(f"pipeline_v3 收到私聊: {message_text[:60]!r}")

        # 状态注入（关系/近期台词）
        relationships = self._load_state("relationships.json", {})
        recent = self._load_state("recent_lines.json", [])
        state_hint = ""
        if relationships:
            state_hint += f"\n[关系状态] {json.dumps(relationships, ensure_ascii=False)}"
        if recent:
            state_hint += f"\n[近期台词] {' / '.join(recent[-6:])}"

        # 轮流询问每个角色（判断+生成合一）
        # 上下文：近期台词（判断「是否在场/你们指代谁」的依据）
        recent_ctx = " / ".join(self._load_state("recent_lines.json", [])[-20:])
        replies = []
        for persona_id in self._persona_ids:
            system_prompt = self._get_persona_prompt(persona_id)
            if not system_prompt:
                continue
            is_narrator = persona_id == "旁白"
            if is_narrator:
                # 旁白：只在用户进行场景/动作/环境描述时叙事，对话一律静默
                probe = (
                    f"判断下面的用户消息是否是场景/动作/环境描述（例如推开门的动作、"
                    f"看到的景色、身体感受等）。\n"
                    f"是：以旁白身份描述这个场景/动作带来的叙事效果{state_hint}\n"
                    f"否（普通对话/问候/提问）：只输出 {_SILENT}，不要输出任何其他内容。\n\n"
                    f"用户消息：{message_text}"
                )
            else:
                probe = (
                    f"近期对话（判断你是否在场的依据）：{recent_ctx or '（无）'}\n\n"
                    f"判断下面的用户消息是否与你有关——包括直接叫你、提及你、"
                    f"问你问题、或你在场参与对话。\n"
                    f"注意：\n"
                    f"- 消息中的群体称呼（「你们」「大家」「各位」）：默认视为与你有关，"
                    f"你应当回应——除非近期对话中有你明确离场/不在场的记录。\n"
                    f"- 没有你的离场记录 = 你默认在场（即使近期没说过话）。\n"
                    f"- 直接叫你/单独问你：一定回应。\n"
                    f"有关：直接以你的身份自然回应这条消息{state_hint}\n"
                    f"无关：只输出 {_SILENT}，不要输出任何其他内容。\n\n"
                    f"用户消息：{message_text}"
                )
            reply = await self._llm(probe, system_prompt=system_prompt, umo=event.unified_msg_origin)
            self.logger.info(f"pipeline_v3 [{persona_id}] 返回: {(reply or '')[:60]!r} silent={_SILENT in (reply or '')}")
            if reply and _SILENT not in reply:
                short = persona_id.split("（")[0]
                replies.append((short, reply))
            await asyncio.sleep(0.2)

        if not replies:
            # 全部无关 → 静默（终止事件传播，默认管道也不响应）
            event.stop_event()
            event.should_call_llm(False)
            return

        # 有响应 → 终止默认管道 + 逐个发送
        event.stop_event()
        event.should_call_llm(False)
        for short, reply in replies:
            await event.send(MessageChain([Plain(f"[{short}] {reply}")]))
            lines = self._load_state("recent_lines.json", [])
            lines.append(f"{short}: {reply[:120]}")
            self._save_state("recent_lines.json", lines[-60:])
            await asyncio.sleep(0.3)
