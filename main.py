"""astrbot_plugin_pipeline_v3 — 多角色路由演出插件（判断/生成分离版）。

单 Bot 私聊场景：轮流询问每个角色「这条消息是否在叫你」——
判断层用轻量配置（非思考 + 低温度，速度优先），
命中角色后用主模型（人格 prompt + 状态注入）正常应答。

架构要点：
- 判断层：llm_generate(judge_provider_id, temperature=0, 非思考) —— 是/否 分类
- 生成层：llm_generate(主 provider) —— 人格 + 状态 → 台词
- context.llm_generate 直调 + event.send 直发（绕过管道调度）
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

# 默认角色列表（WebUI 配置 persona_ids 可覆盖）
_DEFAULT_PERSONA_IDS = [
    "庄方宜（RC+YAML满血版本）",
    "陈千语（RC+YAML满血版本）",
    "佩丽卡（RC+YAML满血版本）",
    "旁白",
]

# 判断问题模板（轮流询问每个角色）
_JUDGE_QUESTION = (
    "判断下面的用户消息是否是在叫你、提及你、问你问题、或与你有关（你在这场对话中）。\n"
    "注意：\n"
    "- 消息中的群体称呼（「你们」「大家」「各位」）：默认视为与你有关，你应当回应——"
    "除非近期对话中有你明确离场/不在场的记录。\n"
    "- 没有你的离场记录 = 你默认在场（即使近期没说过话）。\n"
    "- 直接叫你/单独问你：一定回应。\n"
    "只回答一个字：是 或 否\n\n"
    "近期对话：{context}\n\n"
    "用户消息：{message}"
)


@register(
    "astrbot_plugin_pipeline_v3",
    "ArtomYuan",
    "多角色路由演出（按需触发/静默/旁白）",
    "0.2.0",
    "https://github.com/ArtomYuan/astrbot_plugin_pipeline_v3",
)
class PipelineV3Plugin(Star):
    def __init__(self, context: Context, config: dict = None) -> None:
        super().__init__(context)
        self._config = config or {}
        self._data_dir = "/AstrBot/data/plugin_data/astrbot_plugin_pipeline_v3"
        os.makedirs(self._data_dir, exist_ok=True)
        # 角色列表：优先插件配置（WebUI 可编辑），逗号分隔字符串兼容
        raw = self._config.get("persona_ids") or ""
        if isinstance(raw, str):
            self._persona_ids = [p.strip() for p in raw.split(",") if p.strip()] or _DEFAULT_PERSONA_IDS
        else:
            self._persona_ids = raw or _DEFAULT_PERSONA_IDS
        # 判断层配置
        self._judge_provider = self._config.get("judge_provider_id") or "deepseek_talk/deepseek-v4-flash"
        self._judge_temperature = float(self._config.get("judge_temperature", 0) or 0)
        self._judge_no_reasoning = bool(self._config.get("judge_no_reasoning", True))
        self.logger.info(
            f"pipeline_v3 角色列表: {self._persona_ids} | 判断层: {self._judge_provider} "
            f"temp={self._judge_temperature} 非思考={self._judge_no_reasoning}"
        )

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

    # ---------- 生成层（主模型） ----------
    async def _generate(self, prompt: str, system_prompt: str, umo: str) -> str:
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
            self.logger.error(f"pipeline_v3 生成失败: {exc!r}", exc_info=True)
        return ""

    # ---------- 判断层（轻量：非思考 + 低温度） ----------
    async def _judge(self, persona_id: str, message_text: str, recent_ctx: str, umo: str) -> bool:
        system_prompt = self._get_persona_prompt(persona_id)
        if not system_prompt:
            return False
        question = _JUDGE_QUESTION.format(context=recent_ctx or "（无）", message=message_text)
        try:
            kwargs: dict[str, Any] = {"temperature": self._judge_temperature}
            if self._judge_no_reasoning:
                kwargs["thinking"] = {"type": "disabled"}  # DeepSeek 非思考模式
            resp = await self.context.llm_generate(
                chat_provider_id=self._judge_provider,
                prompt=question,
                system_prompt=system_prompt,
                **kwargs,
            )
            text = (resp.completion_text or "").strip() if resp else ""
            self.logger.debug(f"pipeline_v3 判断 [{persona_id}]: {text[:30]!r}")
            return text.startswith("是")
        except Exception as exc:
            # 非思考参数可能不被所有 provider 支持——重试不带
            self.logger.warning(f"pipeline_v3 判断失败（重试不带非思考参数）: {exc!r}")
            try:
                resp = await self.context.llm_generate(
                    chat_provider_id=self._judge_provider,
                    prompt=question,
                    system_prompt=system_prompt,
                    temperature=self._judge_temperature,
                )
                text = (resp.completion_text or "").strip() if resp else ""
                return text.startswith("是")
            except Exception as exc2:
                self.logger.error(f"pipeline_v3 判断重试失败: {exc2!r}")
                return False

    # ---------- 主处理 ----------
    @filter.event_message_type(
        filter.EventMessageType.PRIVATE_MESSAGE, priority=100
    )
    async def route_handler(self, event: AstrMessageEvent) -> None:
        """私聊消息：轮流判断各角色是否被叫 → 命中的用主模型应答。"""
        message_text = event.get_message_outline().strip()
        if not message_text:
            return
        self.logger.info(f"pipeline_v3 收到私聊: {message_text[:60]!r}")

        # 上下文（近期台词——判断「在场/你们指代谁」的依据）
        recent_ctx = " / ".join(self._load_state("recent_lines.json", [])[-20:])
        state_hint = ""
        relationships = self._load_state("relationships.json", {})
        if relationships:
            state_hint += f"\n[关系状态] {json.dumps(relationships, ensure_ascii=False)}"

        # 1. 判断层：轮流询问每个角色「是否在叫你」
        targets = []
        for persona_id in self._persona_ids:
            if await self._judge(persona_id, message_text, recent_ctx, event.unified_msg_origin):
                targets.append(persona_id)
            await asyncio.sleep(0.1)
        self.logger.info(f"pipeline_v3 命中: {targets}")

        if not targets:
            # 全部无关 → 静默（终止事件传播，默认管道也不响应）
            event.stop_event()
            event.should_call_llm(False)
            return

        # 2. 终止默认管道
        event.stop_event()
        event.should_call_llm(False)

        # 3. 生成层：命中的角色用主模型应答
        for persona_id in targets:
            system_prompt = self._get_persona_prompt(persona_id)
            if not system_prompt:
                continue
            reply = await self._generate(
                message_text + state_hint, system_prompt=system_prompt,
                umo=event.unified_msg_origin,
            )
            if reply:
                short = persona_id.split("（")[0]
                await event.send(MessageChain([Plain(f"[{short}] {reply}")]))
                lines = self._load_state("recent_lines.json", [])
                lines.append(f"{short}: {reply[:120]}")
                self._save_state("recent_lines.json", lines[-60:])
            await asyncio.sleep(0.3)
