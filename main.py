"""astrbot_plugin_pipeline_v3 — 多角色路由演出插件（链式生成 + 旁白收尾）。

单 Bot 私聊场景：轮流判断「谁被叫」→ 参与角色链式生成（每个角色看到
用户消息 + 之前所有角色的回应——对话自然衔接）→ 旁白基于完整对话收尾
（场景叙事 + 固化 scene.json）。

架构要点：
- 判断层：严格判定（默认否）——点名/群体（按场景在场）/场景描述
- 生成层：链式——角色按序，上下文累积传递
- 旁白层：每轮收尾——生成场景叙事并固化（旁白是场景唯一权威）
- context.llm_generate 直调 + event.send 直发（绕过管道调度）
- priority=100 高于 AngelHeart(-10)，stop_event 终止事件传播
- 30s 超时保护（防 LLM 挂起卡死）
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

# 判断问题模板（严格判定——默认否）
_JUDGE_QUESTION = (
    "判断下面的用户消息是否在叫你——严格判定，默认「否」：\n"
    "1. 消息明确叫你或你的昵称（你的所有称呼）→ 是\n"
    "2. 消息明确向你提问、提及你 → 是\n"
    "3. 消息是群体称呼（「你们」「大家」「各位」）→ "
    "看「当前场景」：场景中列出的在场角色包含你 → 是；"
    "场景中没你，且近期对话也没有你 → 否\n"
    "4. 其他情况（别人之间的对话、闲聊、无关话题）→ 否"
    "——即使你很想参与，也等被叫到\n"
    "拿不准 → 否（宁可静默，不抢话）\n"
    "只回答一个字：是 或 否\n\n"
    "当前场景：{scene}\n\n"
    "近期对话：{context}\n\n"
    "用户消息：{message}"
)

# 旁白收尾 prompt（每轮对话后——场景叙事 + 固化）
_NARRATOR_PROMPT = (
    "基于以下对话，延续当前场景生成一段简洁的叙事（80-120 字），必须包含：\n"
    "1. 人物：在场角色的动作/神态/状态（谁在做什么、有什么反应）\n"
    "2. 地点：当前所处位置与环境\n"
    "3. 事件：本轮对话正在推进的事情/氛围\n"
    "自然收束本轮对话，为下一轮留出衔接。\n"
    "只输出叙事文本，不要输出其他内容。\n\n"
    "当前场景：{scene}\n\n"
    "本轮对话：{conversation}"
)

# 旁白判断（是否场景描述——用户主动描述时）
_NARRATOR_JUDGE = (
    "判断下面的用户消息是否是场景/动作/环境描述（例如推开门的动作、看到的景色、"
    "身体感受等）。\n"
    "是 → 是\n"
    "否（普通对话/问候/提问）→ 否\n"
    "只回答一个字：是 或 否\n\n"
    "用户消息：{message}"
)


@register(
    "astrbot_plugin_pipeline_v3",
    "ArtomYuan",
    "多角色路由演出（链式生成/旁白收尾/场景固化）",
    "0.4.0",
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
        self.logger.info(
            f"pipeline_v3 角色: {self._persona_ids} | 判断层: {self._judge_provider} temp={self._judge_temperature}"
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

    # ---------- LLM 调用（带超时保护） ----------
    async def _call_llm(self, prompt: str, system_prompt: str, umo: str, provider: str | None = None) -> str:
        try:
            if provider is None:
                provider = await self.context.get_current_chat_provider_id(umo)
            resp = await asyncio.wait_for(
                self.context.llm_generate(
                    chat_provider_id=provider,
                    prompt=prompt,
                    system_prompt=system_prompt or None,
                    temperature=self._judge_temperature,
                ),
                timeout=30,
            )
            if resp and getattr(resp, "completion_text", None):
                return resp.completion_text.strip()
        except asyncio.TimeoutError:
            self.logger.warning("pipeline_v3 LLM 调用超时（30s）")
        except Exception as exc:
            self.logger.error(f"pipeline_v3 LLM 调用失败: {exc!r}")
        return ""

    # ---------- 判断层 ----------
    async def _judge(self, persona_id: str, message_text: str, scene: str, recent_ctx: str, umo: str) -> bool:
        system_prompt = self._get_persona_prompt(persona_id)
        if not system_prompt:
            return False
        if persona_id == "旁白":
            # 旁白：仅场景/动作/环境描述触发
            text = await self._call_llm(
                _NARRATOR_JUDGE.format(message=message_text),
                system_prompt, umo, provider=self._judge_provider,
            )
            return text.startswith("是")
        question = _JUDGE_QUESTION.format(
            scene=scene or "（未固化）", context=recent_ctx or "（无）", message=message_text
        )
        text = await self._call_llm(question, system_prompt, umo, provider=self._judge_provider)
        self.logger.debug(f"pipeline_v3 判断 [{persona_id}]: {text[:30]!r}")
        return text.startswith("是")

    # ---------- 主处理 ----------
    @filter.event_message_type(
        filter.EventMessageType.PRIVATE_MESSAGE, priority=100
    )
    async def route_handler(self, event: AstrMessageEvent) -> None:
        """私聊消息：判断参与列表 → 链式生成 → 旁白收尾固化。"""
        message_text = event.get_message_outline().strip()
        if not message_text:
            return
        self.logger.info(f"pipeline_v3 收到私聊: {message_text[:60]!r}")

        # 状态读取：场景 + 近期台词
        scene = self._load_state("scene.json", {}).get("scene", "")
        recent_ctx = " / ".join(self._load_state("recent_lines.json", [])[-20:])

        # 1. 判断层：轮流询问每个角色
        targets = []
        for persona_id in self._persona_ids:
            if await self._judge(persona_id, message_text, scene, recent_ctx, event.unified_msg_origin):
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

        # 3. 链式生成：参与角色按序——每个看到之前的完整对话
        state_hint = f"\n[当前场景] {scene}" if scene else ""
        chain = message_text
        for persona_id in targets:
            system_prompt = self._get_persona_prompt(persona_id)
            if not system_prompt:
                continue
            if persona_id == "旁白":
                # 用户主动场景描述：旁白直接响应（生成场景叙事——含人物/地点/事件）
                reply = await self._call_llm(
                    f"根据用户描述生成场景叙事（80-120 字，包含人物动作/地点环境/事件氛围）：\n{message_text}",
                    system_prompt, event.unified_msg_origin,
                )
            else:
                reply = await self._call_llm(
                    f"{chain}{state_hint}", system_prompt, event.unified_msg_origin
                )
            if reply:
                short = persona_id.split("（")[0]
                await event.send(MessageChain([Plain(f"[{short}] {reply}")]))
                chain += f"\n[{short}] {reply}"
                lines = self._load_state("recent_lines.json", [])
                lines.append(f"{short}: {reply[:120]}")
                self._save_state("recent_lines.json", lines[-60:])
            await asyncio.sleep(0.3)

        # 4. 旁白收尾（总是——单/多角色对话后都生成场景叙事 + 固化）
        narrator_prompt = self._get_persona_prompt("旁白")
        if narrator_prompt:
            narrative = await self._call_llm(
                _NARRATOR_PROMPT.format(scene=scene or "（尚未建立场景）", conversation=chain),
                narrator_prompt, event.unified_msg_origin,
            )
            if narrative:
                await event.send(MessageChain([Plain(f"[旁白] {narrative}")]))
                self._save_state("scene.json", {"scene": narrative})
                lines = self._load_state("recent_lines.json", [])
                lines.append(f"旁白: {narrative[:120]}")
                self._save_state("recent_lines.json", lines[-60:])
                self.logger.info(f"pipeline_v3 场景固化: {narrative[:50]!r}")
