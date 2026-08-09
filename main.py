"""astrbot_plugin_pipeline_v3 — 多角色路由演出插件（判断/生成分离 + 场景固化）。

单 Bot 私聊场景：轮流询问每个角色「这条消息是否在叫你」——
判断层用轻量配置（非思考 + 低温度），命中角色后用主模型应答。

场景固化：旁白是场景唯一权威——每轮对话后基于上下文固化场景状态
（scene.json），所有角色的判断与生成都注入固化场景——避免场景漂移。

架构要点：
- 判断层：llm_generate(judge_provider_id, temperature=0, 非思考) —— 是/否 分类
- 生成层：llm_generate(主 provider) —— 人格 + 场景 + 状态 → 台词
- 固化层：旁白人格（主模型）—— 对话后更新 scene.json；场景变化时输出叙事句
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

# 静默/无变化标记
_SAME = "__SAME__"

# 判断问题模板（轮流询问每个角色）
_JUDGE_QUESTION = (
    "判断下面的用户消息是否是在叫你、提及你、问你问题、或与你有关（你在这场对话中）。\n"
    "注意：\n"
    "- 消息中的群体称呼（「你们」「大家」「各位」）：默认视为与你有关，你应当回应——"
    "除非近期对话中有你明确离场/不在场的记录。\n"
    "- 没有你的离场记录 = 你默认在场（即使近期没说过话）。\n"
    "- 直接叫你/单独问你：一定回应。\n"
    "只回答一个字：是 或 否\n\n"
    "当前场景：{scene}\n\n"
    "近期对话：{context}\n\n"
    "用户消息：{message}"
)

# 场景固化指令（旁白人格执行——每轮对话后）
_SOLIDIFY_PROMPT = (
    "你是场景固化的权威。基于「当前场景」和「本轮对话内容」，判断场景是否发生了变化"
    "（角色移动/时间流逝/环境变化/重要动作等）。\n"
    "如果场景发生了变化：输出新的场景状态（简洁描述，50-100 字，包含地点/时间/环境/"
    "在场角色状态），不要输出其他内容。\n"
    "如果场景没有变化：只输出 {same}\n\n"
    "当前场景：{scene}\n\n"
    "本轮对话：{conversation}"
)


@register(
    "astrbot_plugin_pipeline_v3",
    "ArtomYuan",
    "多角色路由演出（按需触发/静默/旁白/场景固化）",
    "0.3.0",
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
        # 固化层开关（每轮对话后旁白固化场景）
        self._solidify = bool(self._config.get("solidify_scene", True))
        self.logger.info(
            f"pipeline_v3 角色: {self._persona_ids} | 判断层: {self._judge_provider} "
            f"temp={self._judge_temperature} 非思考={self._judge_no_reasoning} 场景固化={self._solidify}"
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
    async def _judge(self, persona_id: str, message_text: str, scene: str, recent_ctx: str, umo: str) -> bool:
        system_prompt = self._get_persona_prompt(persona_id)
        if not system_prompt:
            return False
        question = _JUDGE_QUESTION.format(
            scene=scene or "（未固化）", context=recent_ctx or "（无）", message=message_text
        )
        try:
            kwargs: dict[str, Any] = {"temperature": self._judge_temperature}
            if self._judge_no_reasoning:
                # DeepSeek 部分模型不支持 thinking 参数（会导致请求挂起）——不再直传，
                # 用 temperature=0 + 简短判断 prompt 保证速度；模型本身快即可。
                pass
            resp = await asyncio.wait_for(
                self.context.llm_generate(
                    chat_provider_id=self._judge_provider,
                    prompt=question,
                    system_prompt=system_prompt,
                    **kwargs,
                ),
                timeout=30,
            )
            text = (resp.completion_text or "").strip() if resp else ""
            self.logger.debug(f"pipeline_v3 判断 [{persona_id}]: {text[:30]!r}")
            return text.startswith("是")
        except asyncio.TimeoutError:
            self.logger.warning(f"pipeline_v3 判断超时 [{persona_id}]——按无关处理")
            return False
        except Exception as exc:
            self.logger.warning(f"pipeline_v3 判断失败（重试）: {exc!r}")
            try:
                resp = await asyncio.wait_for(
                    self.context.llm_generate(
                        chat_provider_id=self._judge_provider,
                        prompt=question,
                        system_prompt=system_prompt,
                        temperature=self._judge_temperature,
                    ),
                    timeout=30,
                )
                text = (resp.completion_text or "").strip() if resp else ""
                return text.startswith("是")
            except Exception as exc2:
                self.logger.error(f"pipeline_v3 判断重试失败: {exc2!r}")
                return False

    # ---------- 场景固化（旁白权威——每轮对话后） ----------
    async def _solidify_scene(self, scene: str, conversation: str, umo: str) -> str:
        """返回：新场景描述（变化）或 __SAME__。"""
        narrator_prompt = self._get_persona_prompt("旁白")
        if not narrator_prompt:
            return _SAME
        prompt = _SOLIDIFY_PROMPT.format(same=_SAME, scene=scene or "（尚未建立场景）", conversation=conversation)
        result = await self._generate(prompt, narrator_prompt, umo)
        result = (result or "").strip()
        if not result or result.startswith(_SAME):
            return _SAME
        return result

    # ---------- 主处理 ----------
    @filter.event_message_type(
        filter.EventMessageType.PRIVATE_MESSAGE, priority=100
    )
    async def route_handler(self, event: AstrMessageEvent) -> None:
        """私聊消息：轮流判断 → 命中角色应答 → 旁白固化场景。"""
        message_text = event.get_message_outline().strip()
        if not message_text:
            return
        self.logger.info(f"pipeline_v3 收到私聊: {message_text[:60]!r}")

        # 状态读取：场景 + 近期台词
        scene = self._load_state("scene.json", {}).get("scene", "")
        recent_ctx = " / ".join(self._load_state("recent_lines.json", [])[-20:])
        state_hint = ""
        relationships = self._load_state("relationships.json", {})
        if relationships:
            state_hint += f"\n[关系状态] {json.dumps(relationships, ensure_ascii=False)}"
        if scene:
            state_hint = f"\n[当前场景] {scene}" + state_hint

        # 1. 判断层：轮流询问每个角色「是否在叫你」
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

        # 3. 生成层：命中的角色用主模型应答
        replies = []
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
                replies.append((short, reply))
                # 旁白主动响应（用户场景描述）→ 同步固化到 scene.json
                if persona_id == "旁白":
                    self._save_state("scene.json", {"scene": reply})
                lines = self._load_state("recent_lines.json", [])
                lines.append(f"{short}: {reply[:120]}")
                self._save_state("recent_lines.json", lines[-60:])
            await asyncio.sleep(0.3)

        for short, reply in replies:
            await event.send(MessageChain([Plain(f"[{short}] {reply}")]))
            await asyncio.sleep(0.3)

        # 4. 场景固化：对话后旁白基于上下文更新场景（防漂移）
        if self._solidify and replies:
            conversation = "\n".join(f"{s}: {r}" for s, r in replies)
            new_scene = await self._solidify_scene(scene, conversation, event.unified_msg_origin)
            if new_scene != _SAME:
                self._save_state("scene.json", {"scene": new_scene})
                self.logger.info(f"pipeline_v3 场景固化: {new_scene[:60]!r}")
                # 场景变化 → 输出叙事句（旁白演出收尾）
                await event.send(MessageChain([Plain(f"[旁白] {new_scene}")]))
