# 开发文档

## 架构

### 判断/生成分离

```
用户消息
  → 判断层：轮流询问每个角色人格「这条消息是否在叫你？」
      · 使用轻量配置（judge_provider_id + temperature=0 + 非思考模式）
      · 输出仅 是/否 —— 低参数 + 非思考保证判定稳定（无生成冲动干扰）
  → 生成层：命中角色用主模型生成
      · 完整人格 prompt + 状态注入（关系/近期台词）→ 台词
  → 无命中 → 静默（stop_event 终止默认管道）
```

### 事件处理

- `@filter.event_message_type(PRIVATE_MESSAGE, priority=100)` —— 只处理私聊，优先级 100
- `stop_event()` 终止事件传播（AngelHeart priority=-10 在 100 之后，不会介入）
- `event.send()` 直接发送（不走管道调度——避免异步竞态）

## 设计决策与踩坑记录

| 版本 | 方案 | 结果 |
|:--|:--|:--|
| v1 | LLM 判断归属 + `yield chain_result` | ❌ 管道竞态——异步 LLM 返回时管道已结束，回复被丢弃 |
| v2 | 关键词匹配 + `event.send` | ❌ 多角色顺序响应不可靠（AngelHeart 预处理 + 管道生命周期干扰） |
| v3 | 判断/生成分离 + `llm_generate` 直调 + `event.send` 直发 | ✅ 绕过管道调度，无竞态 |

### 关键踩坑

1. **`llm_generate()` 必填 `chat_provider_id`**：`context.llm_generate(chat_provider_id=..., prompt=..., system_prompt=...)`——漏传直接 TypeError。获取方式：`await context.get_current_chat_provider_id(umo)`
2. **插件 logger 是 `self.logger`**：不是 `self.context.logger`（后者不存在——异常会被吞，表现为静默失败）
3. **判断+生成合一的缺陷**：角色人设的「生成冲动」会干扰「是否被叫」判定——表现为轮流掉线（每次不同角色不响应）。分离后用轻量非思考模型做纯分类，判定稳定
4. **非思考参数兼容性**：`thinking={"type": "disabled"}` 不被所有 provider 支持——已做容错（失败自动重试不带该参数）
5. **AngelHeart 干扰**：其 handler priority=-10，路由插件 priority=100 先执行 + `stop_event()` 终止传播——无需禁用 AngelHeart（禁用会导致 Bot 完全静默的副作用）
6. **持久化位置**：状态文件必须放 AstrBot data 目录（`plugin_data/`），放插件目录会被重装/更新覆盖
7. **群体称呼在场判定**：没有角色离场记录 = 默认在场（「没说话 ≠ 不在场」）——避免台词少的角色被误判离场

## 状态文件

位于 `<AstrBot>/data/plugin_data/astrbot_plugin_pipeline_v3/`：

| 文件 | 内容 |
|:--|:--|
| `recent_lines.json` | 近期台词历史（滑动窗口，判断上下文 + 状态注入） |
| `relationships.json` | 关系状态（预留——记忆系统阶段实现） |

## 开发流程

1. 本地修改 → 提交推送（GitHub）
2. 部署机插件目录 `git pull`
3. 重启 AstrBot（或 WebUI 重载插件）
4. 日志排查：`docker logs astrbot-core | grep pipeline_v3`

## 路线图

- [ ] 记忆系统（关系图谱/剧情时间线——脚本化更新）
- [ ] 判断层模型可换（OpenAI 兼容端点直连）
