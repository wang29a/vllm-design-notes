# token 预算够用，为什么请求仍然进不了本轮

本文基于 vLLM 源码提交 `04c1f4a407962a37d01b573f3a2aba91dd2ed3a9`，证据等级为静态源码分析。块数、请求长度和调度轨迹均为教学简化，未运行 GPU 实验。

接着上一篇，请求 A 要推进 1 个 token，B 的长 prompt 在本轮准备推进 5 个。token budget 已分配好，是否就能执行？还不能。模型每计算一个位置，需要把该位置的 K/V 写入可被后续 attention 访问的物理空间。token 预算表达本轮愿意做多少工作，KV 容量表达这些工作的状态能否继续存在。两者相关，却不是一个配额；即使某轮只推进几个 token，请求可能已经持有很长上下文。

分页 KV 将请求的连续逻辑位置映射到物理块，避免要求每个请求拥有一整段连续显存。用单一 full-attention KV group、block size 为 4 的教学情形看 B：如果它从进度 0 走到位置 5，共需覆盖两个逻辑块。A 若持有 12 个已计算位置，推进到 13 可能越过块边界并需要额外块。这里的“两个”“一个”只是简化算术；实际实现有多个 KV group、不同 attention 规则、lookahead 和可复用前缀，不能把统一的 ceil 公式套到所有模型。

调度器通过 [KVCacheManager.allocate_slots](https://github.com/vllm-project/vllm/blob/04c1f4a407962a37d01b573f3a2aba91dd2ed3a9/vllm/v1/core/kv_cache_manager.py#L371)把逻辑工作交给容量管理。函数先合并请求既有进度、本地前缀命中和外部缓存进度，再区分主模型需要的范围与 lookahead 预留范围。后者可能为未来或待验证工作提前留槽，并不等于已经产生可公开复用的 token。管理器再让 coordinator 按不同 KV group 计算真正需要新增多少块，避免调度器自己复制每种 attention 的分配规则。

对 B 这种等待中的请求，prefix cache 能降低新增计算量，但命中也不是白拿一组块 ID。[get_computed_blocks](https://github.com/vllm-project/vllm/blob/04c1f4a407962a37d01b573f3a2aba91dd2ed3a9/vllm/v1/core/kv_cache_manager.py#L264)寻找完整缓存块，并把最大命中长度限制为请求 token 数减一，因为还需要执行末端位置以得到 logits；块对齐可能使重算范围超过一个 token。缓存查找给出可复用的候选，分配成功后才把这些块纳入请求的持有关系。这样请求拥有的逻辑前缀才对应到本轮有效的物理存储，而不只是“曾经有人计算过”。

[容量检查处](https://github.com/vllm-project/vllm/blob/04c1f4a407962a37d01b573f3a2aba91dd2ed3a9/vllm/v1/core/kv_cache_manager.py#L553)比较 required blocks 与 available blocks，还考虑 reserved blocks，以及满足条件时针对 waiting/preempted 请求的 watermark。若不足，返回 `None`。这个返回值含义是无法完成本次新增分配，不能误读成函数完全没有任何副作用：源码在检查前可能移除当前 attention 不再需要的 skipped blocks。因此这里应理解为准入边界，而不是一个可无条件回滚的通用事务接口。

`None` 回到调度器后，对 running 和 waiting 的处理不同。运行请求的分配在[重试循环](https://github.com/vllm-project/vllm/blob/04c1f4a407962a37d01b573f3a2aba91dd2ed3a9/vllm/v1/core/sched/scheduler.py#L721)里，可能选择可安全抢占的请求释放资源后再试；具体受策略、在途引用和 connector 条件限制。等待请求的[分配入口](https://github.com/vllm-project/vllm/blob/04c1f4a407962a37d01b573f3a2aba91dd2ed3a9/vllm/v1/core/sched/scheduler.py#L1195)若失败，会停止这段准入，而不会把一个没有物理存储支撑的 B 塞进计划。不能概括为“KV 不足总是抢占”，更不能说抢占无成本：计算进度重置后，已接受的 token 历史还在，之后可能需要重算。

成功后管理器采用命中的块，再分配新增块，并限制可缓存的 token 范围。实际边界不能只看为多少位置预留了槽，因为 speculative token 可能被拒绝。[缓存范围上限](https://github.com/vllm-project/vllm/blob/04c1f4a407962a37d01b573f3a2aba91dd2ed3a9/vllm/v1/core/kv_cache_manager.py#L602)取计算范围与 `request.num_tokens` 的较小值，避免把未定稿 draft 按最终 token 序列纳入复用。这是分配、计算和缓存身份三个维度分开的一个具体例子；它不意味着 CPU 调用分配函数时 GPU 已经写完数据，执行依赖仍由运行时维护。

于是 B 的准入条件更清楚了：它不仅要获得本轮计算份额，还要建立足够的物理 KV 映射，并满足模型特有的分配条件。分页提高空间使用的灵活性，缓存减少重复工作，watermark 和预留为未来运行留余地；代价是碎片、引用管理、准入策略与重算之间必须协调。本篇不测量这些策略的收益，也不把缓存命中率当成端到端速度。

待执行的调试可以在 `allocate_slots()` 的容量比较前观察 `num_blocks_to_allocate`、`reserved_blocks`、`watermark_blocks` 与 pool 空闲数，返回后观察请求块表和 `num_scheduled_tokens`。用很小的 KV 容量验证“有 token budget 的 B 仍未进入计划”，再扩大容量保持其他配置相同。若分配返回 None，B 却在同一执行计划获得没有有效映射的输入位置，就能推翻本文的准入解释；若只是队列没推进，需要先区分等待策略、限流和资源失败。这些实验尚未执行。
