# 为什么模型算完以后，还必须回到调度器处理结果

本文基于 vLLM 源码提交 `04c1f4a407962a37d01b573f3a2aba91dd2ed3a9`，证据等级为静态源码分析。所有数值和机制图均为教学示意，没有记录真实运行轨迹。

前两篇让请求 B 取得本轮 token 份额和 KV 空间，执行器现在可以计算了。自然的想法是 GPU 返回什么就直接发给客户端，调度器只负责下一批。这个分工遗漏了一个事实：本轮计划已经改变请求进度，而执行结果还会改变哪些 token 被接受、请求是否结束、资源是否仍能被下一轮使用。模型执行能算出候选结果，却不能单独决定整个请求生命周期。调度、执行与回写因此必须构成闭环。

[EngineCore.step](https://github.com/vllm-project/vllm/blob/04c1f4a407962a37d01b573f3a2aba91dd2ed3a9/vllm/v1/engine/core.py#L634)先调用 schedule，将计划传给 execute_model，再等待 future。某些执行路径返回 None 时，core 还调用 sample_tokens 取得输出，不能简单把 execute_model 的返回值等同于最终客户端 token。随后 core 处理执行期间发生的 abort，才调用 [Scheduler.update_from_output](https://github.com/vllm-project/vllm/blob/04c1f4a407962a37d01b573f3a2aba91dd2ed3a9/vllm/v1/core/sched/scheduler.py#L1900)。这是本文追踪的 step 路径；有 batch queue 与异步调度时，控制时序更加复杂，不能用这段串行伪代码覆盖全部配置。

![计划、执行与状态修正构成闭环；图为教学示意](../../assets/core-loop.svg)

图里的正向箭头分别表示计算计划、模型工作和结果回流，底部回路表示结果改变下一轮请求状态。KV 管理在计划形成时提供容量支撑，结束时又参与回收。箭头表达接口与状态依赖，没有标注实际耗时，也不表示 CPU/GPU 在所有配置中必须串行。

回到 A/B 的简化例子，A 已知 13 个 token，计算进度是 12，本轮计算第 13 个位置并采样出新 token y。计划返回时进度已推进到 13；结果处理追加 y 后，已知序列变成 14，下一轮又出现 1 个位置的差额。因此“一轮生成一个 token”不意味着生成 y 时 y 自身的 KV 也已经写好。B 若只完成 prompt 的中间分块，runner 应返回空的输出 token 列表；B 的计算进度仍推进，但客户端不会因此收到一个假的生成 token。这一区分把模型输入位置与输出序列接了起来。

第一篇提到的乐观进度也在这里得到约束。[回写循环](https://github.com/vllm-project/vllm/blob/04c1f4a407962a37d01b573f3a2aba91dd2ed3a9/vllm/v1/core/sched/scheduler.py#L1948)会减去本轮 `num_in_flight_tokens`，检查请求是否已经消失或结束，并处理 stale output。执行期间若 A 被取消，即使模型已经产生 token，也不能把它当成仍然有效的普通输出继续推进。core 先处理 abort queue，回写又检查生命周期，是两处互相配合的防线；它们说明输出有效性依赖请求当前状态，而不仅依赖 future 是否成功。

推测解码让“计划不等于最终进度”更加明显。假设已有差额为 1，又安排 3 个 draft token，计划为 4 个位置；若最终只接受其中 1 个 draft，2 个 draft 被拒绝，就要修正这 2 个位置对应的乐观进度。[拒绝处理](https://github.com/vllm-project/vllm/blob/04c1f4a407962a37d01b573f3a2aba91dd2ed3a9/vllm/v1/core/sched/scheduler.py#L1987)依据实际接受数计算 rejected，并在输出非 stale 时回退 computed，必要时也修正输出占位。实际每步采样数由配置决定，这个例子只说明拒绝为什么不能忽略。如果直接保留计划进度，下一轮可能跳过应当重新计算的位置。

更细的边界是抢占与在途输出相遇。[Scheduler._preempt_request](https://github.com/vllm-project/vllm/blob/04c1f4a407962a37d01b573f3a2aba91dd2ed3a9/vllm/v1/core/sched/scheduler.py#L1520)会释放相应资源、重置 computed，把在途份额标记为 stale。旧结果回来时不能再把拒绝数扣到已经重置的新进度上；某些模式交付旧输出但不让它改变重置计数，drop 模式则丢弃相应旧输出。不能把 stale 一概解释为“全部丢弃”。这套区分的代价是更多生命周期字段，收益是重算、异步和结果到达顺序变化时仍有明确的状态归属。

最终接受的输出由 [_update_request_with_output](https://github.com/vllm-project/vllm/blob/04c1f4a407962a37d01b573f3a2aba91dd2ed3a9/vllm/v1/core/sched/scheduler.py#L2344)逐个追加，并在构造返回结果前检查停止条件；触发停止时还会裁掉不应交付的剩余 token。之后的[结束分支](https://github.com/vllm-project/vllm/blob/04c1f4a407962a37d01b573f3a2aba91dd2ed3a9/vllm/v1/core/sched/scheduler.py#L2086)处理状态与资源回收。结束请求释放持有关系，不代表缓存块内容立即全部清零，也不代表异步 connector 的所有传输立即结束；这些需要继续追踪 free 和 connector 的细节。

待执行的验证可在 core 调用 update 前后检查 computed、in-flight、已知 token 长度与输出长度，再对 prompt 中间块、正常 decode、执行期间取消三个场景分别观察结果。正常 decode 后已知序列应接纳新 token，中间 prompt 分块不应随意追加输出，已取消请求不应继续按普通活动请求交付。开启 speculative decoding 后记录 scheduled、accepted 与 rejected 的关系，若拒绝后进度没有得到相应修正且不是 stale 分支，就能推翻本篇解释。本文仅提供断点与观察方案，未运行上述场景。
