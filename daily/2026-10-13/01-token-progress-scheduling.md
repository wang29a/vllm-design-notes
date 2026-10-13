# prefill 和 decode 为什么能用同一套 token 进度调度

本文基于 vLLM 源码提交 `04c1f4a407962a37d01b573f3a2aba91dd2ed3a9`，证据等级为静态源码分析。下面的请求数值是教学例子，没有启动模型服务或测量性能。

理解 vLLM V1，最有用的入口是一个请求在每轮还能推进多少 token。假设请求 A 已经生成一段回答，请求 B 刚带着长 prompt 到达。自然的实现是分别维护 prefill 队列和 decode 队列：B 先完成整段 prompt，A 每轮生成一个 token。这个区分符合模型计算的直觉，但当 prompt 必须分块、某些前缀已经缓存、decode 又加入 draft token 时，两个阶段标签不能直接回答本轮输入哪些位置。调度器最终仍须知道每个请求的当前位置、目标位置和可用预算。

[Scheduler.schedule 的开头](https://github.com/vllm-project/vllm/blob/04c1f4a407962a37d01b573f3a2aba91dd2ed3a9/vllm/v1/core/sched/scheduler.py#L555)明确采用进度表示。它没有靠一个 prefill/decode 阶段枚举决定全部工作，而是让 `num_computed_tokens` 追上 `num_tokens_with_spec`。后者由已知 token 序列和 speculative token 构成。这里的“没有两个阶段”只描述调度的统一抽象；执行器仍然需要不同的 attention 元数据，源码也有 prefill 限流、分块和编码器处理，不能据此宣称系统完全没有阶段差异。

在不启用推测解码、异步占位和前缀缓存的简化情形，设 A 的已知序列有 13 个 token，其中 12 个已纳入计算进度，A 的差额就是 1。B 的 prompt 有 10 个 token，进度是 0，差额是 10。如果本轮 token budget 是 6，且其他容量限制允许，运行中的 A 先获得 1 个位置，B 可获得剩下的 5 个。下一轮 B 仍然是同一个请求，只是差额从 10 变成 5。分块 prefill 因而可以表达为一次未追完的进度，而无需把请求拆成多个独立任务。这条轨迹是用于解释状态的教学模型，实际调度还受 running/waiting 顺序、限流、输入预算和资源可用性影响。

真实运行请求的差额在[调度计算处](https://github.com/vllm-project/vllm/blob/04c1f4a407962a37d01b573f3a2aba91dd2ed3a9/vllm/v1/core/sched/scheduler.py#L650)还加上 `num_output_placeholders`，再减去 `num_computed_tokens`。随后源码应用长 prompt 分块阈值、token budget、输入预算与模型长度边界。异步时输出尚未回到 CPU，占位符使调度器仍可表达下一步工作；推测解码时 draft token 扩大候选计算范围。这些字段不能用“输出了多少 token”替代，因为已知 token、被安排计算的 token 和最终交付的 token 并不是同一个数量。

最终的计划是 [SchedulerOutput](https://github.com/vllm-project/vllm/blob/04c1f4a407962a37d01b573f3a2aba91dd2ed3a9/vllm/v1/core/sched/scheduler.py#L1443)，包含每个请求的 `num_scheduled_tokens`，以及新请求、已缓存请求、块信息和附加元数据。它是执行器下一轮工作的输入协议。新请求需要较完整的描述，已驻留的请求可以使用增量状态，减少每轮重发完整上下文；要讲清具体字段更新，还需继续核对对应数据类与 runner，本篇不从“增量”推断实际 CPU 收益。

最容易误读的地方出现在计划返回前。[Scheduler._update_after_schedule](https://github.com/vllm-project/vllm/blob/04c1f4a407962a37d01b573f3a2aba91dd2ed3a9/vllm/v1/core/sched/scheduler.py#L1565)已经增加 `num_computed_tokens`，同时增加 `num_in_flight_tokens`。这使下一次调度能够继续安排进度，但意味着前者在异步与流水线情形包含乐观的在途工作，不能仅凭名字把它当成 GPU 已写完的 KV 数。计划里还要保留推进前的执行视图，否则 runner 可能从错误位置构造输入；后续结果处理再修正拒绝的 draft 和其他异常。

这一抽象的收益是把不同请求长度和生成状态放在同一个预算框架里，代价是进度字段的语义更加精确，也更难凭名称猜测。它并不自动保证公平或低尾延迟：一个请求有差额不代表它本轮一定被选中，选中也不代表物理 KV 够用。第二篇就沿着 B 的 5 个位置继续追问，调度器怎样确认这些工作真的有地方存放。

待执行的验证可以在 `schedule()` 计算差额、构建 SchedulerOutput，以及 `_update_after_schedule()` 前后设置断点，观察 A/B 的 `num_tokens_with_spec`、`num_output_placeholders`、`num_computed_tokens`、`num_in_flight_tokens` 和计划字典。若返回计划前 computed 已增加而模型 future 尚未完成，就能直接推翻“computed 总是已经执行”的解释；若 B 少于剩余 token budget，接着看输入上限、分块阈值、限流和 KV 分配，而不要把差异当成调度器失算。这些是验证方案，本文没有运行这组断点。
