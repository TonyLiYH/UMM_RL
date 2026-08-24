---
SCORE: 5
VERDICT: not ready
---

### 关键弱点

1. T1 被过早标记为通过。当前只有一个全共享确定性样例的代数检查，不足以覆盖研究计划中的随机任务、重叠图、直接联合最优解、暴力可行性、不定整体曲率、CG/unroll/低秩近似。 → 最小修复方案：改为 T1a algebraic smoke passed、T1b in progress，或补齐全部测试。
2. selector 未验证二进制单选不重复约束；retained gain 没有求解 attainable gain 与 max-min negotiation，重标测试只缩放常数。 → 最小修复方案：验证 selector，实现信赖域 attainable gain 和双任务 negotiation，并比较重标前后的最优解。
3. 新颖性诚实但投稿贡献仍待验证；缺共享/私有 adapter、HyperFormer 类 PEFT MTL，对 retained gain 与 normalized Chebyshev/Nash 区分不足。 → 最小修复方案：补齐对照并将核心贡献收敛到“同等计算下 optimizer-state-aware compensation 是否优于使用相同内层响应的一般 MOBLO/MGDA”。
4. Protocol A 仍有结果相关预算选择空间，复合先到上限会导致样本/更新不一；搜索候选与失败规则未冻结。 → 最小修复方案：每方法预提交预算分配，分别报告等 FLOPs 和等墙钟，冻结搜索空间、试验数和失败重试。
5. D0/公式/manifest 内部不一致。PROGRESS 仍写优于 cosine，E1 公式未使用 headroom floor，低优指标方向未定义，manifest 缺 revision、环境和配置哈希。 → 最小修复方案：统一门与公式，增加环境指纹，并在可引用 revision 后重生成。

### 优点

1. 第一轮的 proximal、驻点、坐标、有限展开状态依赖已真正修复。
2. 条件尺度不变性及失败边界清楚。
3. 新颖性审计诚实纳入 MOBLO/MOML/个性化学习。
4. 公平性、D0 切分、Taylor 基线、统计与 admission gate 已明显改善。
5. 代码可执行且 8 个测试可复跑，但只支持 algebraic smoke。

### 具体建议

1. 拆 T1a/T1b。
2. 增加 PSD 双任务与 overlap 族、直接 KKT 对照。
3. 增加整体 Hessian 不定的 trust-region 拒绝。
4. 实现并测试 attainable gain 与 max-min negotiation。
5. 增加随机 metric/selector 与暴力方向对照。
6. 补 CG/unroll/diagonal/low-rank 或延后到独立 gate。
7. 补 shared/private adapter MTL 与 Nash/Chebyshev 对照。
8. 把预算和 run metadata 变为配置与自动审计输出。

