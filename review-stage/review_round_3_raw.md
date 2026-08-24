---
SCORE: 6
VERDICT: almost
---

### 关键弱点

1. T1a/T1b 状态诚实，但关键数值风险仍留在未完成 T1b：随机 overlap、独立 KKT、整体不定曲率、CG/unroll/diagonal/low-rank 尚未验证。 → 最小修复方案：进入模型实现前完成 T1b，每项使用独立参考、KKT 残差和失败反例。
2. identical-\(A_i^K\) 对照尚未完全冻结；“或 closest executable MOBLO”留下选择空间，公共 hypergradient 接口未具体化。 → 最小修复方案：D0 前指定必跑实现和版本，冻结接口、允许差异及替代顺序；所有 negotiation 读取同一缓存 hypergradient。
3. 公平预算仍缺 schema/validator、method-level resolved allocation、配置哈希和运行时审计；单一 surplus allocation 未必对基线最优。 → 最小修复方案：增加预算 validator/计数器和至少两种预注册 surplus allocation。
4. D0/E1 尚有少量文字不一致：前段仍写优于 cosine，Claim Ladder 仍写 three-seed。 → 最小修复方案：统一为 strongest raw Taylor 和 power-selected confirmatory seeds。
5. 条件尺度证明把 \(r_i\) 的同比例缩放作为假设，略显循环；SLSQP manifest 缺可行性、KKT 和独立参考。 → 最小修复方案：从局部模型缩放推出 \(r_i\) 缩放，并增加约束/KKT/参考误差。

### 优点

1. T1a/T1b 正确拆分，没有冒充完整 T1b 已通过。
2. selector、attainable gain、max-min negotiation 与独立任务重标测试有效。
3. 临时复跑 13 tests passed，manifest 绑定 clean source revision，仓库 clean。
4. 新颖性定位可信，承认 normalized Chebyshev/MOBLO 编码并补齐 PEFT 文献。
5. Protocol A 已拆成等 FLOPs/等 wall-clock，搜索和 surplus 字段有预冻结基础。
6. Proposition A/C 在所列假设下正确，额外数值抽查未发现错误。

### 具体建议

1. 完成 T1b 后再启动 Show-o2 适配器。
2. 建立 identical-\(A_i^K\) 公共缓存接口。
3. 增加预算 schema、哈希、计数与硬失败。
4. 清理 cosine gate 残留。
5. 冻结确认性种子数与最低可检测效应。
6. 增加 negotiation 约束、KKT、迭代数和独立参考误差。
7. evidence 同时记录 source_revision 与 evidence_revision。

