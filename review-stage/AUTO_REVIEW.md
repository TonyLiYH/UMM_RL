# Auto Review Log

## 第 1 轮（2026-08-24）

### 评估

- 分数：4/10
- 结论：not ready

### 审查员原始响应

完整响应见 [review_round_1_raw.md](review_round_1_raw.md)。

### 采取的行动

- 路径一（严格理论对象）：重写 exact、finite-unroll、stop-gradient、implicit 四种对象；把当前私有参数和优化器状态纳入时变值函数；修正 proximal 驻点、局部/全局坐标与 Schur 补代数。
- 路径二（经验近似边界）：保留有限展开作为操作性估计器，要求 fresh meta-batch、实测信赖域接受和随机方向导数置信区间，不把它冒充 exact best response。
- 将 scale-free 改为 conditionally loss-scale invariant，并补出假设、证明和 Adam 混合度量反例边界。
- 增加 MOML、FORUM、gMOBA、WC-MHGD、部分个性化联邦学习的形式化对照；明确当前没有独占的一般双层优化定理，创新性暂定位为 UMM 方法、诊断与公平协议。
- 建立等总计算/数据和等持久共享更新两套公平协议，明确虚拟/持久私有状态与样本计费。
- D0 增加 raw 一阶/二阶 Taylor 基线和防泄漏切分单元；E1 增加功效分析、分层 bootstrap 与小分母规则；E2 增加逐模型 admission gate。
- 路径一（分析修复）：补齐 T1 精确公式、反例和验收阈值。
- 路径二（可执行修复）：按 TDD 增加 NumPy 合成求解器、CLI、配置和一键脚本；8 个测试通过并生成正式 manifest。

### 状态

- 继续第 2 轮。

## 第 2 轮（2026-08-24）

### 评估

- 分数：5/10
- 结论：not ready

### 审查员原始响应

完整响应见 [review_round_2_raw.md](review_round_2_raw.md)。

### 采取的行动

- 路径一（状态收缩）：把完整 T1 拆为 T1a algebraic smoke 与 T1b independent solver/approximation gate；仅 T1a 标记通过，T1b 明确为进行中。
- 路径二（关键代码补强）：按 TDD 增加 selector 合法性、信赖域 attainable gain、双任务 retained-gain negotiation 与独立任务重标后的最优解一致性测试。
- 将 retained gain 明确定位为 specialized normalized Chebyshev，并新增 identical-\(A_i^K\) 的 MOBLO/MGDA、Chebyshev、Nash 决定性基线。
- 补充 HyperFormer、VL-Adapter、progressive shared/private adapter 文献，收束核心问题为同等计算下 compensation 是否提供额外预测/优化价值。
- 将 Protocol A 拆成等 FLOPs 与等 wall-clock 两张结果，新增不可变预算/搜索配置，冻结 surplus allocation、候选数、搜索范围、早停与重试规则。
- 统一 D0 为击败最强 raw Taylor；统一 E1 高低优指标符号和 headroom floor 公式。
- T1a manifest 增加配置哈希、Python/NumPy/SciPy、Git revision 与 dirty flag；删除未实际使用的随机 seed 字段。

### 状态

- 继续第 3 轮。
