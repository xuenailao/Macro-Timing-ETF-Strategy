# Macro-Timing-ETF-Strategy

• Integrated macroeconomic indicators (e.g., consumption, PMI, real estate, fixed asset investment) and liquidity metrics (e.g., M2, TSF, LPR, repo rates), along with TED spread as a proxy for overseas volatility. Aggregated them using entropy weighting, and applied HP filtering to extract trends and denoise.
• Applied a Markov Regime Switching Model (MSM) using Maximum Likelihood Estimation and the EM algorithm to classify economic and liquidity states as “expansion/contraction” and “loose/tight.” Independently modeled TED spread regimes to determine whether to include foreign assets. Daily state signals were generated and aggregated monthly for asset allocation.
• Constructed a 7-ETF portfolio and imposed weight constraints based on regime states. Used the Black-Litterman model for optimization, combining an equal-weighted market prior with historical mean returns to build the prior Π and subjective view matrix Q. Out-of-sample backtesting (2020.07–2023.08) achieved an annual return of 6.83%, volatility of 5.51%, and Sharpe ratio of 1.23.

• 涵盖消费、PMI、房地产、固定资产投资等宏观经济指标，以及M2、社融、LPR、逆回购等流动性指标，同时引入TED利差作为海外市场波动代理。对所有指标进行正/负向标准化处理后，使用熵值法加权合成经济指数与流动性指数，并对数据进行HP滤波以提取趋势项并降噪。
• 引入马尔科夫区制转换模型（MSM），以最大似然估计和EM算法估计参数，将经济指数与流动性指数分别划分为“上行/下行”“宽松/紧缩”两类状态。同时对TED利差构建独立区制模型，用于判断是否纳入海外资产。状态识别按日生成并按月整合，为配置阶段提供月度状态信号。
• 选取7类可交易ETF产品构建组合，并在不同经济状态下设置权重约束；以Black-Litterman模型为优化核心，结合等权市场组合与历史收益均值构造先验收益 Π与主观看法 Q，观点置信度 Ω由协方差矩阵估算得出。最终通过最大化效用函数求解最优资产配置权重。样本外区间（2020.07–2023.08）策略年化收益达6.83%，波动率5.51%，夏普比率1.23，验证了模型的稳健性和宏观择时能力。
