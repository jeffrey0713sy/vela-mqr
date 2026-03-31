# Idea Forecasting — Email Templates (Bilingual)

## Chinese version (directly sendable)

**Subject:** Idea Forecasting 进展更新（时间切分验证已完成）

Hi [Supervisor Name],

我完成了 Vela Summer 2025 的 Idea Forecasting 历史验证，并按照时间切分进行了 out-of-time 测试。

数据与设置：使用 `idea_training.xlsx`（35,823 家公司，2010-2017），标签为 `is_outlier`（>= $250M 代理标准），训练集为 2010-2015，测试集为 2016-2017。测试集正例率为 3.5354%。

结果上，TF-IDF + Logistic Regression 是当前最佳基线：
- AUC = **0.8153**（95% CI: **[0.7939, 0.8417]**）
- PR-AUC = **0.1899**（95% CI: **[0.1545, 0.2302]**）
- Top-decile precision (k=863) = **0.1599**
- Top-decile lift = **4.523x**（95% CI: **[4.005, 5.101]**）

对比的 lexicon baseline 为：AUC 0.7948，PR-AUC 0.1353，Top-decile lift 3.769x。总体说明：在严格时间切分下，idea 文本与成立年份包含可迁移预测信号，且当前模型在未来年份上显著优于随机排序。

我已将完整结果和可复现命令整理到：
- `outputs/idea_forecast_tfidf/summary.md`
- `outputs/idea_forecast_tfidf/metrics.json`
- `outputs/idea_forecasting_progress_20260331.md`

接下来我计划：
1) 引入 Vela Search 时点检索上下文做增强；
2) 小规模对比 LLM scorer；
3) 增加可解释性附录（高权重特征与典型案例）。

Best,  
[Your Name]

---

## English version (directly sendable)

**Subject:** Idea Forecasting update (temporal holdout validation completed)

Hi [Supervisor Name],

I completed the historical validation for the Summer 2025 Idea Forecasting track using a strict temporal split.

Setup: `idea_training.xlsx` (35,823 US companies, founded 2010-2017), label `is_outlier` (>= $250M proxy), train on 2010-2015 and test on 2016-2017. Test positive rate is 3.5354%.

Current best baseline is TF-IDF + Logistic Regression:
- AUC = **0.8153** (95% CI: **[0.7939, 0.8417]**)
- PR-AUC = **0.1899** (95% CI: **[0.1545, 0.2302]**)
- Top-decile precision (k=863) = **0.1599**
- Top-decile lift = **4.523x** (95% CI: **[4.005, 5.101]**)

Compared with the lexicon baseline (AUC 0.7948, PR-AUC 0.1353, top-decile lift 3.769x), this indicates meaningful predictive signal in inception-stage idea text + founding year under out-of-time evaluation.

Full outputs and reproducibility notes are in:
- `outputs/idea_forecast_tfidf/summary.md`
- `outputs/idea_forecast_tfidf/metrics.json`
- `outputs/idea_forecasting_progress_20260331.md`

Next steps:
1) Add time-bounded Vela Search context,
2) Benchmark a cost-controlled LLM scorer slice,
3) Add explainability appendix (top features and representative TP/FP cases).

Best,  
[Your Name]
