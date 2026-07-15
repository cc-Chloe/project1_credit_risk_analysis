# 消费金融平台新用户风险分析与准入策略优化

基于 Kaggle **Give Me Some Credit** 数据集，对约 15 万名用户开展数据清洗、风险因素分析、风险画像与准入策略设计。

## 运行
```bash
pip install -r requirements.txt
python credit_risk_analysis.py
```

## 分析内容
- 缺失值、重复值、异常值与字段类型检查
- MonthlyIncome、NumberOfDependents 缺失值处理
- DebtRatio、信用额度使用率极端值缩尾
- 年龄、收入、历史逾期、信用额度使用率、开放账户数风险分析
- 构建低/中/高/极高风险用户分层与准入建议

## 输出
运行后在 `outputs/` 下生成分析表格和图表。

> 原始数据未上传至仓库，请按 `data/README.md` 下载并放置。
