from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
BASE = Path(__file__).resolve().parent
DATA = BASE / 'data' / 'cs-training.csv'
FIG = BASE / 'outputs' / 'figures'
TAB = BASE / 'outputs' / 'tables'
FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
TARGET = 'SeriousDlqin2yrs'
REN = {
'RevolvingUtilizationOfUnsecuredLines':'credit_utilization','age':'age',
'NumberOfTime30-59DaysPastDueNotWorse':'late_30_59','DebtRatio':'debt_ratio',
'MonthlyIncome':'monthly_income','NumberOfOpenCreditLinesAndLoans':'open_credit_lines',
'NumberOfTimes90DaysLate':'late_90_plus','NumberRealEstateLoansOrLines':'real_estate_loans',
'NumberOfTime60-89DaysPastDueNotWorse':'late_60_89','NumberOfDependents':'dependents'}

def load():
    if not DATA.exists():
        raise FileNotFoundError(f'请将 cs-training.csv 放入 {DATA.parent}')
    df = pd.read_csv(DATA)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')].rename(columns=REN)
    return df

def quality(df):
    rep = pd.DataFrame({'dtype':df.dtypes.astype(str),'missing':df.isna().sum(),
                        'missing_rate_pct':(df.isna().mean()*100).round(2),
                        'nunique':df.nunique(dropna=False)})
    rep.to_csv(TAB/'01_data_quality.csv', encoding='utf-8-sig')
    print('shape:', df.shape, 'duplicates:', df.duplicated().sum()); print(rep)

def clean(df):
    x = df.drop_duplicates().copy()
    x = x[x.age.between(18,100)]
    x['monthly_income'] = x.monthly_income.fillna(x.monthly_income.median())
    x['dependents'] = x.dependents.fillna(x.dependents.median())
    x['debt_ratio_w'] = x.debt_ratio.clip(upper=x.debt_ratio.quantile(.99))
    x['credit_utilization_w'] = x.credit_utilization.clip(upper=x.credit_utilization.quantile(.99))
    for c in ['late_30_59','late_60_89','late_90_plus']:
        x.loc[x[c] > 20, c] = np.nan
        x[c] = x[c].fillna(x[c].median())
    x.to_csv(TAB/'02_cleaned_data.csv', index=False, encoding='utf-8-sig')
    return x

def rate_table(df, col, bins, labels, name):
    z = df[[col,TARGET]].dropna().copy(); z['group'] = pd.cut(z[col], bins=bins, labels=labels, include_lowest=True)
    r = z.groupby('group', observed=False)[TARGET].agg(users='count',defaults='sum',default_rate='mean').reset_index()
    r.default_rate = (r.default_rate*100).round(2); r.to_csv(TAB/name,index=False,encoding='utf-8-sig'); return r

def analyze(df):
    q = np.unique(df.monthly_income.quantile([0,.2,.4,.6,.8,1]).values)
    tables = {
      'age':rate_table(df,'age',[18,29,39,49,59,69,100],['18-29','30-39','40-49','50-59','60-69','70+'],'03_age.csv'),
      'income':rate_table(df,'monthly_income',q,[f'Q{i}' for i in range(1,len(q))],'04_income.csv'),
      'utilization':rate_table(df,'credit_utilization_w',[-np.inf,.2,.4,.6,.8,1,np.inf],['<=20%','20-40%','40-60%','60-80%','80-100%','>100%'],'05_utilization.csv'),
      'open_lines':rate_table(df,'open_credit_lines',[-np.inf,2,5,10,15,20,np.inf],['0-2','3-5','6-10','11-15','16-20','20+'],'06_open_lines.csv')}
    t = df.copy(); t['max_late'] = t[['late_30_59','late_60_89','late_90_plus']].max(axis=1)
    tables['delinquency'] = rate_table(t,'max_late',[-np.inf,0,1,2,3,5,np.inf],['0','1','2','3','4-5','6+'],'07_delinquency.csv')
    return tables

def segment(df):
    x=df.copy(); total_late=x[['late_30_59','late_60_89','late_90_plus']].sum(axis=1)
    cond=[(x.late_90_plus>=2)|(total_late>=4),(x.late_90_plus>=1)|(x.credit_utilization_w>.8),
          (total_late>=1)|(x.credit_utilization_w>.4)|(x.age<30)|(x.monthly_income<x.monthly_income.quantile(.25))]
    x['risk_level']=np.select(cond,['极高风险','高风险','中风险'],default='低风险')
    strategy={'低风险':'自动审批/常规额度','中风险':'收入证明或人工复核/适度控额','高风险':'降低额度/加强审核','极高风险':'拒绝或更严格审核'}
    x['strategy']=x.risk_level.map(strategy)
    s=x.groupby('risk_level')[TARGET].agg(users='count',defaults='sum',default_rate='mean').reset_index(); s.default_rate=(s.default_rate*100).round(2); s['strategy']=s.risk_level.map(strategy)
    s.to_csv(TAB/'08_risk_segments.csv',index=False,encoding='utf-8-sig'); x.to_csv(TAB/'09_data_with_segments.csv',index=False,encoding='utf-8-sig')
    return x,s

def plots(tables, seg):
    sns.set_theme(style='whitegrid')
    for name,t in tables.items():
        plt.figure(figsize=(8,5)); sns.barplot(data=t,x='group',y='default_rate'); plt.xticks(rotation=30); plt.ylabel('Default Rate (%)'); plt.title(name); plt.tight_layout(); plt.savefig(FIG/f'{name}.png',dpi=200); plt.close()
    order=['低风险','中风险','高风险','极高风险']; z=seg.set_index('risk_level').reindex(order).reset_index()
    plt.figure(figsize=(8,5)); sns.barplot(data=z,x='risk_level',y='default_rate'); plt.ylabel('Default Rate (%)'); plt.tight_layout(); plt.savefig(FIG/'risk_segments.png',dpi=200); plt.close()

def main():
    raw=load(); quality(raw); df=clean(raw); tables=analyze(df); _,seg=segment(df); plots(tables,seg)
    print(f'完成：清洗后 {len(df):,} 行，总体违约率 {df[TARGET].mean()*100:.2f}%')
    print('输出目录：', BASE/'outputs')
if __name__=='__main__': main()
