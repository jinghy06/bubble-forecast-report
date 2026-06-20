"""
美股纳斯达克与标普500：DTW滑动窗口分析
数据来源：akshare (新浪财经美股指数)
分析逻辑：
  1. 日线数据 → 周线（取每周最后一个交易日的收盘价）
  2. 归一化：起点=100的百分比变化
  3. DTW动态时间规整 + 皮尔逊相关系数 + 百分比变化相关系数
  4. 滑动窗口：以近年数据为模板，在历史数据中以步长滑动
  5. 输出最相似窗口排名并生成图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
import json

# ========================== 配置 ==========================
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
ASSETS_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets')

for d in [RESULTS_DIR, ASSETS_DIR]:
    os.makedirs(d, exist_ok=True)

# 历史数据范围（Yahoo Finance美股数据）
HIST_START = '1990-01-01'
HIST_END = '2005-12-31'
RECENT_START = '2018-01-01'
RECENT_END = '2025-06-30'

# 滑动窗口参数
WINDOW_STEP = 4   # 周步长

# ========================== 数据读取 ==========================

def load_data(symbol):
    """读取CSV数据，返回日线DataFrame（date, close）"""
    filepath = os.path.join(DATA_DIR, f'{symbol}.csv')
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df[['date', 'close']].copy()

def daily_to_weekly(df):
    """日线转周线：取每周最后一个交易日的收盘价"""
    df = df.copy()
    df['week'] = df['date'].dt.to_period('W')
    weekly = df.groupby('week').tail(1).copy()
    weekly = weekly.sort_values('date').reset_index(drop=True)
    weekly['week_idx'] = np.arange(len(weekly))
    return weekly[['date', 'close', 'week_idx']].copy()

def normalize(series, start_val=100):
    """归一化：起点=100的百分比变化"""
    series = np.array(series)
    return start_val * (series / series[0])

# ========================== DTW算法 ==========================

def dtw_distance(s1, s2):
    """
    动态时间规整（Dynamic Time Warping）
    返回DTW距离和累积距离矩阵
    """
    n, m = len(s1), len(s2)
    # 使用局部约束（Sakoe-Chiba band = max(n,m)//4）加速
    band = max(n, m) // 4
    
    # 初始化
    dtw = np.full((n+1, m+1), np.inf)
    dtw[0, 0] = 0
    
    for i in range(1, n+1):
        j_start = max(1, i - band)
        j_end = min(m+1, i + band + 1)
        for j in range(j_start, j_end):
            cost = abs(s1[i-1] - s2[j-1])
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    
    return dtw[n, m]

def pearson_correlation(s1, s2):
    """皮尔逊相关系数"""
    if len(s1) != len(s2) or len(s1) < 2:
        return 0.0
    s1 = np.array(s1)
    s2 = np.array(s2)
    s1_std = np.std(s1)
    s2_std = np.std(s2)
    if s1_std == 0 or s2_std == 0:
        return 0.0
    return np.corrcoef(s1, s2)[0, 1]

def pct_change_correlation(s1, s2):
    """百分比变化相关系数"""
    if len(s1) != len(s2) or len(s1) < 2:
        return 0.0
    p1 = np.diff(s1) / s1[:-1]
    p2 = np.diff(s2) / s2[:-1]
    if np.std(p1) == 0 or np.std(p2) == 0:
        return 0.0
    return np.corrcoef(p1, p2)[0, 1]

def composite_similarity(template, window):
    """
    综合相似度评分
    - 皮尔逊相关：越高越好（max 0.5 weight）
    - DTW距离：越低越好（invert and normalize）
    - 百分比变化相关：越高越好（max 0.2 weight）
    """
    if len(template) != len(window):
        window = window[:len(template)]
    
    # 皮尔逊相关
    pearson = pearson_correlation(template, window)
    
    # 百分比变化相关
    pct_corr = pct_change_correlation(template, window)
    
    # DTW距离（需要归一化到可比范围）
    dtw = dtw_distance(template, window)
    # 将DTW距离转换为相似度分数（越大越好）
    # 使用指数衰减：sim = exp(-dtw / scale)
    scale = max(np.std(template), np.std(window)) * len(template) * 0.5
    if scale == 0:
        scale = 1
    dtw_sim = np.exp(-dtw / scale)
    
    # 综合评分（0-1范围，越高越相似）
    composite = 0.45 * max(0, pearson) + 0.35 * dtw_sim + 0.20 * max(0, pct_corr)
    
    return {
        'composite': composite,
        'pearson': pearson,
        'dtw_distance': int(dtw),
        'dtw_sim': dtw_sim,
        'pct_corr': pct_corr
    }

# ========================== 滑动窗口分析 ==========================

def sliding_window_analysis(template_series, history_series, history_dates, step=4):
    """
    滑动窗口分析
    template_series: 归一化后的近年数据（模板）
    history_series: 归一化后的历史数据
    history_dates: 历史数据对应的日期
    step: 滑动步长（周数）
    """
    n_template = len(template_series)
    n_history = len(history_series)
    
    results = []
    
    for start in range(0, n_history - n_template + 1, step):
        end = start + n_template
        window = history_series[start:end]
        window_dates = history_dates[start:end]
        
        scores = composite_similarity(template_series, window)
        
        results.append({
            'start_date': window_dates.iloc[0].strftime('%Y-%m-%d'),
            'end_date': window_dates.iloc[-1].strftime('%Y-%m-%d'),
            'start_idx': start,
            'end_idx': end - 1,
            **scores
        })
    
    return pd.DataFrame(results)

# ========================== 主分析 ==========================

def analyze_symbol(symbol_name, hist_file, recent_file):
    """对单个指数进行完整分析"""
    print(f"\n{'='*60}")
    print(f"Analyzing {symbol_name}")
    print(f"{'='*60}")
    
    # 1. 加载数据
    hist_df = load_data(hist_file)
    recent_df = load_data(recent_file)
    
    # 2. 筛选日期范围
    hist_df = hist_df[(hist_df['date'] >= HIST_START) & (hist_df['date'] <= HIST_END)].copy()
    recent_df = recent_df[(recent_df['date'] >= RECENT_START) & (recent_df['date'] <= RECENT_END)].copy()
    
    print(f"History: {hist_df['date'].min()} ~ {hist_df['date'].max()} ({len(hist_df)} days)")
    print(f"Recent: {recent_df['date'].min()} ~ {recent_df['date'].max()} ({len(recent_df)} days)")
    
    # 3. 日线转周线
    hist_weekly = daily_to_weekly(hist_df)
    recent_weekly = daily_to_weekly(recent_df)
    
    print(f"History weekly: {len(hist_weekly)} weeks")
    print(f"Recent weekly: {len(recent_weekly)} weeks")
    
    # 4. 归一化
    hist_norm = normalize(hist_weekly['close'].values)
    recent_norm = normalize(recent_weekly['close'].values)
    
    # 5. 滑动窗口分析
    results_df = sliding_window_analysis(
        recent_norm, hist_norm, hist_weekly['date'], step=WINDOW_STEP
    )
    
    # 6. 排序并输出Top 10
    results_df = results_df.sort_values('composite', ascending=False).reset_index(drop=True)
    results_df['rank'] = results_df.index + 1
    
    print(f"\nTop 10 Most Similar Windows ({len(results_df)} windows tested):")
    top10 = results_df.head(10)[['rank', 'start_date', 'end_date', 'composite', 'pearson', 'dtw_distance', 'pct_corr']]
    print(top10.to_string(index=False))
    
    # 7. 保存结果
    out_file = os.path.join(RESULTS_DIR, f'{symbol_name}_similarity_ranking.csv')
    results_df.to_csv(out_file, index=False, encoding='utf-8-sig')
    print(f"Saved: {out_file}")
    
    return {
        'symbol': symbol_name,
        'hist_weekly': hist_weekly,
        'recent_weekly': recent_weekly,
        'hist_norm': hist_norm,
        'recent_norm': recent_norm,
        'results': results_df,
        'top1': results_df.iloc[0] if len(results_df) > 0 else None
    }

# ========================== 图表生成 ==========================

def generate_charts(nasdaq_result, sp500_result):
    """生成三张图表"""
    
    # ---- 图1: 核心对比图（近年 vs 最相似窗口 vs 泡沫窗口）----
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    for idx, (result, ax, title) in enumerate([
        (nasdaq_result, axes[0], 'Nasdaq'),
        (sp500_result, axes[1], 'S&P 500')
    ]):
        symbol = result['symbol']
        hist_norm = result['hist_norm']
        recent_norm = result['recent_norm']
        hist_weekly = result['hist_weekly']
        recent_weekly = result['recent_weekly']
        results = result['results']
        
        # 绘制近年数据
        x_recent = np.arange(len(recent_norm))
        ax.plot(x_recent, recent_norm, label=f'Recent (2018-2025)', color='#c0392b', linewidth=2)
        
        # 绘制最相似窗口（Top 1）
        if len(results) > 0:
            top1 = results.iloc[0]
            start_idx = int(top1['start_idx'])
            end_idx = int(top1['end_idx']) + 1
            window = hist_norm[start_idx:end_idx]
            x_window = np.arange(len(window))
            ax.plot(x_window, window, '--', label=f"Most Similar: {top1['start_date']}~{top1['end_date']}\nPearson={top1['pearson']:.3f}", color='#27ae60', linewidth=2)
        
        ax.set_xlabel('Time (Weeks)', fontsize=12)
        ax.set_ylabel('Normalized Index (Start=100)', fontsize=12)
        ax.set_title(f'{title}: Trend Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(ASSETS_DIR, 'chart_core.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {os.path.join(ASSETS_DIR, 'chart_core.png')}")
    
    # ---- 图2: 扩展历史对比图 ----
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    for idx, (result, title) in enumerate([(nasdaq_result, 'Nasdaq'), (sp500_result, 'S&P 500')]):
        hist_weekly = result['hist_weekly']
        recent_weekly = result['recent_weekly']
        hist_norm = result['hist_norm']
        recent_norm = result['recent_norm']
        results = result['results']
        
        # Top: 历史 + 高亮窗口
        ax1 = axes[idx, 0]
        ax1.plot(hist_weekly['date'], hist_norm, color='#3498db', alpha=0.7, linewidth=1, label='History 2004-2005')
        if len(results) > 0:
            top1 = results.iloc[0]
            start_idx = int(top1['start_idx'])
            end_idx = int(top1['end_idx']) + 1
            ax1.axvspan(hist_weekly['date'].iloc[start_idx], hist_weekly['date'].iloc[end_idx-1], 
                        alpha=0.2, color='green', label=f'Most Similar: {top1["start_date"]}~{top1["end_date"]}')
        ax1.set_title(f'{title}: Full History (2004-2005)', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Bottom: 对齐对比
        ax2 = axes[idx, 1]
        x = np.arange(len(recent_norm))
        ax2.plot(x, recent_norm, label='Recent (2018-2025)', color='#c0392b', linewidth=2)
        
        if len(results) > 0:
            top1 = results.iloc[0]
            start_idx = int(top1['start_idx'])
            end_idx = int(top1['end_idx']) + 1
            window = hist_norm[start_idx:end_idx]
            x_w = np.arange(len(window))
            ax2.plot(x_w, window, '--', label=f'Most Similar ({top1["start_date"]}~{top1["end_date"]})\nPearson={top1["pearson"]:.3f}', color='#27ae60', linewidth=2)
        
        ax2.set_title(f'{title}: Aligned Comparison', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Weeks')
        ax2.set_ylabel('Normalized (Start=100)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Nasdaq vs S&P 500 - Extended Comparison (2004-2005 vs 2018-2025)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(ASSETS_DIR, 'chart_extended.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {os.path.join(ASSETS_DIR, 'chart_extended.png')}")
    
    # ---- 图3: 时间线映射图 ----
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # 历史时间线
    ax.plot([0, 10], [1, 1], 'o-', color='#3498db', linewidth=3, markersize=8, label='History Timeline (2004-2005)')
    ax.text(0, 1.15, '2004', fontsize=12, ha='center', fontweight='bold')
    ax.text(10, 1.15, '2005', fontsize=12, ha='center', fontweight='bold')
    
    # 当前时间线
    ax.plot([0, 10], [0, 0], 'o-', color='#c0392b', linewidth=3, markersize=8, label='Current Timeline (2018-2025)')
    ax.text(0, -0.15, '2018', fontsize=12, ha='center', fontweight='bold')
    ax.text(10, -0.15, '2025', fontsize=12, ha='center', fontweight='bold')
    
    # 连接线（最相似窗口映射）
    for result, color in [(nasdaq_result, '#27ae60'), (sp500_result, '#f39c12')]:
        if len(result['results']) > 0:
            top1 = result['results'].iloc[0]
            # Map historical window position to current timeline
            start_idx = int(top1['start_idx'])
            end_idx = int(top1['end_idx'])
            hist_len = len(result['hist_norm'])
            
            h_start = (start_idx / hist_len) * 10
            h_end = (end_idx / hist_len) * 10
            
            ax.annotate('', xy=(5, 0), xytext=(h_start, 1),
                       arrowprops=dict(arrowstyle='->', color=color, lw=1.5, ls='--'))
            ax.annotate('', xy=(5, 0), xytext=(h_end, 1),
                       arrowprops=dict(arrowstyle='->', color=color, lw=1.5, ls='--'))
    
    ax.set_xlim(-1, 11)
    ax.set_ylim(-0.5, 1.5)
    ax.set_title('History-Current Timeline Mapping', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    plt.tight_layout()
    fig.savefig(os.path.join(ASSETS_DIR, 'timeline.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {os.path.join(ASSETS_DIR, 'timeline.png')}")

# ========================== 保存分析摘要 ==========================

def save_summary(nasdaq_result, sp500_result):
    """保存JSON分析摘要"""
    summary = {
        "analysis_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "data_source": "akshare (Sina Finance US Stock Index)",
        "data_range": {
            "history": f"{HIST_START} ~ {HIST_END}",
            "recent": f"{RECENT_START} ~ {RECENT_END}"
        },
        "methodology": {
            "normalization": "Start=100 percentage change",
            "weekly_conversion": "Last trading day of each week",
            "dtw": "Dynamic Time Warping with Sakoe-Chiba band",
            "similarity": "Composite = 0.45*Pearson + 0.35*DTW_sim + 0.20*pct_change_corr",
            "sliding_window": f"Step = {WINDOW_STEP} weeks"
        },
        "nasdaq": {
            "top_window": {
                "start": nasdaq_result['top1']['start_date'] if nasdaq_result['top1'] is not None else None,
                "end": nasdaq_result['top1']['end_date'] if nasdaq_result['top1'] is not None else None,
                "composite": float(nasdaq_result['top1']['composite']) if nasdaq_result['top1'] is not None else None,
                "pearson": float(nasdaq_result['top1']['pearson']) if nasdaq_result['top1'] is not None else None,
                "dtw_distance": int(nasdaq_result['top1']['dtw_distance']) if nasdaq_result['top1'] is not None else None
            },
            "windows_tested": len(nasdaq_result['results'])
        },
        "sp500": {
            "top_window": {
                "start": sp500_result['top1']['start_date'] if sp500_result['top1'] is not None else None,
                "end": sp500_result['top1']['end_date'] if sp500_result['top1'] is not None else None,
                "composite": float(sp500_result['top1']['composite']) if sp500_result['top1'] is not None else None,
                "pearson": float(sp500_result['top1']['pearson']) if sp500_result['top1'] is not None else None,
                "dtw_distance": int(sp500_result['top1']['dtw_distance']) if sp500_result['top1'] is not None else None
            },
            "windows_tested": len(sp500_result['results'])
        }
    }
    
    with open(os.path.join(RESULTS_DIR, 'analysis_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Saved: {os.path.join(RESULTS_DIR, 'analysis_summary.json')}")

# ========================== 入口 ==========================

if __name__ == '__main__':
    print("="*60)
    print("美股泡沫分析 - DTW滑动窗口分析")
    print("="*60)
    
    # 分析纳斯达克
    nasdaq_result = analyze_symbol('nasdaq', 'nasdaq_1990_2005', 'nasdaq_2018_2025')
    
    # 分析标普500
    sp500_result = analyze_symbol('sp500', 'sp500_1990_2005', 'sp500_2018_2025')
    
    # 生成图表
    print("\n" + "="*60)
    print("Generating charts...")
    print("="*60)
    generate_charts(nasdaq_result, sp500_result)
    
    # 保存摘要
    save_summary(nasdaq_result, sp500_result)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)
