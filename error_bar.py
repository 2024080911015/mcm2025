import pandas as pd
import matplotlib.pyplot as plt
import ast


def visualize_olympic_error_bars_dual_labels(file_path):
    # 1. 数据准备
    df = pd.read_csv(file_path)

    # 解析置信区间
    def parse_ci(ci_str):
        try:
            return ast.literal_eval(ci_str)
        except:
            return [0, 0]

    df['Gold_CI'] = df['Gold_95_CI'].apply(parse_ci)
    df['Total_CI'] = df['Total_Medals_95_CI'].apply(parse_ci)

    # 计算误差
    df['gold_err_l'] = (df['Predicted_Gold'] - df['Gold_CI'].apply(lambda x: x[0])).clip(lower=0)
    df['gold_err_r'] = (df['Gold_CI'].apply(lambda x: x[1]) - df['Predicted_Gold']).clip(lower=0)

    df['total_err_l'] = (df['Predicted_Total_Medals'] - df['Total_CI'].apply(lambda x: x[0])).clip(lower=0)
    df['total_err_r'] = (df['Total_CI'].apply(lambda x: x[1]) - df['Predicted_Total_Medals']).clip(lower=0)

    # 筛选前 15 名并排序
    plot_df = df.nlargest(15, 'Predicted_Total_Medals').sort_values('Predicted_Total_Medals', ascending=True)

    # 2. 绘图 (不共享 Y 轴)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # --- 左图：金牌 ---
    ax1.barh(plot_df['NOC'], plot_df['Predicted_Gold'], color='#FFB84D', alpha=0.85, edgecolor='#E67E22', linewidth=1)
    ax1.errorbar(plot_df['Predicted_Gold'], plot_df['NOC'],
                 xerr=[plot_df['gold_err_l'], plot_df['gold_err_r']],
                 fmt='none', ecolor='#8B4513', capsize=4, elinewidth=1.5, capthick=1.5)
    ax1.set_xlabel('Predicted Gold Medals', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', linestyle='--', alpha=0.4)
    ax1.set_ylabel('Country Code', fontsize=12, fontweight='bold')

    # --- 右图：总奖牌 ---
    ax2.barh(plot_df['NOC'], plot_df['Predicted_Total_Medals'], color='#5DADE2', alpha=0.85, edgecolor='#2874A6', linewidth=1)
    ax2.errorbar(plot_df['Predicted_Total_Medals'], plot_df['NOC'],
                 xerr=[plot_df['total_err_l'], plot_df['total_err_r']],
                 fmt='none', ecolor='#154360', capsize=4, elinewidth=1.5, capthick=1.5)
    ax2.set_xlabel('Predicted Total Medals', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', linestyle='--', alpha=0.4)
    ax2.set_ylabel('Country Code', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('olympic_chart_with_dual_labels.png', dpi=300)
    plt.show()


visualize_olympic_error_bars_dual_labels('prediction_2028_final_with_CI.csv')