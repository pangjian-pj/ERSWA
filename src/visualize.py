"""
可视化模块
Visualization Module for Recommender System
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_recommendations(user_id, recommendations, explanations, recommender, output_file='recommendations_explanation.png'):
    """
    可视化推荐结果及解释
    
    参数:
        user_id: 用户ID
        recommendations: 推荐列表
        explanations: 解释字典
        recommender: 推荐器对象
        output_file: 输出文件名
    """
    if not recommendations:
        print(f"用户 {user_id} 没有推荐结果")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：推荐物品的预测评分
    items = [f"Item {item_id}" for item_id, _ in recommendations[:10]]
    scores = [score for _, score in recommendations[:10]]
    
    ax1.barh(items, scores, color='steelblue')
    ax1.set_xlabel('Predicted Score', fontsize=12)
    ax1.set_title(f'Top 10 Recommendations for User {user_id}', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    
    # 右图：推荐解释 - 相似度贡献
    if explanations and recommendations[0][0] in explanations:
        top_item = recommendations[0][0]
        exp = explanations[top_item]
        
        if exp['contribution']:
            contrib_items = [f"Item {c['item']}" for c in exp['contribution']]
            contrib_values = [c['contribution'] for c in exp['contribution']]
            
            ax2.barh(contrib_items, contrib_values, color='coral')
            ax2.set_xlabel('Contribution Score', fontsize=12)
            ax2.set_title(f'Explanation for Top Recommendation (Item {top_item})', 
                         fontsize=14, fontweight='bold')
            ax2.invert_yaxis()
        else:
            ax2.text(0.5, 0.5, 'No explanation available', 
                    ha='center', va='center', fontsize=12)
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
    else:
        ax2.text(0.5, 0.5, 'No explanation available', 
                ha='center', va='center', fontsize=12)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"推荐结果可视化已保存: {output_file}")
    plt.close()

def plot_bias_metrics(comparison_df, output_file='bias_metrics_comparison.png'):
    """
    可视化偏差指标对比
    
    参数:
        comparison_df: 对比结果DataFrame
        output_file: 输出文件名
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['Diversity', 'Popularity Bias', 'Novelty', 'Coverage']
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']
    
    for idx, (metric, color) in enumerate(zip(metrics, colors)):
        ax = axes[idx // 2, idx % 2]
        
        algorithms = comparison_df['Algorithm'].values
        values = comparison_df[metric].values
        
        bars = ax.bar(algorithms, values, color=color, alpha=0.7, edgecolor='black')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=10)
        
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'{metric} Comparison', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 旋转x轴标签
        ax.set_xticklabels(algorithms, rotation=45, ha='right')
    
    plt.suptitle('Bias Metrics Comparison Across Algorithms', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"偏差指标对比图已保存: {output_file}")
    plt.close()

def plot_similarity_heatmap(recommender, top_n=20, output_file='item_similarity_heatmap.png'):
    """
    可视化物品相似度热力图
    
    参数:
        recommender: 推荐器对象
        top_n: 展示前N个物品
        output_file: 输出文件名
    """
    if recommender.item_similarity is None:
        print("相似度矩阵未计算")
        return
    
    # 选择前top_n个物品
    items = list(recommender.item_similarity.index[:top_n])
    similarity_subset = recommender.item_similarity.loc[items, items]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_subset, 
                cmap='RdYlBu_r', 
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                xticklabels=[f"Item {i}" for i in items],
                yticklabels=[f"Item {i}" for i in items])
    
    plt.title(f'Item Similarity Heatmap (Top {top_n} Items)\nSimilarity Metric: {recommender.similarity_metric}', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Items', fontsize=12)
    plt.ylabel('Items', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"相似度热力图已保存: {output_file}")
    plt.close()

def plot_popularity_distribution(recommender, output_file='popularity_distribution.png'):
    """
    可视化物品流行度分布
    
    参数:
        recommender: 推荐器对象
        output_file: 输出文件名
    """
    popularity_values = list(recommender.item_popularity.values())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 直方图
    ax1.hist(popularity_values, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Number of Ratings', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Item Popularity Distribution', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 累积分布
    sorted_pop = sorted(popularity_values, reverse=True)
    cumulative = np.cumsum(sorted_pop) / np.sum(sorted_pop)
    
    ax2.plot(range(len(cumulative)), cumulative, color='coral', linewidth=2)
    ax2.set_xlabel('Item Rank', fontsize=12)
    ax2.set_ylabel('Cumulative Proportion of Ratings', fontsize=12)
    ax2.set_title('Cumulative Popularity Distribution', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.axhline(y=0.8, color='red', linestyle='--', label='80% threshold')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"流行度分布图已保存: {output_file}")
    plt.close()

def plot_diversity_vs_accuracy(test_results, output_file='diversity_accuracy_tradeoff.png'):
    """
    可视化多样性与准确性的权衡
    
    参数:
        test_results: {algorithm: {'diversity': float, 'accuracy': float}}
        output_file: 输出文件名
    """
    algorithms = list(test_results.keys())
    diversities = [test_results[alg]['diversity'] for alg in algorithms]
    accuracies = [test_results[alg].get('accuracy', 0) for alg in algorithms]
    
    plt.figure(figsize=(10, 6))
    
    plt.scatter(diversities, accuracies, s=200, alpha=0.6, c=range(len(algorithms)), 
               cmap='viridis', edgecolors='black', linewidth=2)
    
    for i, alg in enumerate(algorithms):
        plt.annotate(alg, (diversities[i], accuracies[i]), 
                    fontsize=10, ha='right', va='bottom')
    
    plt.xlabel('Diversity Score', fontsize=12)
    plt.ylabel('Accuracy/Quality Score', fontsize=12)
    plt.title('Diversity vs Accuracy Trade-off', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"多样性-准确性权衡图已保存: {output_file}")
    plt.close()