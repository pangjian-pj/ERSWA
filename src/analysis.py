"""
偏差分析模块 - 多样性与流行度偏差
Bias Analysis Module: Diversity and Popularity Bias
"""

import numpy as np
from collections import Counter

def calculate_diversity(recommendations, recommender):
    """
    计算推荐列表的多样性
    使用物品间平均相似度的倒数
    
    参数:
        recommendations: 推荐列表 [(item_id, score), ...]
        recommender: 推荐器对象
    
    返回:
        diversity: 多样性分数 (越高越好)
    """
    if len(recommendations) < 2:
        return 0.0
    
    rec_items = [item_id for item_id, _ in recommendations]
    
    # 计算推荐物品间的平均相似度
    similarities = []
    for i in range(len(rec_items)):
        for j in range(i + 1, len(rec_items)):
            item1, item2 = rec_items[i], rec_items[j]
            
            if (item1 in recommender.item_similarity.index and 
                item2 in recommender.item_similarity.columns):
                sim = recommender.item_similarity.loc[item1, item2]
                similarities.append(abs(sim))
    
    if not similarities:
        return 0.0
    
    avg_similarity = np.mean(similarities)
    
    # 多样性 = 1 - 平均相似度
    # 相似度越低，多样性越高
    diversity = 1 - avg_similarity
    
    return diversity

def calculate_popularity_bias(recommendations, recommender):
    """
    计算流行度偏差
    偏差越高，说明推荐越倾向于热门物品
    
    参数:
        recommendations: 推荐列表
        recommender: 推荐器对象
    
    返回:
        popularity_bias: 流行度偏差分数
    """
    rec_items = [item_id for item_id, _ in recommendations]
    
    # 计算推荐物品的平均流行度
    rec_popularity = []
    for item in rec_items:
        if item in recommender.item_popularity:
            rec_popularity.append(recommender.item_popularity[item])
    
    if not rec_popularity:
        return 0.0
    
    avg_rec_pop = np.mean(rec_popularity)
    
    # 计算所有物品的平均流行度
    all_popularity = list(recommender.item_popularity.values())
    avg_all_pop = np.mean(all_popularity)
    
    # 流行度偏差 = 推荐物品平均流行度 / 所有物品平均流行度
    popularity_bias = avg_rec_pop / avg_all_pop if avg_all_pop > 0 else 0
    
    return popularity_bias

def calculate_novelty(recommendations, recommender):
    """
    计算推荐新颖性
    新颖性 = -log2(流行度)的平均值
    
    返回:
        novelty: 新颖性分数 (越高越好)
    """
    rec_items = [item_id for item_id, _ in recommendations]
    
    total_users = len(recommender.user_ratings)
    novelties = []
    
    for item in rec_items:
        if item in recommender.item_popularity:
            popularity = recommender.item_popularity[item]
            # 避免log(0)
            prob = max(popularity / total_users, 1e-10)
            novelty = -np.log2(prob)
            novelties.append(novelty)
    
    return np.mean(novelties) if novelties else 0

def calculate_coverage(all_recommendations, recommender):
    """
    计算目录覆盖率
    覆盖率 = 推荐过的物品数 / 总物品数
    
    参数:
        all_recommendations: 多个用户的推荐列表
    
    返回:
        coverage: 覆盖率
    """
    recommended_items = set()
    for recs in all_recommendations:
        for item_id, _ in recs:
            recommended_items.add(item_id)
    
    total_items = len(recommender.item_ratings)
    coverage = len(recommended_items) / total_items if total_items > 0 else 0
    
    return coverage

def compare_algorithms(test_users, recommenders, algorithm_names, top_k=10):
    """
    对比不同算法的偏差指标
    
    参数:
        test_users: 测试用户列表
        recommenders: 推荐器列表
        algorithm_names: 算法名称列表
        top_k: 推荐数量
    
    返回:
        comparison: DataFrame包含各算法的偏差指标
    """
    results = {
        'Algorithm': [],
        'Diversity': [],
        'Popularity Bias': [],
        'Novelty': [],
        'Coverage': []
    }
    
    for recommender, alg_name in zip(recommenders, algorithm_names):
        print(f"\n评估 {alg_name} 算法...")
        
        diversities = []
        pop_biases = []
        novelties = []
        all_recs = []
        
        for user_id in test_users:
            recs = recommender.recommend(user_id, top_k=top_k)
            if recs:
                all_recs.append(recs)
                diversities.append(calculate_diversity(recs, recommender))
                pop_biases.append(calculate_popularity_bias(recs, recommender))
                novelties.append(calculate_novelty(recs, recommender))
        
        coverage = calculate_coverage(all_recs, recommender)
        
        results['Algorithm'].append(alg_name)
        results['Diversity'].append(np.mean(diversities) if diversities else 0)
        results['Popularity Bias'].append(np.mean(pop_biases) if pop_biases else 0)
        results['Novelty'].append(np.mean(novelties) if novelties else 0)
        results['Coverage'].append(coverage)
        
        print(f"  多样性: {results['Diversity'][-1]:.4f}")
        print(f"  流行度偏差: {results['Popularity Bias'][-1]:.4f}")
        print(f"  新颖性: {results['Novelty'][-1]:.4f}")
        print(f"  覆盖率: {results['Coverage'][-1]:.4f}")
    
    import pandas as pd
    comparison_df = pd.DataFrame(results)
    
    return comparison_df

def generate_bias_report(comparison_df, output_file='bias_analysis_report.txt'):
    """
    生成偏差分析报告
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("推荐系统偏差分析报告\n")
        f.write("Recommender System Bias Analysis Report\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. 指标说明\n")
        f.write("-" * 80 + "\n")
        f.write("多样性 (Diversity): 衡量推荐结果的多样性，越高越好\n")
        f.write("流行度偏差 (Popularity Bias): 衡量推荐倾向于热门物品的程度\n")
        f.write("新颖性 (Novelty): 衡量推荐的新颖程度，越高越好\n")
        f.write("覆盖率 (Coverage): 推荐过的物品占总物品的比例，越高越好\n\n")
        
        f.write("2. 算法对比结果\n")
        f.write("-" * 80 + "\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("3. 分析结论\n")
        f.write("-" * 80 + "\n")
        
        # 找出最佳算法
        best_diversity = comparison_df.loc[comparison_df['Diversity'].idxmax(), 'Algorithm']
        best_novelty = comparison_df.loc[comparison_df['Novelty'].idxmax(), 'Algorithm']
        best_coverage = comparison_df.loc[comparison_df['Coverage'].idxmax(), 'Algorithm']
        lowest_bias = comparison_df.loc[comparison_df['Popularity Bias'].idxmin(), 'Algorithm']
        
        f.write(f"- 多样性最佳: {best_diversity}\n")
        f.write(f"- 新颖性最佳: {best_novelty}\n")
        f.write(f"- 覆盖率最佳: {best_coverage}\n")
        f.write(f"- 流行度偏差最低: {lowest_bias}\n\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"\n偏差分析报告已保存至: {output_file}")