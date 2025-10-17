"""
可解释性模块 - 生成推荐解释
Explanation Module for Recommender System
"""

import pandas as pd

def explain_recommendations(user_id, recommendations, recommender):
    """
    为推荐结果生成解释
    
    参数:
        user_id: 用户ID
        recommendations: 推荐列表 [(item_id, score), ...]
        recommender: 推荐器对象
    
    返回:
        explanations: {item_id: explanation_text}
    """
    if user_id not in recommender.user_ratings:
        return {}
    
    # 获取用户历史评分物品
    user_rated_items = recommender.user_ratings[user_id]
    
    # 获取推荐物品与历史物品的相似度信息
    similarity_info = recommender.get_item_similarity_for_user(user_id, recommendations)
    
    explanations = {}
    
    for item_id, pred_score in recommendations:
        if item_id not in similarity_info:
            explanations[item_id] = {
                'text': f"推荐物品 {item_id}（预测评分: {pred_score:.2f}）",
                'similar_items': [],
                'contribution': []
            }
            continue
        
        similar_items = similarity_info[item_id]
        
        if not similar_items:
            explanations[item_id] = {
                'text': f"推荐物品 {item_id}（预测评分: {pred_score:.2f}）",
                'similar_items': [],
                'contribution': []
            }
            continue
        
        # 生成解释文本
        explanation_parts = []
        explanation_parts.append(f"推荐物品 {item_id}（预测评分: {pred_score:.2f}）")
        explanation_parts.append("\n因为你喜欢以下相似物品:")
        
        contributions = []
        for rated_item, similarity, rating in similar_items[:3]:
            contribution = similarity * rating
            explanation_parts.append(
                f"  - 物品 {rated_item} (你的评分: {rating:.1f}, "
                f"相似度: {similarity:.3f}, 贡献: {contribution:.3f})"
            )
            contributions.append({
                'item': rated_item,
                'rating': rating,
                'similarity': similarity,
                'contribution': contribution
            })
        
        explanations[item_id] = {
            'text': '\n'.join(explanation_parts),
            'similar_items': similar_items,
            'contribution': contributions,
            'predicted_score': pred_score
        }
    
    return explanations

def generate_explanation_report(user_id, recommendations, explanations, output_file='explanation_report.txt'):
    """
    生成详细的解释报告
    
    参数:
        user_id: 用户ID
        recommendations: 推荐列表
        explanations: 解释字典
        output_file: 输出文件名
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"{'=' * 70}\n")
        f.write(f"用户 {user_id} 的推荐解释报告\n")
        f.write(f"{'=' * 70}\n\n")
        
        for i, (item_id, score) in enumerate(recommendations, 1):
            f.write(f"推荐 #{i}\n")
            f.write("-" * 70 + "\n")
            
            if item_id in explanations:
                f.write(explanations[item_id]['text'])
                f.write("\n\n")
            else:
                f.write(f"物品 {item_id} (预测评分: {score:.2f})\n")
                f.write("无详细解释信息\n\n")
        
        f.write("=" * 70 + "\n")
    
    print(f"解释报告已保存至: {output_file}")

def calculate_explanation_quality(explanations):
    """
    评估解释质量
    
    返回:
        metrics: {'avg_similar_items': float, 'avg_contribution': float}
    """
    if not explanations:
        return {'avg_similar_items': 0, 'avg_contribution': 0}
    
    total_similar_items = 0
    total_contribution = 0
    count = 0
    
    for item_id, exp in explanations.items():
        if exp['contribution']:
            total_similar_items += len(exp['contribution'])
            total_contribution += sum(c['contribution'] for c in exp['contribution'])
            count += 1
    
    if count == 0:
        return {'avg_similar_items': 0, 'avg_contribution': 0}
    
    return {
        'avg_similar_items': total_similar_items / count,
        'avg_contribution': total_contribution / count
    }