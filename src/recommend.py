"""
推荐算法模块 - 基于物品的协同过滤
Item-based Collaborative Filtering with Multiple Similarity Metrics
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr

class ItemCFRecommender:
    def __init__(self, data_path, similarity='cosine'):
        """
        初始化推荐器
        
        参数:
            data_path: 数据文件路径
            similarity: 相似度度量方法 ('cosine', 'pearson', 'jaccard')
        """
        self.data_path = data_path
        self.similarity_metric = similarity
        self.ratings_matrix = None
        self.item_similarity = None
        self.user_ratings = defaultdict(dict)
        self.item_ratings = defaultdict(dict)
        self.item_popularity = {}
        
    def load_data(self):
        """加载MovieLens数据"""
        # MovieLens 100K格式: user_id \t item_id \t rating \t timestamp
        df = pd.read_csv(
            self.data_path, 
            sep='\t', 
            names=['user_id', 'item_id', 'rating', 'timestamp']
        )
        
        # 创建评分矩阵
        self.ratings_matrix = df.pivot_table(
            index='user_id', 
            columns='item_id', 
            values='rating'
        ).fillna(0)
        
        # 存储用户评分和物品评分
        for _, row in df.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            rating = row['rating']
            self.user_ratings[user_id][item_id] = rating
            self.item_ratings[item_id][user_id] = rating
        
        # 计算物品流行度
        for item_id in self.item_ratings:
            self.item_popularity[item_id] = len(self.item_ratings[item_id])
        
        return df
    
    def calculate_cosine_similarity(self, item1, item2):
        """计算余弦相似度"""
        users1 = set(self.item_ratings[item1].keys())
        users2 = set(self.item_ratings[item2].keys())
        common_users = users1 & users2
        
        if len(common_users) < 2:
            return 0
        
        ratings1 = [self.item_ratings[item1][u] for u in common_users]
        ratings2 = [self.item_ratings[item2][u] for u in common_users]
        
        dot_product = sum(r1 * r2 for r1, r2 in zip(ratings1, ratings2))
        norm1 = np.sqrt(sum(r ** 2 for r in ratings1))
        norm2 = np.sqrt(sum(r ** 2 for r in ratings2))
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (norm1 * norm2)
    
    def calculate_pearson_similarity(self, item1, item2):
        """计算皮尔逊相关系数"""
        users1 = set(self.item_ratings[item1].keys())
        users2 = set(self.item_ratings[item2].keys())
        common_users = users1 & users2
        
        if len(common_users) < 2:
            return 0
        
        ratings1 = [self.item_ratings[item1][u] for u in common_users]
        ratings2 = [self.item_ratings[item2][u] for u in common_users]
        
        try:
            corr, _ = pearsonr(ratings1, ratings2)
            return corr if not np.isnan(corr) else 0
        except:
            return 0
    
    def calculate_jaccard_similarity(self, item1, item2):
        """计算Jaccard相似度"""
        users1 = set(self.item_ratings[item1].keys())
        users2 = set(self.item_ratings[item2].keys())
        
        intersection = len(users1 & users2)
        union = len(users1 | users2)
        
        if union == 0:
            return 0
        
        return intersection / union
    
    def compute_item_similarity(self):
        """计算物品相似度矩阵"""
        items = list(self.item_ratings.keys())
        n_items = len(items)
        
        similarity_matrix = np.zeros((n_items, n_items))
        
        print(f"  计算物品相似度矩阵 ({n_items} 个物品)...")
        
        for i, item1 in enumerate(items):
            if (i + 1) % 100 == 0:
                print(f"    进度: {i+1}/{n_items}")
            
            for j, item2 in enumerate(items):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                elif i < j:
                    if self.similarity_metric == 'cosine':
                        sim = self.calculate_cosine_similarity(item1, item2)
                    elif self.similarity_metric == 'pearson':
                        sim = self.calculate_pearson_similarity(item1, item2)
                    elif self.similarity_metric == 'jaccard':
                        sim = self.calculate_jaccard_similarity(item1, item2)
                    else:
                        sim = 0
                    
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim
        
        self.item_similarity = pd.DataFrame(
            similarity_matrix, 
            index=items, 
            columns=items
        )
    
    def train(self):
        """训练模型"""
        self.load_data()
        self.compute_item_similarity()
        print(f"  模型训练完成 ({self.similarity_metric} 相似度)")
    
    def recommend(self, user_id, top_k=10):
        """
        为用户推荐物品
        
        参数:
            user_id: 用户ID
            top_k: 推荐物品数量
        
        返回:
            [(item_id, predicted_score), ...]
        """
        if user_id not in self.user_ratings:
            return []
        
        user_rated_items = self.user_ratings[user_id]
        all_items = set(self.item_ratings.keys())
        candidate_items = all_items - set(user_rated_items.keys())
        
        predictions = []
        
        for item in candidate_items:
            if item not in self.item_similarity.index:
                continue
            
            # 计算预测评分
            weighted_sum = 0
            similarity_sum = 0
            
            for rated_item, rating in user_rated_items.items():
                if rated_item in self.item_similarity.columns:
                    sim = self.item_similarity.loc[item, rated_item]
                    if sim > 0:
                        weighted_sum += sim * rating
                        similarity_sum += abs(sim)
            
            if similarity_sum > 0:
                predicted_score = weighted_sum / similarity_sum
                predictions.append((item, predicted_score))
        
        # 按预测评分排序
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:top_k]
    
    def get_item_similarity_for_user(self, user_id, recommended_items):
        """获取推荐物品与用户历史物品的相似度信息"""
        if user_id not in self.user_ratings:
            return {}
        
        user_rated_items = self.user_ratings[user_id]
        similarity_info = {}
        
        for rec_item, _ in recommended_items:
            if rec_item not in self.item_similarity.index:
                continue
            
            item_sims = []
            for rated_item, rating in user_rated_items.items():
                if rated_item in self.item_similarity.columns:
                    sim = self.item_similarity.loc[rec_item, rated_item]
                    if sim > 0:
                        item_sims.append((rated_item, sim, rating))
            
            # 按相似度排序
            item_sims.sort(key=lambda x: x[1], reverse=True)
            similarity_info[rec_item] = item_sims[:5]  # 只保留前5个最相似物品
        
        return similarity_info