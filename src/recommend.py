"""
推荐算法模块 - 多种推荐算法实现
Multiple Recommendation Algorithms Implementation
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from scipy.sparse.linalg import svds

class BaseRecommender:
    """推荐器基类"""
    def __init__(self, data_path):
        self.data_path = data_path
        self.ratings_matrix = None
        self.user_ratings = defaultdict(dict)
        self.item_ratings = defaultdict(dict)
        self.item_popularity = {}
        # 确保所有推荐器都初始化此属性
        self.item_similarity = None
        
    def load_data(self):
        """加载MovieLens数据"""
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
    
    def train(self):
        """训练模型 - 子类实现"""
        raise NotImplementedError
    
    def recommend(self, user_id, top_k=10):
        """生成推荐 - 子类实现"""
        raise NotImplementedError


class ItemCFRecommender(BaseRecommender):
    """基于物品的协同过滤推荐器"""
    def __init__(self, data_path, similarity='cosine'):
        super().__init__(data_path)
        self.similarity_metric = similarity
        
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
    
    def compute_item_similarity(self):
        """计算物品相似度矩阵"""
        items = list(self.ratings_matrix.columns)
        n_items = len(items)
        
        similarity_matrix = np.zeros((n_items, n_items))
        
        print(f"  [ItemCF] 计算物品相似度矩阵 ({n_items} 个物品)...")
        
        for i in range(n_items):
            if (i + 1) % 100 == 0:
                print(f"    进度: {i+1}/{n_items}")
            for j in range(i, n_items):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    item1 = items[i]
                    item2 = items[j]
                    if self.similarity_metric == 'cosine':
                        sim = self.calculate_cosine_similarity(item1, item2)
                    elif self.similarity_metric == 'pearson':
                        sim = self.calculate_pearson_similarity(item1, item2)
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
        print(f"  [ItemCF] 模型训练完成 ({self.similarity_metric} 相似度)")
    
    def recommend(self, user_id, top_k=10):
        """为用户推荐物品"""
        if user_id not in self.user_ratings:
            return []
        
        user_rated_items = self.user_ratings[user_id]
        all_items = set(self.item_ratings.keys())
        candidate_items = all_items - set(user_rated_items.keys())
        
        predictions = []
        
        for item in candidate_items:
            if item not in self.item_similarity.index:
                continue
            
            weighted_sum = 0
            similarity_sum = 0
            
            similar_items = self.item_similarity.loc[item].sort_values(ascending=False)[1:21]
            
            for rated_item, rating in user_rated_items.items():
                if rated_item in similar_items.index:
                    sim = similar_items.get(rated_item, 0)
                    if sim > 0:
                        weighted_sum += sim * rating
                        similarity_sum += abs(sim)
            
            if similarity_sum > 0:
                predicted_score = weighted_sum / similarity_sum
                predictions.append((item, predicted_score))
        
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
            
            item_sims.sort(key=lambda x: x[1], reverse=True)
            similarity_info[rec_item] = item_sims[:5]
        
        return similarity_info


class UserCFRecommender(BaseRecommender):
    """基于用户的协同过滤推荐器"""
    def __init__(self, data_path, similarity='cosine'):
        super().__init__(data_path)
        self.similarity_metric = similarity
        self.user_similarity = None
        
    def calculate_user_similarity(self, user1, user2):
        """计算两个用户之间的相似度"""
        items1 = set(self.user_ratings.get(user1, {}).keys())
        items2 = set(self.user_ratings.get(user2, {}).keys())
        common_items = items1 & items2
        
        if len(common_items) < 2:
            return 0
        
        ratings1 = [self.user_ratings[user1][item] for item in common_items]
        ratings2 = [self.user_ratings[user2][item] for item in common_items]
        
        if self.similarity_metric == 'cosine':
            sim = 1 - cosine(ratings1, ratings2)
            return sim if not np.isnan(sim) else 0

        elif self.similarity_metric == 'pearson':
            try:
                corr, _ = pearsonr(ratings1, ratings2)
                return corr if not np.isnan(corr) else 0
            except:
                return 0
        
        return 0
    
    def compute_user_similarity(self):
        """计算用户相似度矩阵"""
        users = list(self.ratings_matrix.index)
        n_users = len(users)
        
        similarity_matrix = np.zeros((n_users, n_users))
        
        print(f"  [UserCF] 计算用户相似度矩阵 ({n_users} 个用户)...")
        
        for i in range(n_users):
            if (i + 1) % 100 == 0:
                print(f"    进度: {i+1}/{n_users}")
            for j in range(i, n_users):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    sim = self.calculate_user_similarity(users[i], users[j])
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim
        
        self.user_similarity = pd.DataFrame(
            similarity_matrix,
            index=users,
            columns=users
        )
        
    # --- 新增 ---
    def calculate_item_cosine_similarity(self, item1, item2):
        users1 = set(self.item_ratings[item1].keys())
        users2 = set(self.item_ratings[item2].keys())
        common_users = users1 & users2
        if len(common_users) < 2: return 0
        ratings1 = [self.item_ratings[item1][u] for u in common_users]
        ratings2 = [self.item_ratings[item2][u] for u in common_users]
        dot_product = sum(r1 * r2 for r1, r2 in zip(ratings1, ratings2))
        norm1 = np.sqrt(sum(r ** 2 for r in ratings1))
        norm2 = np.sqrt(sum(r ** 2 for r in ratings2))
        if norm1 == 0 or norm2 == 0: return 0
        return dot_product / (norm1 * norm2)

    def compute_item_similarity(self):
        """为UserCF计算物品相似度矩阵 (用于偏差分析)"""
        items = list(self.ratings_matrix.columns)
        n_items = len(items)
        similarity_matrix = np.zeros((n_items, n_items))
        print(f"  [UserCF] 为偏差分析计算物品相似度矩阵...")
        for i in range(n_items):
            if (i + 1) % 100 == 0:
                print(f"    进度: {i+1}/{n_items}")
            for j in range(i, n_items):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    sim = self.calculate_item_cosine_similarity(items[i], items[j])
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim
        self.item_similarity = pd.DataFrame(similarity_matrix, index=items, columns=items)

    def train(self):
        """训练模型"""
        self.load_data()
        self.compute_user_similarity()
        self.compute_item_similarity() # 关键修复
        print(f"  [UserCF] 模型训练完成 ({self.similarity_metric} 相似度)")
    
    def recommend(self, user_id, top_k=10, n_similar_users=50):
        """为用户推荐物品"""
        if user_id not in self.user_ratings:
            return []

        user_rated_items = self.user_ratings[user_id]
        
        if user_id not in self.user_similarity.index:
            print(f"  [UserCF] 新用户 {user_id}，动态计算相似度...")
            sims = {}
            for other_user_id in self.user_similarity.index:
                sim = self.calculate_user_similarity(user_id, other_user_id)
                if sim > 0:
                    sims[other_user_id] = sim
            
            similar_users = pd.Series(sims).sort_values(ascending=False)[:n_similar_users]
        else:
            similar_users = self.user_similarity.loc[user_id].sort_values(ascending=False)[1:n_similar_users+1]

        if similar_users.empty:
            return []

        predictions = defaultdict(float)
        similarity_sum = defaultdict(float)

        for similar_user, sim in similar_users.items():
            if sim <= 0: continue
            for item_id, rating in self.user_ratings.get(similar_user, {}).items():
                if item_id not in user_rated_items:
                    predictions[item_id] += sim * rating
                    similarity_sum[item_id] += sim

        final_predictions = []
        for item_id, total_score in predictions.items():
            if similarity_sum[item_id] > 0:
                predicted_score = total_score / similarity_sum[item_id]
                final_predictions.append((item_id, predicted_score))

        final_predictions.sort(key=lambda x: x[1], reverse=True)
        return final_predictions[:top_k]
    
    def get_item_similarity_for_user(self, user_id, recommended_items, n_similar_users=50):
        """获取推荐解释（基于相似用户）"""
        if user_id not in self.user_ratings:
            return {}
        
        if user_id not in self.user_similarity.index:
            sims = {}
            for other_user_id in self.user_similarity.index:
                sim = self.calculate_user_similarity(user_id, other_user_id)
                if sim > 0:
                    sims[other_user_id] = sim
            similar_users = pd.Series(sims).sort_values(ascending=False)[:n_similar_users]
        else:
            similar_users = self.user_similarity.loc[user_id].sort_values(ascending=False)[1:n_similar_users+1]

        similarity_info = {}
        
        for rec_item, _ in recommended_items:
            user_sims = []
            for similar_user_id, similarity in similar_users.items():
                if rec_item in self.user_ratings.get(similar_user_id, {}):
                    rating = self.user_ratings[similar_user_id][rec_item]
                    user_sims.append((similar_user_id, similarity, rating))
            
            user_sims.sort(key=lambda x: x[1], reverse=True)
            similarity_info[rec_item] = user_sims[:5]
        
        return similarity_info


class SVDRecommender(BaseRecommender):
    """基于矩阵分解（SVD）的推荐器"""
    def __init__(self, data_path, n_factors=50):
        super().__init__(data_path)
        self.n_factors = n_factors
        self.user_factors = None
        self.item_factors = None
        self.sigma = None
        self.global_mean = 0
        self.predicted_ratings = None
        
    # --- 新增 ---
    def compute_item_similarity_from_factors(self):
        """从SVD因子计算物品相似度矩阵"""
        print(f"  [SVD] 从因子计算物品相似度...")
        if self.item_factors is None:
            return

        norm = np.linalg.norm(self.item_factors, axis=1)
        non_zero_norm_mask = norm > 0
        item_factors_normalized = np.zeros_like(self.item_factors)
        item_factors_normalized[non_zero_norm_mask] = self.item_factors[non_zero_norm_mask] / norm[non_zero_norm_mask, np.newaxis]
        
        similarity_matrix = np.dot(item_factors_normalized, item_factors_normalized.T)
        
        items = list(self.ratings_matrix.columns)
        self.item_similarity = pd.DataFrame(
            similarity_matrix,
            index=items,
            columns=items
        )

    def train(self):
        """训练SVD模型"""
        self.load_data()
        
        print(f"  [SVD] 训练矩阵分解模型 (k={self.n_factors})...")
        
        ratings_matrix_values = self.ratings_matrix.values
        
        self.global_mean = np.mean(ratings_matrix_values[ratings_matrix_values > 0])
        
        ratings_matrix_centered = ratings_matrix_values - self.global_mean
        
        k = min(self.n_factors, min(ratings_matrix_centered.shape) - 1)
        U, sigma, Vt = svds(ratings_matrix_centered, k=k)
        
        self.sigma = np.diag(sigma)
        self.user_factors = U
        self.item_factors = Vt.T
        
        self.predicted_ratings = np.dot(np.dot(self.user_factors, self.sigma), Vt) + self.global_mean
        
        self.compute_item_similarity_from_factors() # 关键修复
        
        print(f"  [SVD] 模型训练完成 (因子数: {k})")
    
    def recommend(self, user_id, top_k=10):
        """为用户推荐物品"""
        rated_items = set(self.user_ratings.get(user_id, {}).keys())

        if user_id in self.ratings_matrix.index:
            user_idx = self.ratings_matrix.index.get_loc(user_id)
            user_predictions = self.predicted_ratings[user_idx]
        else:
            print(f"  [SVD] 新用户 {user_id}，执行 folding-in...")
            user_ratings_dict = self.user_ratings.get(user_id, {})
            if not user_ratings_dict:
                return []

            new_user_vector = pd.Series(index=self.ratings_matrix.columns).fillna(0)
            
            rated_item_ids = [item_id for item_id in user_ratings_dict.keys() if item_id in new_user_vector.index]
            for item_id in rated_item_ids:
                new_user_vector[item_id] = user_ratings_dict[item_id]
            
            new_user_centered_vector = new_user_vector.copy()
            new_user_centered_vector[rated_item_ids] -= self.global_mean
            
            V = self.item_factors
            Sigma_inv = np.linalg.inv(self.sigma)
            
            new_user_factors = new_user_centered_vector.values @ V @ Sigma_inv
            
            user_predictions = (new_user_factors @ self.sigma @ V.T) + self.global_mean
        
        user_predictions = np.array(user_predictions).flatten()

        predictions = []
        for item_idx, item_id in enumerate(self.ratings_matrix.columns):
            if item_id not in rated_items:
                predicted_score = user_predictions[item_idx]
                predicted_score = max(1, min(5, predicted_score))
                predictions.append((item_id, predicted_score))

        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_k]

    def get_item_similarity_for_user(self, user_id, recommended_items):
        """获取推荐解释（基于因子分析）"""
        if not self.user_ratings.get(user_id):
            return {}
        
        user_rated_items = self.user_ratings[user_id]
        similarity_info = {}
        
        for rec_item, _ in recommended_items:
            try:
                rec_item_idx = self.ratings_matrix.columns.get_loc(rec_item)
                rec_item_factors = self.item_factors[rec_item_idx]
                
                item_sims = []
                for rated_item, rating in user_rated_items.items():
                    try:
                        rated_item_idx = self.ratings_matrix.columns.get_loc(rated_item)
                        rated_item_factors = self.item_factors[rated_item_idx]
                        
                        sim = 1 - cosine(rec_item_factors, rated_item_factors)
                        
                        if sim > 0:
                            item_sims.append((rated_item, sim, rating))
                    except KeyError:
                        continue
                
                item_sims.sort(key=lambda x: x[1], reverse=True)
                similarity_info[rec_item] = item_sims[:5]
            except KeyError:
                continue
        
        return similarity_info

