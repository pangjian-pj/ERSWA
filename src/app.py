"""
可解释推荐系统 Web 应用
Explainable Recommender System Web Application
基于用户自定义偏好的推荐算法可解释性与偏差分析系统
"""

import streamlit as st
from streamlit_echarts import st_echarts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import recommend
from explain import explain_recommendations
from analysis import calculate_diversity, calculate_popularity_bias, calculate_novelty
import io
import os
import time

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 页面配置
st.set_page_config(
    page_title="可解释推荐系统",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# 加载电影数据
@st.cache_data
def load_movie_info():
    """加载电影信息"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        movies = pd.read_csv(
            os.path.join(script_dir,'data/u.item'),
            sep='|',
            encoding='latin-1',
            names=['movie_id', 'title', 'release_date', 'video_release_date', 
                   'imdb_url', 'unknown', 'Action', 'Adventure', 'Animation',
                   'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                   'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                   'Thriller', 'War', 'Western']
        )
        return movies
    except Exception as e:
        st.error(f"加载电影数据失败: {e}")
        return None

@st.cache_resource
def load_recommenders():
    """加载所有推荐模型（加载完成后清空提示）"""
    recommenders = {}
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dp = os.path.join(script_dir, 'data/u.data')

    # 创建占位符
    progress_text = st.empty()     # 文本提示
    progress_bar = st.empty()      # 进度条
    total_steps = 3

    # 创建进度条对象（必须写在 st.empty 里）
    pb = progress_bar.progress(0)

    # Step 1: ItemCF
    progress_text.write(f"🔄 (1/{total_steps}) 正在加载 ItemCF 模型...")
    itemcf = recommend.ItemCFRecommender(data_path=dp, similarity='cosine')
    itemcf.train()
    recommenders['ItemCF'] = itemcf
    pb.progress(1 / total_steps)

    # Step 2: UserCF
    progress_text.write(f"🔄 (2/{total_steps}) 正在加载 UserCF 模型...")
    usercf = recommend.UserCFRecommender(data_path=dp, similarity='cosine')
    usercf.train()
    recommenders['UserCF'] = usercf
    pb.progress(2 / total_steps)

    # Step 3: SVD
    progress_text.write(f"🔄 (3/{total_steps}) 正在加载 SVD 模型...")
    svd = recommend.SVDRecommender(data_path=dp, n_factors=50)
    svd.train()
    recommenders['SVD'] = svd
    pb.progress(1.0)

    # ✅ 全部加载完成后清空提示与进度条
    time.sleep(0.3)  # 给用户一点视觉缓冲，可去掉
    progress_text.empty()
    progress_bar.empty()

    return recommenders

def create_virtual_user(user_ratings, recommender):
    """
    根据用户输入创建虚拟用户评分向量
    
    参数:
        user_ratings: {movie_id: rating}
        recommender: 推荐器对象
    
    返回:
        virtual_user_id: 虚拟用户ID
    """
    # 为虚拟用户分配一个新ID (确保不会与现有用户冲突)
    max_user_id = max(recommender.ratings_matrix.index) if recommender.ratings_matrix is not None else 0
    virtual_user_id = max_user_id + 1
    
    # --- FIX: 简化处理 ---
    # 只需将虚拟用户的评分添加到 user_ratings 字典中
    # recommend 方法已被修改以处理新用户
    recommender.user_ratings[virtual_user_id] = user_ratings
    
    return virtual_user_id

def plot_recommendations_echarts(recommendations, movie_info):
    titles, scores = [], []
    for item_id, score in recommendations[:10]:
        movie_row = movie_info[movie_info['movie_id'] == item_id]
        title = movie_row['title'].values[0] if not movie_row.empty else f"Movie {item_id}"
        titles.append(title[:25])
        scores.append(round(score, 3))

    option = {
        "title": {"text": "Top 10 推荐电影", "left": "center", "textStyle": {"fontSize": 16}},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "grid": {"left": "22%", "right": "10%", "top": 60, "bottom": 40},
        "xAxis": {"type": "value", "name": "预测评分", "nameTextStyle": {"fontSize": 12}},
        "yAxis": {
            "type": "category",
            "data": titles[::-1],
            "axisLabel": {"fontSize": 12, "interval": 0, "overflow": "truncate"}
        },
        "series": [{
            "type": "bar",
            "data": scores[::-1],
            "barWidth": "55%",
            "label": {"show": True, "position": "right", "fontSize": 12},
            "itemStyle": {
                "color": {
                    "type": "linear",
                    "x": 0, "y": 0, "x2": 1, "y2": 0,
                    "colorStops": [
                        {"offset": 0, "color": "#667eea"},
                        {"offset": 1, "color": "#764ba2"}
                    ]
                }
            }
        }]
    }
    st_echarts(option, height="420px", key="rec_chart")


def plot_explanation_echarts(explanations, top_item, movie_info):
    if top_item not in explanations or not explanations[top_item]['contribution']:
        st.warning("暂无详细解释信息")
        return

    contrib_items, contrib_values = [], []
    for c in explanations[top_item]['contribution']:
        movie_row = movie_info[movie_info['movie_id'] == c['item']]
        title = movie_row['title'].values[0] if not movie_row.empty else f"Item {c['item']}"
        contrib_items.append(title[:25])
        contrib_values.append(round(c['contribution'], 3))

    option = {
        "title": {"text": "Top 1 推荐贡献度分析", "left": "center", "textStyle": {"fontSize": 16}},
        "tooltip": {"trigger": "axis"},
        "grid": {"left": "22%", "right": "10%", "top": 60, "bottom": 40},
        "xAxis": {"type": "value", "name": "贡献度"},
        "yAxis": {"type": "category", "data": contrib_items[::-1]},
        "series": [{
            "type": "bar",
            "data": contrib_values[::-1],
            "barWidth": "55%",
            "label": {"show": True, "position": "right"},
            "itemStyle": {
                "color": {
                    "type": "linear",
                    "x": 0, "y": 0, "x2": 1, "y2": 0,
                    "colorStops": [
                        {"offset": 0, "color": "#ff9a9e"},
                        {"offset": 1, "color": "#fad0c4"}
                    ]
                }
            }
        }]
    }
    st_echarts(option, height="420px", key="exp_chart")


def plot_bias_metrics_echarts(diversity, pop_bias, novelty):
    option = {
        "title": {"text": "推荐系统偏差指标分析", "left": "center", "textStyle": {"fontSize": 16}},
        "tooltip": {},
        "radar": {
            "indicator": [
                {"name": "多样性", "max": 1},
                {"name": "流行度平衡", "max": 2},
                {"name": "新颖性", "max": 10}
            ],
            "radius": "60%",
            "center": ["50%", "55%"],
            "splitArea": {"areaStyle": {"color": ["#f9f9f9", "#fff"]}},
            "axisName": {"color": "#333", "fontSize": 12}
        },
        "series": [{
            "type": "radar",
            "data": [{
                "value": [diversity, pop_bias, novelty],
                "name": "指标得分"
            }],
            "lineStyle": {"color": "#667eea", "width": 2},
            "areaStyle": {"opacity": 0.3, "color": "#667eea"},
            "symbol": "circle",
            "symbolSize": 8,
            "itemStyle": {"color": "#667eea"}
        }]
    }
    st_echarts(option, height="400px", key="bias_chart")



def plot_recommendations_web(recommendations, movie_info):
    """生成推荐结果条形图"""
    plt.rcParams['font.sans-serif'] = ['STHeiti']  # macOS 中文字体
    plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
    fig, ax = plt.subplots(figsize=(10, 6))
    
    items = []
    scores = []
    for item_id, score in recommendations[:10]:
        title = f"Movie {item_id}"
        if movie_info is not None:
             movie_row = movie_info[movie_info['movie_id'] == item_id]
             if not movie_row.empty:
                 title = movie_row['title'].values[0]
        
        items.append(f"{title[:30]}..." if len(title) > 30 else title)
        scores.append(score)
    
    ax.barh(items, scores, color='steelblue')
    ax.set_xlabel('预测评分 (Predicted Score)', fontsize=12)
    ax.set_title('Top 10 推荐电影', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    
    return fig

def plot_explanation_web(explanations, top_item, movie_info):
    """生成推荐解释图"""
    plt.rcParams['font.sans-serif'] = ['STHeiti']  # macOS 中文字体
    plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if top_item in explanations and explanations[top_item]['contribution']:
        contrib_items = []
        contrib_values = []
        
        for c in explanations[top_item]['contribution']:
            title = f"Item {c['item']}"
            if movie_info is not None:
                movie_row = movie_info[movie_info['movie_id'] == c['item']]
                if not movie_row.empty:
                    title = movie_row['title'].values[0]

            contrib_items.append(f"{title[:25]}..." if len(title) > 25 else title)
            contrib_values.append(c['contribution'])
        
        ax.barh(contrib_items, contrib_values, color='coral')
        ax.set_xlabel('贡献度 (Contribution)', fontsize=12)
        ax.set_title(f'Top 1 推荐原因解释', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, '无详细解释信息', ha='center', va='center', fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    return fig

def plot_bias_metrics_web(diversity, pop_bias, novelty):
    """生成偏差指标图"""
    plt.rcParams['font.sans-serif'] = ['STHeiti']  # macOS 中文字体
    plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    metrics = ['多样性\nDiversity', '流行度偏差\nPopularity Bias', '新颖性\nNovelty']
    values = [diversity, pop_bias, novelty]
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    
    for idx, (metric, value, color) in enumerate(zip(metrics, values, colors)):
        axes[idx].bar([metric], [value], color=color, alpha=0.7, edgecolor='black', linewidth=2)
        axes[idx].set_ylabel('分数', fontsize=11)
        axes[idx].set_title(metric.split('\n')[0], fontsize=12, fontweight='bold')
        axes[idx].text(0, value, f'{value:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        axes[idx].set_ylim(0, max(value * 1.3, 1)) # 动态调整Y轴
    
    plt.tight_layout()
    return fig

# 主应用
def main():
    # 标题
    st.markdown('<h1 class="main-header">🎬 可解释推荐系统</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">基于用户自定义偏好的推荐算法可解释性与偏差分析系统</p>', unsafe_allow_html=True)
    
    # 加载数据
    movie_info = load_movie_info()
    recommenders = load_recommenders()
    
    # 侧边栏 - 用户输入
    st.sidebar.header("📝 用户偏好设置")
    st.sidebar.markdown("---")
    
    # 算法选择
    st.sidebar.subheader("🤖 选择推荐算法")
    algorithm_choice = st.sidebar.selectbox(
        "请选择推荐算法：",
        ["ItemCF (基于物品协同过滤)", "UserCF (基于用户协同过滤)", "SVD (矩阵分解)"],
        help="不同算法有不同的推荐策略"
    )
    
    # 算法说明
    algorithm_descriptions = {
        "ItemCF (基于物品协同过滤)": "📊 基于物品相似度推荐。找到与您喜欢的电影相似的其他电影。",
        "UserCF (基于用户协同过滤)": "👥 基于用户相似度推荐。找到和您口味相似的用户喜欢的电影。",
        "SVD (矩阵分解)": "🧮 基于隐含因子推荐。通过深层次特征分析发现您的潜在偏好。"
    }
    
    st.sidebar.info(algorithm_descriptions[algorithm_choice])
    st.sidebar.markdown("---")
    
    # 获取选中的推荐器
    algorithm_map = {
        "ItemCF (基于物品协同过滤)": "ItemCF",
        "UserCF (基于用户协同过滤)": "UserCF",
        "SVD (矩阵分解)": "SVD"
    }
    selected_algorithm = algorithm_map[algorithm_choice]
    recommender = recommenders[selected_algorithm]
    
    # 输入方式选择
    input_method = st.sidebar.radio(
        "选择输入方式：",
        ["手动输入电影ID和评分", "按类型选择电影"]
    )
    
    user_ratings = {}
    
    if input_method == "手动输入电影ID和评分":
        st.sidebar.subheader("🎯 输入您的电影评分")
        st.sidebar.info("请输入电影ID（1-1682）和您的评分（1-5分）")
        
        num_ratings = st.sidebar.slider("您想评分几部电影？", 3, 20, 5)
        
        for i in range(num_ratings):
            col1, col2 = st.sidebar.columns(2)
            movie_id = col1.number_input(f"电影 {i+1} ID", min_value=1, max_value=1682, 
                                          value=i*10+1, key=f"movie_{i}")
            rating = col2.slider(f"评分", 1, 5, 4, key=f"rating_{i}")
            
            if movie_id and rating:
                user_ratings[movie_id] = rating
                
                # 显示电影标题
                if movie_info is not None:
                    movie_row = movie_info[movie_info['movie_id'] == movie_id]
                    if not movie_row.empty:
                        title = movie_row['title'].values[0]
                        st.sidebar.caption(f"   → {title}")
    
    else:
        st.sidebar.subheader("🎭 按电影类型选择")
        
        if movie_info is not None:
            genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 
                     'Crime', 'Documentary', 'Drama', 'Fantasy', 'Horror', 
                     'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
            
            selected_genre = st.sidebar.selectbox("选择您喜欢的类型：", genres)
            
            genre_movies = movie_info[movie_info[selected_genre] == 1]
            
            if not genre_movies.empty:
                st.sidebar.info(f"找到 {len(genre_movies)} 部 {selected_genre} 电影")
                
                sample_movies = genre_movies.sample(min(10, len(genre_movies)), random_state=1)
                
                for _, movie in sample_movies.iterrows():
                    movie_id = movie['movie_id']
                    title = movie['title']
                    
                    rating = st.sidebar.slider(
                        title[:40],
                        1, 5, 3,
                        key=f"genre_movie_{movie_id}"
                    )
                    user_ratings[movie_id] = rating
    
    st.sidebar.markdown("---")
    
    generate_button = st.sidebar.button("🚀 生成推荐", type="primary", use_container_width=True)
    
    if generate_button:
        if len(user_ratings) < 3:
            st.error("⚠️ 请至少评分 3 部电影！")
        else:
            st.success(f"✅ 已收到您对 {len(user_ratings)} 部电影的评分！")
            
            with st.expander("📊 查看您的评分"):
                rating_list = []
                for mid, rating in user_ratings.items():
                    title = f"Movie {mid}"
                    if movie_info is not None:
                        movie_row = movie_info[movie_info['movie_id'] == mid]
                        if not movie_row.empty:
                            title = movie_row['title'].values[0]
                    rating_list.append({'电影ID': mid, '电影标题': title, '您的评分': rating})
                
                rating_df = pd.DataFrame(rating_list)
                st.dataframe(rating_df, use_container_width=True)
            
            virtual_user_id = create_virtual_user(user_ratings, recommender)
            
            with st.spinner('🔮 正在为您生成个性化推荐...'):
                recommendations = recommender.recommend(virtual_user_id, top_k=10)
            
            if not recommendations:
                st.error("❌ 无法生成推荐，请尝试调整您的评分或选择其他算法。")
            else:
                tab1, tab2, tab3, tab4 = st.tabs(["📋 推荐结果", "💡 可解释性分析", "⚖️ 偏差分析", "📥 下载报告"])
                
                with tab1:
                    st.header("🎬 为您推荐的 Top 10 电影")
                    
                    rec_data = []
                    for rank, (item_id, score) in enumerate(recommendations, 1):
                        title = "Unknown"
                        if movie_info is not None:
                            movie_row = movie_info[movie_info['movie_id'] == item_id]
                            if not movie_row.empty:
                                title = movie_row['title'].values[0]
                        
                        rec_data.append({
                            '排名': rank,
                            '电影ID': item_id,
                            '电影标题': title,
                            '预测评分': f"{score:.3f}"
                        })
                    
                    rec_df = pd.DataFrame(rec_data)
                    st.dataframe(rec_df, use_container_width=True)
                    
                    st.subheader("📊 推荐评分可视化")
                    fig_rec = plot_recommendations_echarts(recommendations, movie_info)
                    # st.pyplot(fig_rec, use_container_width=True)
                
                with tab2:
                    st.header("💡 推荐解释 - 为什么推荐这些电影？")
                    
                    with st.spinner('🧠 正在生成解释...'):
                        explanations = explain_recommendations(virtual_user_id, recommendations, recommender)
                    
                    if selected_algorithm == "UserCF":
                        st.info("🎯 **UserCF 算法**: 基于与您口味相似的其他用户进行推荐。")
                    elif selected_algorithm == "SVD":
                        st.info("🎯 **SVD 算法**: 基于电影和用户的深层“隐含特征”进行推荐。")
                    else:
                        st.info("🎯 **ItemCF 算法**: 基于您喜欢的电影，推荐与之内容最相似的其他电影。")
                    
                    if recommendations and recommendations[0][0] in explanations:
                        top_item = recommendations[0][0]
                        exp = explanations[top_item]
                        
                        st.subheader(f"🎯 Top 1 推荐的详细解释")
                        
                        title = f"Movie {top_item}"
                        if movie_info is not None:
                            movie_row = movie_info[movie_info['movie_id'] == top_item]
                            if not movie_row.empty: title = movie_row['title'].values[0]
                        
                        st.info(f"**推荐电影**: {title} (ID: {top_item})")
                        st.write(f"**预测评分**: {exp['predicted_score']:.3f}")
                        
                        if exp['contribution']:
                            if selected_algorithm == "UserCF":
                                st.write("**推荐原因**: 因为以下与您品味相似的用户也喜欢这部电影：")
                                for c in exp['contribution']:
                                    st.write(f"- 👤 **相似用户 {c['item']}** (相似度: {c['similarity']:.3f}, 该用户评分: {c['rating']:.1f}, 贡献: {c['contribution']:.3f})")
                            else:
                                st.write("**推荐原因**: 因为您喜欢以下与推荐电影相似的电影：")
                                for c in exp['contribution']:
                                    contrib_title = f"Movie {c['item']}"
                                    if movie_info is not None:
                                        movie_row = movie_info[movie_info['movie_id'] == c['item']]
                                        if not movie_row.empty: contrib_title = movie_row['title'].values[0]
                                    
                                    st.write(f"- 📽️ **{contrib_title}** (您的评分: {c['rating']:.1f}, 相似度: {c['similarity']:.3f}, 贡献: {c['contribution']:.3f})")
                        
                        st.subheader("📊 贡献度可视化")
                        fig_exp = plot_explanation_echarts(explanations, top_item, movie_info)
                        # st.pyplot(fig_exp, use_container_width=True)
                    else:
                        st.warning("暂无详细解释信息")
                
                with tab3:
                    st.header("⚖️ 推荐偏差分析")
                    
                    diversity = calculate_diversity(recommendations, recommender)
                    pop_bias = calculate_popularity_bias(recommendations, recommender)
                    novelty = calculate_novelty(recommendations, recommender)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f'<div class="metric-card"><h3>多样性 Diversity</h3><h1>{diversity:.4f}</h1><p>越高越好 - 结果越多样化</p></div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown(f'<div class="metric-card"><h3>流行度偏差</h3><h1>{pop_bias:.4f}</h1><p>接近1为理想 - 是否偏向热门</p></div>', unsafe_allow_html=True)
                    with col3:
                        st.markdown(f'<div class="metric-card"><h3>新颖性 Novelty</h3><h1>{novelty:.4f}</h1><p>越高越好 - 推荐冷门的能力</p></div>', unsafe_allow_html=True)
                    
                    with st.expander("📖 指标说明"):
                        st.markdown("""
                        - **多样性 (Diversity)**: 衡量推荐结果之间的差异程度。值越高表示推荐的电影越不相似，能为用户提供更丰富的选择。
                        - **流行度偏差 (Popularity Bias)**: 衡量推荐系统是否过度倾向于推荐热门电影。值大于1表示偏向热门，小于1表示偏向冷门，接近1为平衡。
                        - **新颖性 (Novelty)**: 衡量推荐系统推荐冷门、小众电影的能力。值越高表示推荐的电影越新颖独特。
                        """)
                    
                    st.subheader("📊 偏差指标可视化")
                    fig_bias = plot_bias_metrics_echarts(diversity, pop_bias, novelty)
                    # st.pyplot(fig_bias, use_container_width=True)

                with tab4:
                    st.header("📄 完整分析报告")
                    
                    report_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.markdown(f"**📊 使用算法**: {algorithm_choice}\n\n**📅 生成时间**: {report_time}")
                    st.markdown("---")
                    
                    st.subheader("1️⃣ 您的评分")
                    st.dataframe(pd.DataFrame(rating_list), use_container_width=True)
                    
                    st.subheader("2️⃣ 推荐结果 (Top 10)")
                    st.dataframe(rec_df, use_container_width=True)
                    
                    st.subheader("3️⃣ 偏差分析")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("多样性 (Diversity)", f"{diversity:.4f}")
                    col2.metric("流行度偏差", f"{pop_bias:.4f}")
                    col3.metric("新颖性 (Novelty)", f"{novelty:.4f}")

                    st.subheader("4️⃣ 综合结论")
                    conclusions = []
                    if diversity > 0.7: conclusions.append("✅ **多样性优秀**: 推荐结果涵盖多种不同类型的电影。")
                    else: conclusions.append("⚠️ **多样性偏低**: 推荐结果相似度较高，可尝试评分更多类型电影。")
                    
                    if 0.8 < pop_bias < 1.2: conclusions.append("✅ **流行度平衡**: 能在热门和冷门电影间取得良好平衡。")
                    elif pop_bias > 1.2: conclusions.append("📈 **偏向热门**: 推荐结果倾向于大众化热门电影。")
                    else: conclusions.append("🔍 **偏向冷门**: 推荐结果倾向于小众电影，助您发现惊喜。")
                    
                    if novelty > 9: conclusions.append("✨ **新颖性优秀**: 能帮助您发现许多独特、小众的电影。")
                    else: conclusions.append("📺 **新颖性一般**: 推荐偏向常见电影，探索性较弱。")
                    
                    for conclusion in conclusions:
                        st.markdown(f"- {conclusion}")
                    
                    # 生成文本报告供下载
                    report_text = f"推荐系统分析报告\n==================\n\n"
                    report_text += f"算法: {algorithm_choice}\n时间: {report_time}\n\n"
                    report_text += "1. 您的评分:\n" + pd.DataFrame(rating_list).to_string(index=False) + "\n\n"
                    report_text += "2. 推荐结果:\n" + rec_df.to_string(index=False) + "\n\n"
                    report_text += "3. 偏差分析:\n" + f"   - 多样性: {diversity:.4f}\n   - 流行度偏差: {pop_bias:.4f}\n   - 新颖性: {novelty:.4f}\n\n"
                    report_text += "4. 结论:\n" + "\n".join(conclusions)

                    st.download_button(
                        label="📥 下载完整报告 (TXT)",
                        data=report_text.encode('utf-8'),
                        file_name=f"recommendation_report_{pd.Timestamp.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
    st.markdown("""
<hr style="border: none; border-top: 2px solid #bbb; margin-top: 40px;">
<div style="text-align: center; font-size: 15px; color: #555;">
    <p><strong>Authors:</strong> Jian Pang, Yongliang Ye, and Junjie Chen</p>
    <p><strong>Supervised by:</strong> Associate Professor Shouqiang Liu</p>
    <p><strong>Affiliation:</strong> School of Artificial Intelligence, South China Normal University, Foshan, Guangdong, China</p>
</div>
""", unsafe_allow_html=True)



if __name__ == "__main__":
    main()
