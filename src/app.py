"""
可解释推荐系统 Web 应用
Explainable Recommender System Web Application
基于用户自定义偏好的推荐算法可解释性与偏差分析系统
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from recommend import ItemCFRecommender
from explain import explain_recommendations
from analysis import calculate_diversity, calculate_popularity_bias, calculate_novelty
import io
import os

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
    except:
        return None

# 加载推荐模型
@st.cache_resource
def load_recommender():
    """加载训练好的推荐模型"""
    with st.spinner('正在加载推荐模型... 这可能需要几分钟时间'):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        recommender = ItemCFRecommender(data_path=os.path.join(script_dir,'data/u.data'), similarity='cosine')
        recommender.train()
    return recommender

def create_virtual_user(user_ratings, recommender):
    """
    根据用户输入创建虚拟用户评分向量
    
    参数:
        user_ratings: {movie_id: rating}
        recommender: 推荐器对象
    
    返回:
        virtual_user_id: 虚拟用户ID
    """
    # 为虚拟用户分配一个新ID
    max_user_id = max(recommender.user_ratings.keys()) if recommender.user_ratings else 0
    virtual_user_id = max_user_id + 1
    
    # 添加虚拟用户评分
    recommender.user_ratings[virtual_user_id] = user_ratings
    
    return virtual_user_id

def plot_recommendations_web(recommendations, movie_info):
    """生成推荐结果条形图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    items = []
    scores = []
    for item_id, score in recommendations[:10]:
        if movie_info is not None and item_id in movie_info['movie_id'].values:
            title = movie_info[movie_info['movie_id'] == item_id]['title'].values[0]
            items.append(f"{title[:30]}...")
        else:
            items.append(f"Movie {item_id}")
        scores.append(score)
    
    ax.barh(items, scores, color='steelblue')
    ax.set_xlabel('预测评分 (Predicted Score)', fontsize=12)
    ax.set_title('Top 10 推荐电影', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    
    return fig

def plot_explanation_web(explanations, top_item, movie_info):
    """生成推荐解释图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if top_item in explanations and explanations[top_item]['contribution']:
        contrib_items = []
        contrib_values = []
        
        for c in explanations[top_item]['contribution']:
            if movie_info is not None and c['item'] in movie_info['movie_id'].values:
                title = movie_info[movie_info['movie_id'] == c['item']]['title'].values[0]
                contrib_items.append(f"{title[:25]}...")
            else:
                contrib_items.append(f"Movie {c['item']}")
            contrib_values.append(c['contribution'])
        
        ax.barh(contrib_items, contrib_values, color='coral')
        ax.set_xlabel('贡献度 (Contribution)', fontsize=12)
        ax.set_title(f'推荐原因解释', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, '无详细解释信息', ha='center', va='center', fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    return fig

def plot_bias_metrics_web(diversity, pop_bias, novelty):
    """生成偏差指标图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    metrics = ['多样性\nDiversity', '流行度偏差\nPopularity Bias', '新颖性\nNovelty']
    values = [diversity, pop_bias, novelty]
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    
    for idx, (metric, value, color) in enumerate(zip(metrics, values, colors)):
        axes[idx].bar([metric], [value], color=color, alpha=0.7, edgecolor='black', linewidth=2)
        axes[idx].set_ylabel('分数', fontsize=11)
        axes[idx].set_title(metric, fontsize=12, fontweight='bold')
        axes[idx].text(0, value, f'{value:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        axes[idx].set_ylim(0, max(value * 1.3, 1))
    
    plt.tight_layout()
    return fig

# 主应用
def main():
    # 标题
    st.markdown('<h1 class="main-header">🎬 可解释推荐系统</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">基于用户自定义偏好的推荐算法可解释性与偏差分析系统</p>', unsafe_allow_html=True)
    
    # 加载数据
    movie_info = load_movie_info()
    recommender = load_recommender()
    
    # 侧边栏 - 用户输入
    st.sidebar.header("📝 用户偏好设置")
    st.sidebar.markdown("---")
    
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
            with col1:
                movie_id = st.number_input(f"电影 {i+1} ID", min_value=1, max_value=1682, 
                                          value=1, key=f"movie_{i}")
            with col2:
                rating = st.slider(f"评分", 1, 5, 4, key=f"rating_{i}")
            
            if movie_id and rating:
                user_ratings[movie_id] = rating
                
                # 显示电影标题
                if movie_info is not None and movie_id in movie_info['movie_id'].values:
                    title = movie_info[movie_info['movie_id'] == movie_id]['title'].values[0]
                    st.sidebar.caption(f"   → {title}")
    
    else:
        st.sidebar.subheader("🎭 按电影类型选择")
        
        if movie_info is not None:
            genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 
                     'Crime', 'Documentary', 'Drama', 'Fantasy', 'Horror', 
                     'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
            
            selected_genre = st.sidebar.selectbox("选择您喜欢的类型：", genres)
            
            # 筛选该类型的电影
            genre_movies = movie_info[movie_info[selected_genre] == 1]
            
            if len(genre_movies) > 0:
                st.sidebar.info(f"找到 {len(genre_movies)} 部 {selected_genre} 电影")
                
                # 随机选择几部电影让用户评分
                sample_movies = genre_movies.sample(min(10, len(genre_movies)))
                
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
    
    # 推荐按钮
    generate_button = st.sidebar.button("🚀 生成推荐", type="primary")
    
    # 主内容区域
    if generate_button:
        if len(user_ratings) < 3:
            st.error("⚠️ 请至少评分 3 部电影！")
        else:
            st.success(f"✅ 已收到您对 {len(user_ratings)} 部电影的评分！")
            
            # 显示用户输入
            with st.expander("📊 查看您的评分", expanded=False):
                rating_df = pd.DataFrame([
                    {
                        '电影ID': mid,
                        '电影标题': movie_info[movie_info['movie_id'] == mid]['title'].values[0] 
                                   if movie_info is not None and mid in movie_info['movie_id'].values 
                                   else f"Movie {mid}",
                        '您的评分': rating
                    }
                    for mid, rating in user_ratings.items()
                ])
                st.dataframe(rating_df, use_container_width=True)
            
            # 创建虚拟用户
            virtual_user_id = create_virtual_user(user_ratings, recommender)
            
            # 生成推荐
            with st.spinner('🔮 正在为您生成个性化推荐...'):
                recommendations = recommender.recommend(virtual_user_id, top_k=10)
            
            if not recommendations:
                st.error("❌ 无法生成推荐，请尝试调整您的评分。")
            else:
                # Tab布局
                tab1, tab2, tab3, tab4 = st.tabs(["📋 推荐结果", "💡 可解释性分析", "⚖️ 偏差分析", "📥 下载报告"])
                
                with tab1:
                    st.header("🎬 为您推荐的 Top 10 电影")
                    
                    # 推荐列表表格
                    rec_data = []
                    for rank, (item_id, score) in enumerate(recommendations, 1):
                        title = "Unknown"
                        if movie_info is not None and item_id in movie_info['movie_id'].values:
                            title = movie_info[movie_info['movie_id'] == item_id]['title'].values[0]
                        
                        rec_data.append({
                            '排名': rank,
                            '电影ID': item_id,
                            '电影标题': title,
                            '预测评分': f"{score:.3f}"
                        })
                    
                    rec_df = pd.DataFrame(rec_data)
                    st.dataframe(rec_df, use_container_width=True, hide_index=True)
                    
                    # 可视化
                    st.subheader("📊 推荐评分可视化")
                    fig_rec = plot_recommendations_web(recommendations, movie_info)
                    st.pyplot(fig_rec)
                
                with tab2:
                    st.header("💡 推荐解释 - 为什么推荐这些电影？")
                    
                    # 生成解释
                    explanations = explain_recommendations(virtual_user_id, recommendations, recommender)
                    
                    if recommendations and recommendations[0][0] in explanations:
                        top_item = recommendations[0][0]
                        exp = explanations[top_item]
                        
                        st.subheader(f"🎯 Top 1 推荐的详细解释")
                        
                        title = "Unknown Movie"
                        if movie_info is not None and top_item in movie_info['movie_id'].values:
                            title = movie_info[movie_info['movie_id'] == top_item]['title'].values[0]
                        
                        st.info(f"**推荐电影**: {title} (ID: {top_item})")
                        st.write(f"**预测评分**: {exp['predicted_score']:.3f}")
                        
                        if exp['contribution']:
                            st.write("**推荐原因**: 因为您喜欢以下相似电影：")
                            
                            for c in exp['contribution']:
                                contrib_title = "Unknown"
                                if movie_info is not None and c['item'] in movie_info['movie_id'].values:
                                    contrib_title = movie_info[movie_info['movie_id'] == c['item']]['title'].values[0]
                                
                                st.write(f"- 📽️ **{contrib_title}** (您的评分: {c['rating']:.1f}, "
                                        f"相似度: {c['similarity']:.3f}, 贡献: {c['contribution']:.3f})")
                        
                        # 可视化解释
                        st.subheader("📊 贡献度可视化")
                        fig_exp = plot_explanation_web(explanations, top_item, movie_info)
                        st.pyplot(fig_exp)
                    else:
                        st.warning("暂无详细解释信息")
                
                with tab3:
                    st.header("⚖️ 推荐偏差分析")
                    
                    # 计算偏差指标
                    diversity = calculate_diversity(recommendations, recommender)
                    pop_bias = calculate_popularity_bias(recommendations, recommender)
                    novelty = calculate_novelty(recommendations, recommender)
                    
                    # 显示指标卡片
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>多样性 Diversity</h3>
                            <h1>{diversity:.4f}</h1>
                            <p>越高越好 - 推荐结果越多样化</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>流行度偏差</h3>
                            <h1>{pop_bias:.4f}</h1>
                            <p>接近1为理想 - 是否偏向热门电影</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>新颖性 Novelty</h3>
                            <h1>{novelty:.4f}</h1>
                            <p>越高越好 - 推荐冷门电影的能力</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # 指标解释
                    with st.expander("📖 指标说明", expanded=False):
                        st.markdown("""
                        - **多样性 (Diversity)**: 衡量推荐结果之间的差异程度。值越高表示推荐的电影越不相似，能为用户提供更丰富的选择。
                        - **流行度偏差 (Popularity Bias)**: 衡量推荐系统是否过度倾向于推荐热门电影。值大于1表示偏向热门，小于1表示偏向冷门。
                        - **新颖性 (Novelty)**: 衡量推荐系统推荐冷门、小众电影的能力。值越高表示推荐的电影越新颖独特。
                        """)
                    
                    # 可视化偏差指标
                    st.subheader("📊 偏差指标可视化")
                    fig_bias = plot_bias_metrics_web(diversity, pop_bias, novelty)
                    st.pyplot(fig_bias)
                
                with tab4:
                    st.header("📄 完整分析报告")
                    
                    # 报告生成时间
                    report_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # 第一部分：用户输入评分
                    st.subheader("1️⃣ 用户输入评分")
                    st.markdown("---")
                    
                    input_data = []
                    for mid, rating in user_ratings.items():
                        title = "Unknown"
                        if movie_info is not None and mid in movie_info['movie_id'].values:
                            title = movie_info[movie_info['movie_id'] == mid]['title'].values[0]
                        input_data.append({
                            '电影ID': mid,
                            '电影标题': title,
                            '您的评分': f"⭐ {rating}"
                        })
                    
                    input_df = pd.DataFrame(input_data)
                    st.dataframe(input_df, use_container_width=True, hide_index=True)
                    st.caption(f"共评分 {len(user_ratings)} 部电影")
                    
                    # 第二部分：推荐结果
                    st.subheader("2️⃣ 推荐结果 (Top 10)")
                    st.markdown("---")
                    
                    rec_report_data = []
                    for rank, (item_id, score) in enumerate(recommendations, 1):
                        title = "Unknown"
                        if movie_info is not None and item_id in movie_info['movie_id'].values:
                            title = movie_info[movie_info['movie_id'] == item_id]['title'].values[0]
                        
                        # 生成星级显示
                        stars = "⭐" * int(score)
                        
                        rec_report_data.append({
                            '排名': f"#{rank}",
                            '电影ID': item_id,
                            '电影标题': title,
                            '预测评分': f"{score:.3f}",
                            '星级': stars
                        })
                    
                    rec_report_df = pd.DataFrame(rec_report_data)
                    st.dataframe(rec_report_df, use_container_width=True, hide_index=True)
                    
                    # 第三部分：可解释性分析
                    st.subheader("3️⃣ 可解释性分析")
                    st.markdown("---")
                    
                    explanations = explain_recommendations(virtual_user_id, recommendations, recommender)
                    
                    if recommendations and recommendations[0][0] in explanations:
                        top_item = recommendations[0][0]
                        exp = explanations[top_item]
                        
                        top_title = "Unknown Movie"
                        if movie_info is not None and top_item in movie_info['movie_id'].values:
                            top_title = movie_info[movie_info['movie_id'] == top_item]['title'].values[0]
                        
                        st.markdown(f"**Top 1 推荐电影**: 🎬 {top_title}")
                        st.markdown(f"**预测评分**: {exp['predicted_score']:.3f}")
                        
                        if exp['contribution']:
                            st.markdown("**推荐原因分析**:")
                            
                            contrib_data = []
                            for c in exp['contribution']:
                                contrib_title = "Unknown"
                                if movie_info is not None and c['item'] in movie_info['movie_id'].values:
                                    contrib_title = movie_info[movie_info['movie_id'] == c['item']]['title'].values[0]
                                
                                contrib_data.append({
                                    '相似电影': contrib_title,
                                    '您的评分': f"{c['rating']:.1f}",
                                    '相似度': f"{c['similarity']:.3f}",
                                    '贡献度': f"{c['contribution']:.3f}"
                                })
                            
                            contrib_df = pd.DataFrame(contrib_data)
                            st.dataframe(contrib_df, use_container_width=True, hide_index=True)
                            
                            st.info("💡 解释：系统推荐此电影是因为它与您高评分的这些电影非常相似")
                    else:
                        st.warning("暂无详细解释信息")
                    
                    # 第四部分：偏差分析
                    st.subheader("4️⃣ 偏差分析")
                    st.markdown("---")
                    
                    # 创建三列显示指标
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            label="多样性 (Diversity)",
                            value=f"{diversity:.4f}",
                            delta="良好" if diversity > 0.6 else "一般",
                            delta_color="normal" if diversity > 0.6 else "inverse"
                        )
                    
                    with col2:
                        st.metric(
                            label="流行度偏差",
                            value=f"{pop_bias:.4f}",
                            delta="平衡" if 0.8 < pop_bias < 1.2 else "偏向热门" if pop_bias > 1.2 else "偏向冷门",
                            delta_color="normal" if 0.8 < pop_bias < 1.2 else "inverse"
                        )
                    
                    with col3:
                        st.metric(
                            label="新颖性 (Novelty)",
                            value=f"{novelty:.4f}",
                            delta="优秀" if novelty > 8 else "一般",
                            delta_color="normal" if novelty > 8 else "inverse"
                        )
                    
                    # 指标详细说明
                    st.markdown("**指标详细说明**:")
                    
                    metric_details = {
                        "多样性": {
                            "数值": f"{diversity:.4f}",
                            "评价": "优秀" if diversity > 0.7 else "良好" if diversity > 0.6 else "一般",
                            "说明": "推荐结果的多样化程度。值越高表示推荐的电影类型越丰富，不会过于集中在某一类型。"
                        },
                        "流行度偏差": {
                            "数值": f"{pop_bias:.4f}",
                            "评价": "平衡" if 0.8 < pop_bias < 1.2 else "偏向热门" if pop_bias > 1.2 else "偏向冷门",
                            "说明": "衡量推荐系统是否过度推荐热门电影。接近1.0表示推荐平衡，大于1.2表示倾向热门电影，小于0.8表示倾向冷门电影。"
                        },
                        "新颖性": {
                            "数值": f"{novelty:.4f}",
                            "评价": "优秀" if novelty > 9 else "良好" if novelty > 8 else "一般",
                            "说明": "推荐冷门、独特电影的能力。值越高表示推荐的电影越新颖，有助于用户发现小众佳作。"
                        }
                    }
                    
                    for metric, details in metric_details.items():
                        with st.expander(f"📊 {metric}: {details['数值']} - {details['评价']}"):
                            st.write(details['说明'])
                    
                    # 第五部分：综合结论
                    st.subheader("5️⃣ 综合分析结论")
                    st.markdown("---")
                    
                    # 生成结论
                    conclusions = []
                    
                    # 多样性结论
                    if diversity > 0.7:
                        conclusions.append("✅ **多样性优秀**: 推荐结果涵盖多种不同类型的电影，能为您提供丰富的选择。")
                    elif diversity > 0.6:
                        conclusions.append("✅ **多样性良好**: 推荐结果有一定的多样性，但仍有提升空间。")
                    else:
                        conclusions.append("⚠️ **多样性偏低**: 推荐结果相似度较高，建议扩展评分的电影类型以获得更多样化的推荐。")
                    
                    # 流行度偏差结论
                    if 0.8 < pop_bias < 1.2:
                        conclusions.append("✅ **流行度平衡**: 推荐在热门电影和冷门佳作之间保持了良好的平衡。")
                    elif pop_bias > 1.2:
                        conclusions.append("📈 **偏向热门**: 推荐结果倾向于热门电影，这些电影通常评价较高但缺乏惊喜。")
                    else:
                        conclusions.append("🔍 **偏向冷门**: 推荐结果倾向于小众电影，有助于发现独特作品但可能风险较高。")
                    
                    # 新颖性结论
                    if novelty > 9:
                        conclusions.append("✨ **新颖性优秀**: 推荐包含许多独特、小众的电影，能帮助您发现新的惊喜。")
                    elif novelty > 8:
                        conclusions.append("✨ **新颖性良好**: 推荐有一定的新颖性，包含一些不太主流的选择。")
                    else:
                        conclusions.append("📺 **新颖性一般**: 推荐偏向常见电影，如果想发现更多新作品，可以尝试评分一些冷门电影。")
                    
                    for conclusion in conclusions:
                        st.markdown(conclusion)
                    
                    # 总体建议
                    st.markdown("---")
                    st.markdown("**💡 个性化建议**:")
                    
                    suggestions = []
                    if diversity < 0.6:
                        suggestions.append("• 尝试评分不同类型的电影，以获得更多样化的推荐")
                    if pop_bias > 1.5:
                        suggestions.append("• 如果想发现小众佳作，可以尝试评分一些冷门但高质量的电影")
                    if novelty < 7:
                        suggestions.append("• 推荐结果较为保守，可以主动探索一些非主流类型")
                    
                    if suggestions:
                        for suggestion in suggestions:
                            st.markdown(suggestion)
                    else:
                        st.success("🎉 您的推荐结果质量很好，各项指标都表现优秀！")
                    
                    # 报告元信息
                    st.markdown("---")
                    st.caption(f"📅 报告生成时间: {report_time}")
                    st.caption(f"🔢 分析样本: {len(user_ratings)} 部输入电影, {len(recommendations)} 条推荐结果")
                    
                    # 生成下载用的文本报告
                    report_text = f"""
==============================================================
推荐系统分析报告
Recommender System Analysis Report
==============================================================

报告生成时间: {report_time}

1. 用户输入评分
{'='*60}
"""
                    for mid, rating in user_ratings.items():
                        title = "Unknown"
                        if movie_info is not None and mid in movie_info['movie_id'].values:
                            title = movie_info[movie_info['movie_id'] == mid]['title'].values[0]
                        report_text += f"电影 {mid}: {title} - 评分: {rating}\n"
                    
                    report_text += f"""
2. 推荐结果 (Top 10)
{'='*60}
"""
                    for rank, (item_id, score) in enumerate(recommendations, 1):
                        title = "Unknown"
                        if movie_info is not None and item_id in movie_info['movie_id'].values:
                            title = movie_info[movie_info['movie_id'] == item_id]['title'].values[0]
                        report_text += f"{rank}. {title} (ID: {item_id}) - 预测评分: {score:.3f}\n"
                    
                    report_text += f"""
3. 偏差分析
{'='*60}
多样性 (Diversity): {diversity:.4f}
流行度偏差 (Popularity Bias): {pop_bias:.4f}
新颖性 (Novelty): {novelty:.4f}

4. 分析结论
{'='*60}
"""
                    for conclusion in conclusions:
                        report_text += conclusion.replace('✅ ', '').replace('⚠️ ', '').replace('📈 ', '').replace('🔍 ', '').replace('✨ ', '').replace('📺 ', '').replace('**', '') + "\n"
                    
                    report_text += f"\n{'='*60}\n"
                    
                    # 下载按钮
                    st.markdown("---")
                    st.download_button(
                        label="📥 下载完整报告 (TXT格式)",
                        data=report_text,
                        file_name=f"recommendation_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

if __name__ == "__main__":
    main()