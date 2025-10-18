# 🎬 Explainable Recommender System Web Application (ERSWA) 可解释推荐系统 Web 应用
基于用户自定义偏好的推荐算法可解释性与偏差分析系统

> Authors: Jian Pang, Yongliang Ye, and Junjie Chen
> 
> Affiliation: School of Artificial Intelligence, South China Normal University, Foshan, Guangdong, China
> 
> Supervised by: Associate Professor Shouqiang Liu

## 📋 Introduction
这是一个交互式 Web 应用，允许用户：
- ✅ 输入自己的电影评分或选择喜欢的电影类型
- ✅ 获得个性化的 Top 10 电影推荐
- ✅ 查看每条推荐的详细解释（为什么推荐这部电影）
- ✅ 分析推荐结果的偏差指标（多样性、流行度偏差、新颖性）
- ✅ 下载完整的分析报告

## Quick Start

运行环境要求 `python=3.11`

`conda create -n recommender python=3.11 ` 为该项目创建一个专门的虚拟环境

`conda activate recommender` 激活该项目的虚拟环境

`pip install -r requirements.txt` 安装依赖

`streamlit run app.py` 运行web应用，浏览器会自动打开 http://localhost:8501