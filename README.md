# Explainable Recommender System Web Application (ERSWA) 可解释推荐系统 Web 应用
基于用户自定义偏好的推荐算法可解释性与偏差分析系统

> Authors: Jian Pang, Yongliang Ye, and Junjie Chen
> 
> Affiliation: School of Artificial Intelligence, South China Normal University, Foshan, Guangdong, China
> 
> Supervised by: Associate Professor Shouqiang Liu

## Introduction
这是一个交互式 Web 应用，允许用户：
- ✅ 选择一个推荐算法
- ✅ 输入自己的电影评分或选择喜欢的电影类型
- ✅ 获得个性化的 Top 10 电影推荐
- ✅ 查看每条推荐的详细解释（为什么推荐这部电影）
- ✅ 分析推荐结果的偏差指标（多样性、流行度偏差、新颖性）
- ✅ 下载完整的分析报告

## Quick Start

### Run in Local

environment request: `python=3.11`

create a special virtual environment for this project: `conda create -n recommender python=3.11 `

activate that virtual environment: `conda activate recommender`

install python dependecies `pip install -r requirements.txt`

run application `streamlit run app.py` and open http://localhost:8501 in your browser.

### Run in Docker

We recommend you running in docker. 

pull image: `docker pull 1226643780/erswa:1.0.0` we build the image in multi-arch to suport both amd64 and arm64.

run container: `docker run -p 8501:8501 1226643780/erswa:1.0.0` forward container port to your localhost.

## Contact
If you want to contribute, please contact:

Lead developer: pangjian0523@163.com

For questions or support, please use GitHub's issue system.