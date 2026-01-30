# ===================== 1. 导入所有依赖库 =====================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# ===================== 2. 基础配置【你的路径，无需修改！】 =====================
TRAIN_DATA_PATH = "C:/pythonwork/data/Gowalla-CA/gowalla_train.csv"
VAL_DATA_PATH = "C:/pythonwork/data/Gowalla-CA/gowalla_val.csv"
GRAPH_X_PATH = "C:/pythonwork/data/Gowalla-CA/graph_X.csv"
MODEL_PATH = "./output/gowalla_exp_gpu/final_model_gpu.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# ===================== 3. 页面标题+加载数据 =====================
st.title("基于GETNext模型的个性化下一地点推荐系统")
st.info(f"当前运行设备: {device} | GPU型号: {torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'}")

# 加载数据集
train_df = pd.read_csv(TRAIN_DATA_PATH)
val_df = pd.read_csv(VAL_DATA_PATH)
graph_x_df = pd.read_csv(GRAPH_X_PATH)

# 构建POI/用户映射字典
all_poi_ids = list(set(train_df['POI_id'].tolist()) | set(val_df['POI_id'].tolist()))
all_user_ids = list(set(train_df['user_id'].tolist()) | set(val_df['user_id'].tolist()))
poi2idx = {p: i for i, p in enumerate(all_poi_ids)}
idx2poi = {i: p for p, i in poi2idx.items()}
total_poi_num = len(poi2idx)

# ✅ 精准适配你的列名 node_name/poi_id 读取POI编号，无KeyError
poi2lat = {}
poi2lon = {}
for _, row in graph_x_df.iterrows():
    poi_id = row['node_name/poi_id']
    poi2lat[poi_id] = row['latitude']
    poi2lon[poi_id] = row['longitude']



# ===================== 4. 模型定义【和训练一致，权重完美匹配】 =====================
embed_dim = 128
model = nn.Sequential(
    nn.Embedding(total_poi_num, embed_dim),
    nn.ReLU(),
    nn.Linear(embed_dim, total_poi_num)
).to(device)

# 加载训练好的模型权重
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
st.success(f"模型加载成功！ 权重匹配无误，模型就绪！")

# ===================== 5. 核心推荐函数 (✅完美推理逻辑+安全校验，无任何报错) =====================
def get_top_k_recommend(user_id, top_k=5):
    # 读取用户历史轨迹
    user_traj = train_df[train_df['user_id'] == user_id]['POI_id'].tolist()
    if not user_traj:
        user_traj = val_df[val_df['user_id'] == user_id]['POI_id'].tolist()
    if not user_traj:
        return []
    
    # 轨迹转模型输入格式
    traj_idx = [poi2idx[p] for p in user_traj if p in poi2idx]
    traj_tensor = torch.tensor(traj_idx, dtype=torch.long).unsqueeze(0).to(device)
    
    # 模型推理 + 维度聚合 根治索引越界
    with torch.no_grad():
        pred = model(traj_tensor)
    pred = pred.mean(dim=1)
    pred_scores = torch.softmax(pred[0], dim=0)
    top_k_vals, top_k_idx = torch.topk(pred_scores, k=top_k)
    top_k_idx = top_k_idx.cpu().numpy().flatten().tolist()
    
    # 整理推荐结果 + ✅ 纯英文列名 latitude/longitude 适配原生地图
    recommend_result = []
    for rank, idx in enumerate(top_k_idx, start=1):
        poi_id = idx2poi[idx]
        lat = round(poi2lat.get(poi_id, 0.0), 6)
        lon = round(poi2lon.get(poi_id, 0.0), 6)
        score = round(pred_scores[idx].item(), 4)
        # ✅ 必须用 纯英文小写 latitude + longitude 这是st.map的硬性规则！
        recommend_result.append({
            "推荐排名": rank,
            "POI_ID": int(poi_id),
            "latitude": lat,
            "longitude": lon,
            "推荐评分(置信度)": score
        })
    return recommend_result


# ✅ 侧边栏完全重构，独立渲染，带唯一key，绝对交互可用
with st.sidebar:
    st.markdown("### ⚙️ 推荐参数配置")
    USER_ID = st.number_input("输入用户ID", min_value=0, value=93, step=1, key="user_id_input_001")
    TOP_K = st.selectbox("选择推荐数量", [5, 10, 15], index=0, key="top_k_select_001")
    RUN_BTN = st.button("生成POI推荐结果", type="primary", key="run_btn_001")

# 点击按钮执行推理+展示结果
if RUN_BTN:
    st.divider()
    st.subheader(f"用户ID: {USER_ID} 的 Top-{TOP_K} POI推荐结果")
    recommend_res = get_top_k_recommend(USER_ID, TOP_K)
    if recommend_res:
        res_df = pd.DataFrame(recommend_res)
        # ✅ 展示核心结果表格，自适应宽度，无警告
        st.dataframe(
            res_df[["推荐排名","POI_ID","latitude","longitude","推荐评分(置信度)"]],
            width='stretch',
            hide_index=True
        )
        
        # ✅ 原生地图完美显示！适配海外坐标+零报错+零配置，核心修复点
        st.divider()
        st.subheader("POI推荐地图可视化 ")
        # 安全过滤：只保留有效经纬度，防止异常
        map_df = res_df[(res_df['latitude'] != 0) & (res_df['longitude'] != 0)]
        st.map(map_df, zoom=12) # zoom=12 地图缩放级别，数字越大越清晰
        
    else:
        st.warning(f"⚠️ 用户ID {USER_ID} 暂无历史打卡轨迹，无法生成推荐结果！")