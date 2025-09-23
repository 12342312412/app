import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

# 创建一个可伸缩的图表，调整宽度为12
fig1, ax1 = plt.subplots(figsize=(8, 8))  # 调整 figsize 参数控制图表大小

# 生成随机数据，确保是小数形式
data = np.random.rand(7, 3)  # 7天，3个症状

# 归一化每一列，使每一列的和为1
data = data / data.sum(axis=0, keepdims=True)

# 设置横纵坐标标签
ax1.set_xticks(range(3))
ax1.set_xticklabels(['Symptom1', 'Symptom2', 'Symptom3'])  # 横坐标命名为症状
ax1.set_yticks(range(7))
ax1.set_yticklabels([f'{i+1}day' for i in range(7)])  # 纵坐标命名为天数

# 填充数据
cax1 = ax1.matshow(data, cmap='viridis')

# 添加颜色条
fig1.colorbar(cax1)

# 在每个格子中添加数值标签
for (i, j), value in np.ndenumerate(data):
    ax1.text(j, i, f'{value:.2f}', ha='center', va='center', color='white')

# 设置标题
ax1.set_title('The Relationship between Symptoms and Duration of Days')

# 使用 st.columns 实现两幅图并排显示
col1, col2 = st.columns(2)

with col1:
    st.pyplot(fig1)

with col2:
    # 创建多选框，允许用户选择需要显示的折线
    symptoms = ['Symptom1', 'Symptom2', 'Symptom3']
    selected_symptoms = st.multiselect('Select symptoms to display:', symptoms, default=symptoms)

    # 创建第二幅折线图
    fig2, ax2 = plt.subplots(figsize=(12, 6))  # 调整 figsize 参数控制图表大小

    # 横坐标是天数
    days = range(1, 8)

    # 绘制选定的折线
    for i, symptom in enumerate(symptoms):
        if symptom in selected_symptoms:
            ax2.plot(days, data[:, i], label=f'{symptom}')

    # 设置横纵坐标标签
    ax2.set_xlabel('Days')
    ax2.set_ylabel('Probability')

    # 添加图例
    ax2.legend()

    # 设置标题
    ax2.set_title('Probability of Symptoms over Days')

    # 显示第二幅图
    st.pyplot(fig2)

# 生成10个人出现症状的天数
st.write("### 10 People's Symptom Days")

# 根据概率生成10个人出现症状的天数
people_data = np.zeros((10, 3), dtype=int)

# 为每个症状生成随机天数
for i in range(3):
    people_data[:, i] = np.random.choice(range(1, 8), size=10, p=data[:, i])

# 创建表格
people_df = pd.DataFrame(people_data, columns=['Symptom1', 'Symptom2', 'Symptom3'])

# 显示表格
st.write(people_df)

# 生成核密度估计图
st.write("### Kernel Density Estimation (KDE) of Symptom Days")

# 创建核密度估计图
fig3, ax3 = plt.subplots(figsize=(12, 6))

# 为每个症状生成KDE曲线
for i, symptom in enumerate(symptoms):
    kde = gaussian_kde(people_data[:, i])
    x = np.linspace(1, 7, 1000)
    ax3.plot(x, kde(x), label=f'{symptom}')

# 设置横纵坐标标签
ax3.set_xlabel('Days')
ax3.set_ylabel('Density')

# 添加图例
ax3.legend()

# 设置标题
ax3.set_title('KDE of Symptom Days for 10 People')

# 显示核密度估计图
st.pyplot(fig3)
