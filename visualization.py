import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

'''
访问Results/Qwen/final.csv文件，表格的表头和内容如下
...1,Solutions,FacScoresQ,FacScoresO,Dataset,ProblemID,set,ID,wordcount,DSI,promptCosDis,peerCosDis
1,She could leave an anonymous note for the manager saying one of the employees was stealing.,-1.411283851,0.313362334,RLPS1,Becky,training,201,16,0.0,0.4247221946716308,0.4832862615585327

请你提取出problemID为"Joan"的所有行，关注他们的4个指标：DSI，promptCosDis,peerCosDis,FacScoresO

用matplotlib的三维散点图可视化，每个点的坐标分别对应DSI，promptCosDis,peerCosDis，用点的颜色表示FacScoresO
'''

import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('Results/bert/final.csv')

# 筛选出ProblemID为"Joan"的行
#joan_df = df[df['ProblemID'] == 'Joan']
joan_df = df

# 提取需要的列
x = joan_df['DSI']
y = joan_df['promptCosDis']
z = joan_df['peerCosDis']
color = joan_df['FacScoresO']

# 绘制三维散点图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(x, y, z, c=color, cmap='viridis')

ax.set_xlabel('DSI')
ax.set_ylabel('promptCosDis')
ax.set_zlabel('peerCosDis')
plt.colorbar(sc, label='FacScoresO')
plt.title('Joan Problem: 3D Visualization')

'''
[1.3351754  -1.77417215  2.55246432] -0.8963868624998153
场面四个参数对应x,y,z的一个预测器 a*x + b*y + c*z + d
请你画一条垂直于分类器平面的线，线上每个点的颜色对应预测值
'''
# # 分类器参数
# a, b, c, d = 1.3351754, -1.77417215, 2.55246432, -0.8963868624998153

# # 计算平面法向量
# normal = np.array([a, b, c])

# # 取平面中心点
# center = np.array([x.mean(), y.mean(), z.mean()])
# # 计算平面上的一点，使其满足 a*x + b*y + c*z + d = 0
# offset = -(a*center[0] + b*center[1] + c*center[2] + d)
# plane_point = center + normal * (offset / np.dot(normal, normal))

# # 沿法向量方向生成一条线
# t = np.linspace(0, 1, 100)
# line_points = plane_point[None, :] + t[:, None] * normal

# # 计算线上每个点的预测值
# predicted = a*line_points[:,0] + b*line_points[:,1] + c*line_points[:,2] + d

# # 绘制线，颜色映射预测值
# line = ax.plot(line_points[:,0], line_points[:,1], line_points[:,2], 
#                c='k', linewidth=2, label='Normal Line')[0]
# for i in range(len(line_points)-1):
#     ax.plot(line_points[i:i+2,0], line_points[i:i+2,1], line_points[i:i+2,2], 
#             color=plt.cm.viridis((predicted[i]-predicted.min())/(predicted.max()-predicted.min())))

# plt.legend()
plt.show()