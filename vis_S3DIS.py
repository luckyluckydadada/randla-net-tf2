from helper_tool import Plot
from helper_ply import read_ply
import numpy as np
import os

basedir="/home/lucky/data/S3DIS/original_ply/"  # for label

vis_list=os.listdir(basedir)
plot_colors = Plot.random_colors(13, seed=2) #13 CLASS
Flag=True
for name in vis_list:
    print(basedir+name)
    # data = read_ply(basedir+name)
    data = read_ply("/home/lucky/data/S3DIS/original_ply/Area_1_office_16.ply") 
    if Flag:
        PID = os.fork()
        Flag=False
    if PID == 0:
        pc_xyzrgb = np.vstack((data['x'], data['y'], data['z'], data['red'], data['green'], data['blue'])).T
        Plot.draw_pc(pc_xyzrgb)# 绘制原图
    else:
        pc_xyz = np.vstack((data['x'], data['y'], data['z'])).T
        pc_sem_ins = np.vstack((data['class']))
        Plot.draw_pc_sem_ins(pc_xyz, pc_sem_ins)# 绘制预测结果图

# basedir="./test/Log_2021-08-16_01-25-39/val_preds/"  # for test
# vis_list=os.listdir(basedir)
# plot_colors =[
#     (1.0, 0.0, 0.9461538461538463), 
#     (0.0, 1.0, 0.2846153846153845), 
#     (0.0, 0.3307692307692305, 1.0), 
#     (1.0, 0.4384615384615387, 0.0), 
#     (1.0, 0.0, 0.48461538461538467), 
#     (0.1307692307692303, 0.0, 1.0), 
#     (0.5923076923076929, 0.0, 1.0), 
#     (0.0, 1.0, 0.7461538461538462), 
#     (0.17692307692307674, 1.0, 0.0), 
#     (0.0, 0.7923076923076922, 1.0), 
#     (1.0, 0.0, 0.023076923076922995), 
#     (0.6384615384615384, 1.0, 0.0),
#      (1.0, 0.8999999999999999, 0.0)]
# # plot_colors =Plot.random_colors(13, seed=2) #13 CLASS
# Flag=True
# for name in vis_list:
#     pred_path=basedir+name
#     original_path="/home/lucky/data/S3DIS/original_ply/"+name
#     print(pred_path)
#     pred_data = read_ply(pred_path)
#     pred = np.vstack((pred_data['pred']))
#     label = np.vstack((pred_data['label']))
#     original_data = read_ply(original_path)
#     xyz = np.vstack((original_data['x'], original_data['y'], original_data['z'])).T
#     if Flag:
#         PID = os.fork()
#         Flag=False
#     if PID == 0:
#         Plot.draw_pc_sem_ins(xyz, pred,plot_colors) 
#     else:
#         Plot.draw_pc_sem_ins(xyz, label,plot_colors) 