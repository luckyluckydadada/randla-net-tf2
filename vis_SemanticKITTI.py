from helper_tool import Plot
from os.path import join, dirname, abspath
from helper_tool import DataProcessing as DP
import numpy as np
import os
import pickle
import yaml
 
def get_file_list(dataset_path):
    seq_list = np.sort(os.listdir(dataset_path))
    test_file_list = []
    for seq_id in seq_list:
        seq_path = join(dataset_path, seq_id)
        pc_path = join(seq_path, 'velodyne')
        if int(seq_id) >= 11:
            for f in np.sort(os.listdir(pc_path)):
                test_file_list.append([join(pc_path, f)])
                break
    test_file_list = np.concatenate(test_file_list, axis=0)
    return test_file_list
 
def get_test_result_file_list(dataset_path):
    seq_list = np.sort(os.listdir(dataset_path))
    test_result_file_list = []
    for seq_id in seq_list:
        seq_path = join(dataset_path, seq_id)
        pred_path = join(seq_path, 'predictions')
        for f in np.sort(os.listdir(pred_path)):
            test_result_file_list.append([join(pred_path, f)])
            break
    test_file_list = np.concatenate(test_result_file_list, axis=0)
    return test_file_list
 
 
if __name__ == '__main__':
    dataset_path = '/home/lucky/data/semanticKITTI/dataset/sequences' 
    predict_path = '/home/lucky/data/semanticKITTI/dataset/test/sequences'
    test_list = get_file_list(dataset_path)
    test_label_list = get_test_result_file_list(predict_path)
    BASE_DIR = dirname(abspath(__file__))
 
    #  remap_lut  #
    data_config = join(BASE_DIR, 'utils', 'semantic-kitti.yaml')
    DATA = yaml.safe_load(open(data_config, 'r'))
    remap_dict = DATA["learning_map"]
    max_key = max(remap_dict.keys())
    remap_lut = np.zeros((max_key + 100), dtype=np.int32)
    remap_lut[list(remap_dict.keys())] = list(remap_dict.values())
    #  remap_lut  #

# （1）random_colors输入为N，N是要表达的类别数，如SemanticKITTI是20类，Semantic3D是9类。
# （2）draw_pc，输入x-y-z-r-g-b格式的数据绘图
# （3）draw_pc_sem_ins，输入pc_xyz是点云坐标信息，pc_sem_ins是对应的label，plot_colors是label对应的颜色，SemanticKITTI中，每个序列、每帧中的label是不一定的，如果要使每个label对应的颜色相同，需要提前调用random_colors或自己生成，然后输入进去。

    plot_colors = Plot.random_colors(21, seed=2)
 
    for i in range(len(test_list)):
        pc_path = test_list[i]
        labels_path = test_label_list[i]
        points = DP.load_pc_kitti(pc_path)
 
        # 用深蓝色画初始点云 #
        rpoints = np.zeros((points.shape[0],6),dtype=np.int)
        rpoints[:,0:3] = points
        rpoints[:,5] = 1
        Plot.draw_pc(rpoints)
 
        # 画对应的预测点云 #
        labels = DP.load_label_kitti(labels_path, remap_lut)
        Plot.draw_pc_sem_ins(points, labels,plot_colors)
