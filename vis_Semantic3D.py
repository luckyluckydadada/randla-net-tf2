from helper_tool import Plot
from os.path import join, exists
from helper_ply import read_ply
import numpy as np
import os
 
 
if __name__ == '__main__':
    path = '/../../data'
    label_folder = './test/Log_2020-04-09_03-32-36/predictions/'
 
    label_to_names = {0: 'unlabeled',
                           1: 'man-made terrain',
                           2: 'natural terrain',
                           3: 'high vegetation',
                           4: 'low vegetation',
                           5: 'buildings',
                           6: 'hard scape',
                           7: 'scanning artefacts',
                           8: 'cars'}
 
 
    original_folder = join(path, 'semantic3d')
    full_pc_folder = join(path, 'original_ply')
 
    test_files_names = []
    cloud_names = [file_name[:-4] for file_name in os.listdir(original_folder) if file_name[-4:] == '.txt']
    for pc_name in cloud_names:
        if not exists(join(original_folder, pc_name + '.labels')):
            test_files_names.append(pc_name + '.ply')
    test_files_names = np.sort(test_files_names)
    # Ascii files dict for testing
    ascii_files = {
        'MarketplaceFeldkirch_Station4_rgb_intensity-reduced.ply': 'marketsquarefeldkirch4-reduced.labels',
        'sg27_station10_rgb_intensity-reduced.ply': 'sg27_10-reduced.labels',
        'sg28_Station2_rgb_intensity-reduced.ply': 'sg28_2-reduced.labels',
        'StGallenCathedral_station6_rgb_intensity-reduced.ply': 'stgallencathedral6-reduced.labels',
        'birdfountain_station1_xyz_intensity_rgb.ply': 'birdfountain1.labels',
        'castleblatten_station1_intensity_rgb.ply': 'castleblatten1.labels',
        'castleblatten_station5_xyz_intensity_rgb.ply': 'castleblatten5.labels',
        'marketplacefeldkirch_station1_intensity_rgb.ply': 'marketsquarefeldkirch1.labels',
        'marketplacefeldkirch_station4_intensity_rgb.ply': 'marketsquarefeldkirch4.labels',
        'marketplacefeldkirch_station7_intensity_rgb.ply': 'marketsquarefeldkirch7.labels',
        'sg27_station10_intensity_rgb.ply': 'sg27_10.labels',
        'sg27_station3_intensity_rgb.ply': 'sg27_3.labels',
        'sg27_station6_intensity_rgb.ply': 'sg27_6.labels',
        'sg27_station8_intensity_rgb.ply': 'sg27_8.labels',
        'sg28_station2_intensity_rgb.ply': 'sg28_2.labels',
        'sg28_station5_xyz_intensity_rgb.ply': 'sg28_5.labels',
        'stgallencathedral_station1_intensity_rgb.ply': 'stgallencathedral1.labels',
        'stgallencathedral_station3_intensity_rgb.ply': 'stgallencathedral3.labels',
        'stgallencathedral_station6_intensity_rgb.ply': 'stgallencathedral6.labels'}
    
    plot_colors = Plot.random_colors(11, seed=2)
    for file in test_files_names:
        print(file)
        test_files = join(full_pc_folder, file)
        label_files = join(label_folder, ascii_files[file])
        data = read_ply(test_files)
        # 绘制原图
        pc_xyzrgb = np.vstack((data['x'], data['y'], data['z'], data['red'], data['green'], data['blue'])).T
        Plot.draw_pc(pc_xyzrgb)
        # 绘制预测结果图
        pc_xyz = np.vstack((data['x'], data['y'], data['z'])).T
        pc_sem_ins = np.loadtxt(label_files)
        pc_sem_ins = pc_sem_ins.astype(int)
        Plot.draw_pc_sem_ins(pc_xyz, pc_sem_ins,plot_colors)
