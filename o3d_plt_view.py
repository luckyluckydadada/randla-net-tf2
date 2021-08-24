import sys
import open3d as o3d

base="/home/lucky/map/gongdi1_filter_voxel_inliers.pcd"
pcd = o3d.io.read_point_cloud(base)
o3d.visualization.draw_geometries([pcd])


# pcd = o3d.io.read_point_cloud(sys.argv[1])
# o3d.visualization.draw_geometries([pcd])

