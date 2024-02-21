from open3d import *    


cloud = io.read_point_cloud("ThesisCode\DepthSensing\metric_depth\my_test\output\sidewalk2.ply") # Read point cloud
visualization.draw_geometries([cloud])    # Visualize point cloud      