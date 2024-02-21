# Born out of Issue 36. 
# Allows  the user to set up own test files to infer on (Create a folder my_test and add subfolder input and output in the metric_depth directory before running this script.)
# Make sure you have the necessary libraries
# Code by @1ssb

import argparse
import os
import glob
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import open3d as o3d
from tqdm import tqdm
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.misc import colorize
from matplotlib import pyplot as plt
# Global settings
#Modify these settings to adjust to specific camera setings
FL = 715.0873
FY = 256 * 0.6
FX = 256 * 0.6
NYU_DATA = False
FINAL_HEIGHT = 256
FINAL_WIDTH = 256
INPUT_DIR = './my_test/input'
OUTPUT_DIR = './my_test/output'
DATASET = 'nyu' # Lets not pick a fight with the model's dataloader

def find_closest_point_in_point_cloud(x, y, z, point_cloud_data):
    """
    Find the closest point in the point cloud data to the given 3D coordinates (x, y, z).
    
    Parameters:
        x (float): X-coordinate of the point.
        y (float): Y-coordinate of the point.
        z (float): Z-coordinate of the point.
        point_cloud_data (numpy.ndarray): Array containing the point cloud data, where each row represents a point
                                           with three coordinates (X, Y, Z).
    
    Returns:
        closest_point (numpy.ndarray): Coordinates of the closest point in the point cloud data.
    """
    # Calculate Euclidean distances between the given point and all points in the point cloud data
    distances = np.linalg.norm(point_cloud_data - np.array([x, y, z]), axis=1)
    
    # Find the index of the point with the minimum distance
    closest_index = np.argmin(distances)
    
    # Retrieve the closest point from the point cloud data
    closest_point = point_cloud_data[closest_index]
    
    return closest_point


def convert_2d_to_3d(pixel_x,pixel_y,img,depth_map):

    # Retrieve depth value from the depth map or estimation model output
    depth_value = depth_map[pixel_y, pixel_x]  # Example: retrieve from depth map

    img_width, img_height = img.size
    img_center_x = img_width/2
    img_center_y = img_height/2
    focal_length_x, focal_length_y = (FX, FY) if not NYU_DATA else (FL, FL)
    
    # Reverse projection to obtain 3D coordinates in camera space
    x_ndc = (pixel_x - img_center_x) / focal_length_x
    y_ndc = (pixel_y - img_center_y) / focal_length_y
    x_cam = depth_value * x_ndc
    y_cam = depth_value * y_ndc
    z_cam = depth_value

    return x_cam,y_cam,z_cam

    # #TODO
    
    # - Find how to convert original image pixel into resized image pixel
    # - Try to use the conversion to calculate distance between two sides of sidewalk

def process_images(model):
    print("Processing started")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    image_paths = glob.glob(os.path.join(INPUT_DIR, '*.png')) + glob.glob(os.path.join(INPUT_DIR, '*.jpg'))
    for image_path in tqdm(image_paths, desc="Processing Images"):
        try:
            color_image = Image.open(image_path).convert('RGB')
            original_width, original_height = color_image.size
            image_tensor = transforms.ToTensor()(color_image).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')

            pred = model(image_tensor, dataset=DATASET)
            if isinstance(pred, dict):
                pred = pred.get('metric_depth', pred.get('out'))
            elif isinstance(pred, (list, tuple)):
                pred = pred[-1]
            pred = pred.squeeze().detach().cpu().numpy()

            # Resize color image and depth to final size 
            # Output is resized because the model outputs this size
            resized_color_image = color_image.resize((FINAL_WIDTH, FINAL_HEIGHT), Image.LANCZOS)
            resized_pred = Image.fromarray(pred).resize((FINAL_WIDTH, FINAL_HEIGHT), Image.NEAREST)
            
            fig, axes = plt.subplots(1, 2, figsize=(24, 12))
            # Plot original image
            axes[0].imshow(color_image)
            axes[0].set_title('Original Image')
            
            # Plot original image
            axes[0].imshow(resized_color_image)
            axes[0].set_title('Resized Image')
            
            plt.show()
            
            # Plot the colormap for the metric depth perception
            # plt.figure(figsize=(8, 6))
            # plt.imshow(pred, cmap='inferno_r',vmin=np.min(pred), vmax=np.max(pred))  # Choose colormap (e.g., 'viridis')
            # plt.colorbar(label='Depth')  # Add colorbar with label
            # plt.title('Depth Map Colormap')  # Add title
            # plt.xlabel('Pixel X')  # Add x-axis label
            # plt.ylabel('Pixel Y')  # Add y-axis label
            # plt.show()
            
            #x and y coordinates are calculated using 
            focal_length_x, focal_length_y = (FX, FY) if not NYU_DATA else (FL, FL)
            x, y = np.meshgrid(np.arange(FINAL_WIDTH), np.arange(FINAL_HEIGHT))
            x = (x - FINAL_WIDTH / 2) / focal_length_x
            y = (y - FINAL_HEIGHT / 2) / focal_length_y
            z = np.array(resized_pred)
            points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
            colors = np.array(resized_color_image).reshape(-1, 3) / 255.0

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(image_path))[0] + ".ply"), pcd)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

def main(model_name, pretrained_resource):
    config = get_config(model_name, "eval", DATASET)
    config.pretrained_resource = pretrained_resource
    model = build_model(config).to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    process_images(model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default='zoedepth', help="Name of the model to test")
    parser.add_argument("-p", "--pretrained_resource", type=str, default='local::./checkpoints/depth_anything_metric_depth_outdoor.pt', help="Pretrained resource to use for fetching weights.")

    args = parser.parse_args()
    main(args.model, args.pretrained_resource)
