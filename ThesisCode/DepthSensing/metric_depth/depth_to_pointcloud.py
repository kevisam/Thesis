# Born out of Issue 36. 
# Allows  the user to set up own test files to infer on (Create a folder my_test and add subfolder input and output in the metric_depth directory before running this script.)
# Make sure you have the necessary libraries
# Code by @1ssb
import csv

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
from SidewalkSegmentation.segmentation_utils import *
# Global settings
#Modify these settings to adjust to specific camera setings
FL = 715.0873
FY = 300
FX = 300
#FY = 256 * 0.6
#FX = 256 * 0.6
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


def convert_2d_to_3d(pixel_x,pixel_y,depth_map):

    # Retrieve depth value from the depth map or estimation model output
    depth_value = depth_map[pixel_y, pixel_x]  # Example: retrieve from depth map

    img_width, img_height = FINAL_WIDTH,FINAL_HEIGHT
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


def original_to_resized_pixel(original_pixel, original_size, resized_size):
    """
    Convert a pixel from the original image to the corresponding pixel in the resized image.
    
    Parameters:
        original_pixel (tuple): Pixel coordinates (x, y) in the original image.
        original_size (tuple): Original image size (original_width, original_height).
        resized_size (tuple): Resized image size (resized_width, resized_height).
    
    Returns:
        resized_pixel (tuple): Pixel coordinates (x_resized, y_resized) in the resized image.
    """
    print(f"orig size {original_size} ")
    # Unpack original pixel coordinates
    x_original = original_pixel[0]
    y_original = original_pixel[1]
    
    # Unpack original and resized image sizes
    original_width, original_height = original_size
    resized_width, resized_height = resized_size
    
    # Calculate scaling factors for width and height
    scale_x = resized_width / original_width
    scale_y = resized_height / original_height
    
    # Calculate corresponding pixel coordinates in the resized image
    x_resized = int(x_original * scale_x)
    y_resized = int(y_original * scale_y)
    
    return x_resized, y_resized

def euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two 3D points.
    
    Parameters:
        point1 (tuple or list): Coordinates of the first point (x1, y1, z1).
        point2 (tuple or list): Coordinates of the second point (x2, y2, z2).
    
    Returns:
        distance (float): Euclidean distance between the two points.
    """
    # Convert input points to numpy arrays
    point1 = np.array(point1)
    point2 = np.array(point2)
    
    # Calculate Euclidean distance using numpy's linalg.norm function
    distance = np.linalg.norm(point2 - point1)
    
    return distance


def process_images(model):
    print("Processing started")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Load semgmentation model directly
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
    from PIL import Image

    processor = AutoImageProcessor.from_pretrained("nickmuchi/segformer-b4-finetuned-segments-sidewalk")
    seg_model = SegformerForSemanticSegmentation.from_pretrained("nickmuchi/segformer-b4-finetuned-segments-sidewalk")
    
    #image_paths = glob.glob(os.path.join(INPUT_DIR, '*.png')) + glob.glob(os.path.join(INPUT_DIR, '*.jpg')) + glob.glob(os.path.join(INPUT_DIR, '*.jpeg'))
    
        # List of folder sizes
    folder_sizes = ["90", "115", "120", "150", "160", "175", "210", "240", "290", "360"]

    # List to store image paths
    image_paths = []

    # Iterate over each folder size
    for size in folder_sizes:
        # Search for PNG files
        jpg_files = glob.glob(os.path.join(INPUT_DIR, size, "*.jpg"))
        # Search for JPEG files
        jpeg_files = glob.glob(os.path.join(INPUT_DIR, size, "*.jpeg"))
        
        # Extend the image_paths list with found PNG and JPEG files
        image_paths.extend(jpg_files)
        image_paths.extend(jpeg_files)
    
    estimation_data = []
    for image_path in tqdm(image_paths, desc="Processing Images"):
        try:
            color_image = Image.open(image_path).convert('RGB')
            if color_image.height < color_image.width: # if image is horizontal, make it vertical
                color_image = color_image.rotate(-90, Image.NEAREST, expand = 1)
            original_width, original_height = color_image.size
            
            print(f'Image size : {color_image.size}')
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
            
            #Making axes for visualisation
            fig, axes = plt.subplots(1, 3, figsize=(24, 12))
            
            ###########Segment sidewalk
            inputs = processor(images=color_image, return_tensors="pt")
            outputs = seg_model(**inputs)
            logits = outputs.logits.squeeze().detach().cpu().numpy()
            make_mask(seg_model,logits)
            
            mask = cv.imread('sidewalk_mask.png')
            largest_contour = find_sidewalk_contours(mask)
            intersecting_points = points_to_calc_distance(mask,largest_contour)
                
            side1, side2 = separate_sides(intersecting_points,20)
            
            #points at both side of the sidewalk
            point1 = average_point(side1)
            point2 = average_point(side2)
            
            mask_width, mask_height,_ = mask.shape 
            resized_point1 = original_to_resized_pixel(point1,[mask_width,mask_height],[FINAL_WIDTH,FINAL_HEIGHT])
            resized_point2 = original_to_resized_pixel(point2,[mask_width,mask_height],[FINAL_WIDTH,FINAL_HEIGHT])
            
            #Project 2d points in 3d space
            point1_3d = convert_2d_to_3d(resized_point1[0],resized_point1[1],np.array(resized_pred))
            point2_3d = convert_2d_to_3d(resized_point2[0],resized_point2[1],np.array(resized_pred))
            
            #Calculate Euclidean distance in 3d space
            distance = euclidean_distance(point1_3d,point2_3d)
            print(f"Image: {image_path}")
            print(f"The sidewalk is estimated to be {distance}m long")
            estimation_data.append({"image_name": image_path, "true_size": os.path.basename(image_path), "estimated_size": distance})
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
            
            # ## Plotting for visualisation
            # # Plot original image
            # axes[0].imshow(np.asarray(color_image))
            # axes[0].set_title('Original Image')
            
            # axes[1].imshow(mask)
            # axes[1].set_title('Sidewalk Mask')
            
            # axes[1].scatter(point1[0],point1[1], color="red", marker="x", s=50)
            # axes[1].scatter(point2[0],point2[1], color="red", marker="x", s=50)
            
            # # Plot original image
            # axes[2].imshow(resized_color_image)
            # axes[2].set_title('Resized Image')
            
            # axes[2].scatter(resized_point1[0],resized_point1[1], color="red", marker="x", s=50)
            # axes[2].scatter(resized_point2[0],resized_point2[1], color="red", marker="x", s=50)
            # plt.show()       
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            
    # Make a CSV to compare the original size and the estimated size
    csv_file_path = "image_sizes.csv"

    # Writing data to CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["image_name", "true_size", "estimated_size"])
        
        # Write header
        writer.writeheader()
        
        # Write data
        for row in estimation_data:
            writer.writerow(row)

    print("CSV file created successfully:", csv_file_path)

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
