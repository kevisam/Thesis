# source https://github.com/niconielsen32/depth-anything/tree/main

import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import matplotlib.pyplot as plt


# encoders = ['vits', 'vitb', 'vitl']
encoder = "vits"
video_path = 0


margin_width = 50
caption_height = 60

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

depth_anything = DepthAnything.from_pretrained(
    "LiheYoung/depth_anything_{}14".format(encoder)
).to(DEVICE)

total_params = sum(param.numel() for param in depth_anything.parameters())
print("Total parameters: {:.2f}M".format(total_params / 1e6))

depth_anything.eval()

transform = Compose(
    [
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ]
)


# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or 'x264' might also be available
out_video = cv2.VideoWriter("output_video.mp4", fourcc, 30.0, (640, 480))


# cap = cv2.VideoCapture(video_path)

# while cap.isOpened():
#     ret, raw_image = cap.read()

#     if not ret:
#         break

#     raw_image = cv2.resize(raw_image, (640, 480))

#     image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

#     h, w = image.shape[:2]

#     image = transform({"image": image})["image"]
#     image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

#     with torch.no_grad():
#         depth = depth_anything(image)

#   depth = F.interpolate(depth[None], (h, w), mode="bilinear", align_corners=False)[
#       0, 0
#   ]
#   depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

#   depth = depth.cpu().numpy().astype(np.uint8)
#   depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

#     split_region = np.ones((raw_image.shape[0], margin_width, 3), dtype=np.uint8) * 255
#     combined_results = cv2.hconcat([raw_image, split_region, depth_color])

#     caption_space = (
#         np.ones((caption_height, combined_results.shape[1], 3), dtype=np.uint8) * 255
#     )
#     captions = ["Raw image", "Depth Anything"]
#     segment_width = w + margin_width
#     for i, caption in enumerate(captions):
#         # Calculate text size
#         text_size = cv2.getTextSize(caption, font, font_scale, font_thickness)[0]

#         # Calculate x-coordinate to center the textq
#         text_x = int((segment_width * i) + (w - text_size[0]) / 2)

#         # Add text caption
#         cv2.putText(
#             caption_space,
#             caption,
#             (text_x, 40),
#             font,
#             font_scale,
#             (0, 0, 0),
#             font_thickness,
#         )

#     final_result = cv2.vconcat([caption_space, combined_results])

#     # Write the frame to the video file
#     out_video.write(depth_color)
#     cv2.imshow("Depth Anything", final_result)

#     # Press q on keyboard to exit
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break


# cap.release()
# out_video.release()
# cv2.destroyAllWindows()


# Testing on a sample image

img_path = "/Users/kevinsam/Desktop/Unif/Master/ThesisWheelchair/ThesisProject/ThesisCode/DepthSensing/images/sidewalk2.jpeg"
img = cv2.imread(img_path)
img = cv2.resize(img, (640, 480))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
h, w = img.shape[:2]
img = transform({"image": img})["image"]
img = torch.from_numpy(img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    depth = depth_anything(img)

depth = F.interpolate(depth[None], (h, w), mode="bilinear", align_corners=False)[0, 0]
depth = (depth - depth.min()) / (depth.max() - depth.min())  # * 255.0
depth = 1 - depth  # inverse depth such that closer items have a smaller depth value


### Using depth map info to calculate distances in image.

camera_pos = [h / 2, w / 2, 0]


# reference object information
ro_distance = 100  # distance noted in centimeters

ro_position = [210, 421]
x = ro_position[0]
y = ro_position[1]


other_point = [532, 288]
ox = other_point[0]
oy = other_point[1]

# Plot the depth map using matplotlib
plt.figure(figsize=(8, 6))
plt.imshow(depth, cmap="inferno_r")

# Add the color bar using the same colormap
cbar = plt.colorbar(label="Depth")
cbar.set_label("Depth", rotation=270)

plt.scatter(x, y, color="red", marker="x", s=50)

plt.scatter(ox, oy, color="blue", marker="x", s=50)

plt.title("Depth Map")


# Adjust the color limits based on the range of depth values
plt.clim(depth.min(), depth.max())

op_distance = ro_distance / depth[y, x] * depth[oy, ox]


print(f"distance to reference object : {ro_distance} cm")
print(f"distance to other point : {op_distance} cm")

plt.show()
