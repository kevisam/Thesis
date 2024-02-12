import os
import shutil

# Set the path to your original dataset
dataset_path = (
    "/Users/kevinsam/Desktop/Unif/Master/ThesisWheelchair/DATA/RoadSaW/RoadSaW6-075_l"
)

# Set the path where you want to create the new merged dataset
merged_dataset_path = (
    "/Users/kevinsam/Desktop/Unif/Master/ThesisWheelchair/DATA/RoadSaW/MergedDataset"
)

# Create the merged dataset folder if it doesn't exist
if not os.path.exists(merged_dataset_path):
    os.makedirs(merged_dataset_path)

# List all set folders
set_folders = ["train", "validation", "test"]

# List all class folders
class_folders = [
    "asphalt_dry",
    "asphalt_wet",
    "cobble_dry",
    "cobble_wet",
    "concrete_dry",
    "concrete_wet",
]

# Iterate through each set folder
for set_folder in set_folders:
    # Iterate through each class folder
    for class_folder in class_folders:
        class_path = os.path.join(dataset_path, set_folder, class_folder)

        # Check if the class folder exists before attempting to copy
        if os.path.exists(class_path):
            # Function to copy files from source to destination
            def copy_files(source, destination):
                for file_name in os.listdir(source):
                    source_file = os.path.join(source, file_name)
                    destination_file = os.path.join(destination, file_name)
                    shutil.copy(source_file, destination_file)

            # Copy all files from the class folder to the merged dataset folder
            merged_class_path = os.path.join(merged_dataset_path, class_folder)
            os.makedirs(merged_class_path, exist_ok=True)
            copy_files(class_path, merged_class_path)
        else:
            print(f"Class folder not found: {class_path}")

print("Dataset merged successfully.")
