import os
import shutil


def organize_data(dataset_path, subset):
    # List of materials
    materials = ["asphalt", "cobble", "concrete"]

    # Iterate over each material
    for material in materials:
        # Create new folders for dry and wet in the subset
        dry_folder = os.path.join(
            "/Users/kevinsam/Desktop/Unif/Master/ThesisWheelchair/DATA/RoadSaW/RoadSaW6-075_l",
            subset,
            f"{material}_dry",
        )
        wet_folder = os.path.join(
            "/Users/kevinsam/Desktop/Unif/Master/ThesisWheelchair/DATA/RoadSaW/RoadSaW6-075_l",
            subset,
            f"{material}_wet",
        )

        # Create folders if they don't exist
        os.makedirs(dry_folder, exist_ok=True)
        os.makedirs(wet_folder, exist_ok=True)

        # Iterate over dry and damp
        for condition in ["dry", "damp"]:
            source_folder = os.path.join(
                dataset_path, subset, f"{material}_{condition}"
            )
            # Copy files from source to destination
            for file_name in os.listdir(source_folder):
                source_path = os.path.join(source_folder, file_name)
                destination_path = os.path.join(dry_folder, file_name)
                shutil.copy(source_path, destination_path)

        # Iterate over wet and verywet
        for condition in ["wet", "verywet"]:
            source_folder = os.path.join(
                dataset_path, subset, f"{material}_{condition}"
            )
            # Copy files from source to destination
            for file_name in os.listdir(source_folder):
                source_path = os.path.join(source_folder, file_name)
                destination_path = os.path.join(wet_folder, file_name)
                shutil.copy(source_path, destination_path)

    print(f"{subset.capitalize()} dataset organization completed.")


# Specify the path to your dataset
dataset_path = (
    "/Users/kevinsam/Desktop/Unif/Master/ThesisWheelchair/DATA/RoadSaW/RoadSaW-075_l"
)

# Organize train, validation, and test datasets
organize_data(dataset_path, "train")
organize_data(dataset_path, "validation")
organize_data(dataset_path, "test")
