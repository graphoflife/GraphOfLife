import os
import shutil


def delete_folders_with_few_pngs(directory, min_png_count=10):
    if not os.path.exists(directory):
        print(f"The directory {directory} does not exist.")
        return

    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)

        # Check if the item is indeed a directory
        if os.path.isdir(folder_path):
            png_files = [file for file in os.listdir(folder_path) if file.lower().endswith('.png')]

            if len(png_files) < min_png_count:
                # Delete the folder
                shutil.rmtree(folder_path)
                print(f"Deleted folder: {folder_path}")


directory = r'D:\NewGraphCompetition Outputs\outputs16'  # Replace with your folder path
delete_folders_with_few_pngs(directory, min_png_count=30)