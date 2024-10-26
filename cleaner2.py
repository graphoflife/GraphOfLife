import os
import shutil


def delete_folders_with_few_pngs(directory, min_png_count=10):
    if not os.path.exists(directory):
        print(f"The directory {directory} does not exist.")
        return


    for folder1 in os.listdir(directory):
        try:
            folder_path1 = os.path.join(directory, folder1)
            print(folder_path1)
            for folder in os.listdir(folder_path1):

                    folder_path = os.path.join(folder_path1, folder)

                    # Check if the item is indeed a directory
                    if os.path.isdir(folder_path):
                        png_files = [file for file in os.listdir(folder_path) if file.lower().endswith('.png')]

                        if len(png_files) < min_png_count:
                            # Delete the folder
                            shutil.rmtree(folder_path)
                            print(f"Deleted folder: {folder_path}")
        except:
            pass


directory = r"D:\NewGraphCompetition Outputs\Outputs6" # Replace with your folder path
delete_folders_with_few_pngs(directory, min_png_count=20)