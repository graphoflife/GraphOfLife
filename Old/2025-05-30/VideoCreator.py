import os
from moviepy.editor import ImageSequenceClip
from natsort import natsorted

# Set the path to the directory containing the numbered PNG files

path = r'D:\NewGraphCompetition Outputs\Single2'
min_amount = 1
make_one = True

all_pngs = []
name = "Video1"

for cur_dir in os.listdir(path):

    name = cur_dir
    png_dir = os.path.join(path, name)
    print(name)

    # Create a list of file paths to the PNG files
    png_files = [os.path.join(png_dir, f) for f in os.listdir(png_dir) if f.endswith('.png')]

    # Sort the file paths so that they are in the correct order
    png_files = natsorted(png_files, reverse=False)  # Sort the images
    png_files = png_files[:-1]
    if len(png_files) > min_amount:
        # Create a video file from the PNG files using a frame rate of 30 frames per second

        if make_one == False:
            clip = ImageSequenceClip(png_files, fps=60)

            # Write the video file to disk
            clip.write_videofile(fr'{name}.mp4')
            del clip
            del png_files
        else:
            all_pngs += png_files

if make_one == True:

    clip = ImageSequenceClip(all_pngs, fps=30)

    # Write the video file to disk
    clip.write_videofile(fr'{name}_series.mp4')
