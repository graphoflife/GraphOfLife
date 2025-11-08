from SimOptions import *

from hashlib import sha256

LOADING_FOLDER = r"C:\Users\stefa\OneDrive\Projekte\PythonProjects\Game of Evolution\Graph of Life v013a ++"
SAVE_FOLDER = r"C:\Users\stefa\OneDrive\Projekte\PythonProjects\Game of Evolution\Graph of Life v013a +\unique"

sample_files = [f for f in os.listdir(LOADING_FOLDER) if f.endswith('.csv')]
all_options = []
shas = []
for file in sample_files:
    simulation_options = NewSimOptions()
    simulation_options.load(file, LOADING_FOLDER)
    all_options.append(simulation_options)

unique_options = []
for i, cur_option in enumerate(all_options):

    is_uniques = []
    for cur_unique_option in unique_options:

        is_uniques_cur = []
        for key in cur_unique_option.all_options:
            if cur_option.all_options[key].option == cur_unique_option.all_options[key].option:
                is_uniques_cur.append(True)
            else:
                is_uniques_cur.append(False)


        is_uniques.append(not all(is_uniques_cur))

    if all(is_uniques):
        unique_options.append(cur_option)


for cur_unique_option in unique_options:
    index = all_options.index(cur_unique_option)
    name = str(sample_files[index])
    cur_unique_option.save(filename=name, foldername=SAVE_FOLDER)