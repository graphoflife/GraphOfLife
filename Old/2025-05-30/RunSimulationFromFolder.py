
from Simulation import *
from SimOptions import *


run_options_dict = {}

# Either set to 0.0 or 1.0, decides which options to load
# (it defines if decicions can be made by neighbors in meta evolution)

# String of Folder Path, if not set to None: all options in LOADING_FOLDER will be run after each other
LOADING_FOLDER = rf"C:\Users\stefa\OneDrive\Projekte\PythonProjects\Game of Evolution\Graph of Life v015a"

# What is the probability any single option of all the options get changed
CHANGE_SETTINGS_PROBABILITY = 0.00

# How many Frames should be saved until the Simulation stops
FRAME_AMOUNT = int(1*60*60)

# The path to the output folder, where the images get saved
run_options_dict[RunOptionsEnum.SAVE_FOLDER] = r"C:\Users\stefa\OneDrive\Projekte\PythonProjects\GraphOfLifeCodeSettingCompetition\outputs" #r"D:\GameOfEvolution\outputs"

"""
# The simulation exits if certain criteria are met if this is set to True
run_options_dict[RunOptionsEnum.CAN_EXIT] = True
"""

# Amount of Agents at the beginning of the simulation
run_options_dict[RunOptionsEnum.INIT_AGENT_AMOUNT] = 1

# Amount of tokens range that exist. Determines the max size of Network and max Memory usage
TOKEN_AMOUNT_RANGE = [200, 500]

# Should save images
run_options_dict[RunOptionsEnum.PLOTTING] = True

# Should use random colors and sizes in VisOptions
run_options_dict[RunOptionsEnum.RANDOM_VIS_OPTIONS] = True

# Should the output images have analytic diagrams
run_options_dict[RunOptionsEnum.PLOT_WITH_DIAGRAMS] = False

# Rotation Speed Range of visualization
ANGLE_SHIFT_RANGE = [0.5 * np.pi / 180, 1.0 * np.pi / 180]

# Image amount plotted during each "plot iteration", higher means smoother (but slower)
SMOOTH_PLOTS_RANGE = [10, 20]

# Every X iteration is a "plot iteration"
PLOT_EVERY_X_ITERATIONS_RANGE = [1, 1]

# How many iterations for the spring model when determining the coordinates of the agents during plotting
run_options_dict[RunOptionsEnum.SPRING_MODEL_ITERATIONS] = 10

while True:
    run_options_dict[RunOptionsEnum.TOKEN_AMOUNT] = int(np.random.uniform(TOKEN_AMOUNT_RANGE[0], TOKEN_AMOUNT_RANGE[1]))
    run_options_dict[RunOptionsEnum.ANGLE_SHIFT] = np.random.uniform(ANGLE_SHIFT_RANGE[0], ANGLE_SHIFT_RANGE[1])
    run_options_dict[RunOptionsEnum.SMOOTH_PLOTS] = int(np.random.uniform(SMOOTH_PLOTS_RANGE[0], SMOOTH_PLOTS_RANGE[1]))
    run_options_dict[RunOptionsEnum.PLOT_EVERY_X_ITERATIONS] = int(
        np.random.uniform(PLOT_EVERY_X_ITERATIONS_RANGE[0], PLOT_EVERY_X_ITERATIONS_RANGE[1]))
    run_options_dict[RunOptionsEnum.ITERATION_AMOUNT] = int(
        FRAME_AMOUNT * run_options_dict[RunOptionsEnum.PLOT_EVERY_X_ITERATIONS] / run_options_dict[
            RunOptionsEnum.SMOOTH_PLOTS])

    sample_files = [f for f in os.listdir(LOADING_FOLDER) if f.endswith('.csv')]
    file = np.random.choice(sample_files)
    simulation_options = NewSimOptions()
    simulation_options.load(file, LOADING_FOLDER)
    #simulation_options.all_options[SimOptionsEnum.KEEPS_ONE_TOKEN_WHEN_REPRODUCTION].option = False
    simulation_options.set_random_settings(change_settings_probability=CHANGE_SETTINGS_PROBABILITY)
    simulation = Simulation(sim_options=simulation_options, run_options_dict=run_options_dict)
    simulation.run_main_loop()