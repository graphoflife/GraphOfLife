
from Simulation import *
from SimOptions import *


run_options_dict = {}


# String of Folder Path, if not set to None: all options in LOADING_FOLDER will be run after each other
LOADING_FOLDER = rf"C:\Users\stefa\GameOfEvolution\Promising SimOptions Candidates"
LOADING_FILE = "Engineered 5 Probabilistic except Blotto.csv"


# How many Frames should be saved until the Simulation stops
FRAME_AMOUNT = int(1000*60*60)

# The path to the output folder, where the images get saved
run_options_dict[RunOptionsEnum.SAVE_FOLDER] = r"C:\Users\stefa\GameOfEvolution\outputs\Single1" #r"D:\GameOfEvolution\outputs"



# Amount of Agents at the beginning of the simulation
run_options_dict[RunOptionsEnum.INIT_AGENT_AMOUNT] = 10

# Amount of tokens range that exist. Determines the max size of Network and max Memory usage
run_options_dict[RunOptionsEnum.TOKEN_AMOUNT] = 750

# Should save images
run_options_dict[RunOptionsEnum.PLOTTING] = False

# Should use random colors and sizes in VisOptions
run_options_dict[RunOptionsEnum.RANDOM_VIS_OPTIONS] = False
# Should the output images have analytic diagrams
run_options_dict[RunOptionsEnum.PLOT_WITH_DIAGRAMS] = True

run_options_dict[RunOptionsEnum.KEEP_X_GROUPS] = 1

# Rotation Speed Range of visualization
run_options_dict[RunOptionsEnum.ANGLE_SHIFT]  = 0.5 * np.pi / 180

# Image amount plotted during each "plot iteration", higher means smoother (but slower)
run_options_dict[RunOptionsEnum.SMOOTH_PLOTS] = 1

# Every X iteration is a "plot iteration"
run_options_dict[RunOptionsEnum.PLOT_EVERY_X_ITERATIONS] = 10

# How many iterations for the spring model when determining the coordinates of the agents during plotting
run_options_dict[RunOptionsEnum.SPRING_MODEL_ITERATIONS] = 20

run_options_dict[RunOptionsEnum.ITERATION_AMOUNT] = int(
    FRAME_AMOUNT * run_options_dict[RunOptionsEnum.PLOT_EVERY_X_ITERATIONS] / run_options_dict[
        RunOptionsEnum.SMOOTH_PLOTS])

while True:
    simulation_options = NewSimOptions()
    simulation_options.load(LOADING_FILE, LOADING_FOLDER)
    simulation_options.set_random_settings(change_settings_probability=0.0)
    simulation = Simulation(sim_options=simulation_options, run_options_dict=run_options_dict)
    simulation.run_main_loop(fragmentation_redistribution=True)