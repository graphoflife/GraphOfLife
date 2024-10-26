
from Simulation import *
from SimOptions import *


run_options_dict = {}

# Either set to 0.0 or 1.0, decides which options to load
# (it defines if decicions can be made by neighbors in meta evolution)
# How many Frames should be saved until the Simulation stops
FRAME_AMOUNT = int(0.1*60*30)

# The path to the output folder, where the images get saved
run_options_dict[RunOptionsEnum.SAVE_FOLDER] = r"C:\Users\stefa\OneDrive\Projekte\PythonProjects\GraphOfLifeCodeSettingCompetition\outputs"

# The simulation exits if certain criteria are met if this is set to True
run_options_dict[RunOptionsEnum.CAN_EXIT] = True

# Amount of Agents at the beginning of the simulation
run_options_dict[RunOptionsEnum.INIT_AGENT_AMOUNT] = 50

# TODO
run_options_dict[RunOptionsEnum.KEEP_X_GROUPS] = 10
run_options_dict[RunOptionsEnum.EVERY_X_TOKEN_FOR_NEW_SIMULATION] = 5
run_options_dict[RunOptionsEnum.TOKEN_PER_SIMULATION] = 20


# Amount of tokens range that exist. Determines the max size of Network and max Memory usage
TOKEN_AMOUNT_RANGE = [500, 500]

# Should save images
run_options_dict[RunOptionsEnum.PLOTTING] = True

# Should use random colors and sizes in VisOptions
run_options_dict[RunOptionsEnum.RANDOM_VIS_OPTIONS] = True

# Should the output images have analytic diagrams
run_options_dict[RunOptionsEnum.PLOT_WITH_DIAGRAMS] = True

# Rotation Speed Range of visualization
ANGLE_SHIFT_RANGE = [0.5 * np.pi / 180, 2 * np.pi / 180]

# Image amount plotted during each "plot iteration", higher means smoother (but slower)
SMOOTH_PLOTS_RANGE = [1, 1]

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

    simulation_options = NewSimOptions()
    simulation_options.set_random_settings(change_settings_probability=1.0)
    simulation = Simulation(sim_options=simulation_options,run_options_dict=run_options_dict)

    while True:

        simulation.run_main_loop(redistribute=True)
