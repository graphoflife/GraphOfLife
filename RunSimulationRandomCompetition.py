from Simulation import *
from SimOptions import *
import datetime as datetime

# Either set to 0.0 or 1.0, decides which options to load
# (it defines if decicions can be made by neighbors in meta evolution)
# How many Frames should be saved until the Simulation stops

# The path to the output folder, where the images get saved

# The simulation exits if certain criteria are met if this is set to True

# Amount of Agents at the beginning of the simulation

# TODO



# Amount of tokens range that exist. Determines the max size of Network and max Memory usage
ALL_TOKEN_AMOUNT_INIT = 2000

# Should save images


# Should use random colors and sizes in VisOptions


# Should the output images have analytic diagrams


# Rotation Speed Range of visualization
ANGLE_SHIFT_RANGE = [0.5 * np.pi / 180, 2 * np.pi / 180]

# Image amount plotted during each "plot iteration", higher means smoother (but slower)
SMOOTH_PLOTS_RANGE = [1, 1]

# Every X iteration is a "plot iteration"
PLOT_EVERY_X_ITERATIONS_RANGE = [10, 10]

# How many iterations for the spring model when determining the coordinates of the agents during plotting




while True:

    print("Start of New Competition")

    all_simulations = []
    iteration = 0
    TOKEN_PER_SIMULATION = 100
    ALL_TOKEN_AMOUNT = ALL_TOKEN_AMOUNT_INIT

    name = f"Competition_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    parent_dir = r"D:\NewGraphCompetition Outputs\Outputs"

    path = os.path.join(parent_dir, name)
    os.makedirs(path, exist_ok=True)

    iterations_per_competition = 100

    for i in range(iterations_per_competition):


        while ALL_TOKEN_AMOUNT > TOKEN_PER_SIMULATION:



            run_options_dict = {}
            run_options_dict[RunOptionsEnum.SAVE_FOLDER] = path
            run_options_dict[RunOptionsEnum.INIT_AGENT_AMOUNT] = 10
            run_options_dict[RunOptionsEnum.KEEP_X_GROUPS] = 10
            run_options_dict[RunOptionsEnum.PERCENTAGE_OF_TOKENS_FOR_NEW_SIMULATIONS] = 0.35
            run_options_dict[RunOptionsEnum.PLOTTING] = True
            run_options_dict[RunOptionsEnum.RANDOM_VIS_OPTIONS] = True
            run_options_dict[RunOptionsEnum.PLOT_WITH_DIAGRAMS] = True
            run_options_dict[RunOptionsEnum.SPRING_MODEL_ITERATIONS] = 10
            run_options_dict[RunOptionsEnum.TOKEN_AMOUNT] = TOKEN_PER_SIMULATION
            run_options_dict[RunOptionsEnum.ANGLE_SHIFT] = np.random.uniform(ANGLE_SHIFT_RANGE[0], ANGLE_SHIFT_RANGE[1])
            run_options_dict[RunOptionsEnum.SMOOTH_PLOTS] = int(
                np.random.uniform(SMOOTH_PLOTS_RANGE[0], SMOOTH_PLOTS_RANGE[1]))
            run_options_dict[RunOptionsEnum.PLOT_EVERY_X_ITERATIONS] = int(
                np.random.uniform(PLOT_EVERY_X_ITERATIONS_RANGE[0], PLOT_EVERY_X_ITERATIONS_RANGE[1]))
            run_options_dict[RunOptionsEnum.ITERATION_AMOUNT] = 1

            simulation_options = NewSimOptions()
            simulation_options.set_random_settings(change_settings_probability=1.0)
            simulation = Simulation(sim_options=simulation_options, run_options_dict=run_options_dict)
            all_simulations.append(simulation)
            ALL_TOKEN_AMOUNT -= TOKEN_PER_SIMULATION

        for cur_sim in all_simulations:
            cur_sim.run_single_iteration(redistribute=False)


        fragmentation_groups = []

        tokens_to_distribute = 0

        MAX_FRAGMENTATIONS_PER_GROUP = 1


        for cur_sim in all_simulations:
            cur_fragmentations = cur_sim.get_fragmentation()

            cur_fragmentations.sort(key=lambda x: len(x), reverse=True)

            for cur_group in cur_fragmentations[:MAX_FRAGMENTATIONS_PER_GROUP]:

                fragmentation_groups.append(cur_group)

            for cur_group in cur_fragmentations[MAX_FRAGMENTATIONS_PER_GROUP:]:
                for cur_par in cur_group:
                    tokens_to_distribute += cur_par.token
                    cur_par.token = 0

        for cur_sim in all_simulations:

            np.random.shuffle(cur_sim.particles)

            for cur_par in cur_sim.particles.copy():

                cur_par.check_death(sim_options=cur_sim.sim_options, all_links=cur_sim.links,
                                    all_particles=cur_sim.particles, data=cur_sim.data, dead_particles=cur_sim.dead_particles,
                                    dead_links=cur_sim.dead_links)



        np.random.shuffle(fragmentation_groups)
        fragmentation_groups.sort(key=lambda x: len(x), reverse=True)
        print(fragmentation_groups[0])
        print(fragmentation_groups[-1])
        if len(fragmentation_groups) > run_options_dict[RunOptionsEnum.KEEP_X_GROUPS]:
            for cur_group in fragmentation_groups[run_options_dict[RunOptionsEnum.KEEP_X_GROUPS]:]:
                for cur_par in cur_group:
                    tokens_to_distribute += cur_par.token
                    cur_par.token = 0



        MAX_LINK_AMOUNT = 16
        MAX_BEHAVIOR_AGE = 999
        DEATH_PROB = 0.0
        # Check Death
        all_particles = []
        for cur_sim in all_simulations:

            np.random.shuffle(cur_sim.particles)

            for cur_par in cur_sim.particles.copy():
                if np.random.rand() < DEATH_PROB or len(cur_par.nodes) > MAX_LINK_AMOUNT or cur_par.behavior_age > MAX_BEHAVIOR_AGE:
                    tokens_to_distribute += cur_par.token
                    cur_par.token = 0

                cur_par.check_death(sim_options=cur_sim.sim_options, all_links=cur_sim.links,
                                    all_particles=cur_sim.particles, data=cur_sim.data, dead_particles=cur_sim.dead_particles,
                                    dead_links=cur_sim.dead_links)

            all_particles += cur_sim.particles





        if tokens_to_distribute > 0:
            KEEP = max([1, round(tokens_to_distribute*run_options_dict[RunOptionsEnum.PERCENTAGE_OF_TOKENS_FOR_NEW_SIMULATIONS])])
            ALL_TOKEN_AMOUNT += KEEP
            tokens_to_distribute -= KEEP
            p = np.array([cur_par.behavior_age for cur_par in all_particles])
            p = p.max() + 1 - p
            p=p / p.sum()
            for _ in range(int(tokens_to_distribute)):

                chosen_particle = np.random.choice(all_particles)
                chosen_particle.token += 1


        print(f"Next Iteration {iteration}")
        print("All tokens", ALL_TOKEN_AMOUNT + sum([sum([cur_par.token for cur_par in cur_sim.particles]) for cur_sim in all_simulations]))
        iteration += 1

        for cur_sim in all_simulations:
            if sum([cur_par.token for cur_par in cur_sim.particles]) > 0.1*ALL_TOKEN_AMOUNT_INIT:
                cur_sim.run_options_dict[RunOptionsEnum.SMOOTH_PLOTS] = 1
                cur_sim.run_options_dict[RunOptionsEnum.PLOT_EVERY_X_ITERATIONS] = 1
            else:
                cur_sim.run_options_dict[RunOptionsEnum.SMOOTH_PLOTS] = 1
                cur_sim.run_options_dict[RunOptionsEnum.PLOT_EVERY_X_ITERATIONS] = 10

        for cur_sim in all_simulations.copy():
            if len(cur_sim.particles) == 0:
                all_simulations.remove(cur_sim)