from Simulation import *
from SimOptions import *
import datetime as datetime



# Amount of tokens range that exist. Determines the max size of Network and max Memory usage
ALL_TOKEN_AMOUNT_INIT = 25*10*10


def redistribute_tokens_across_simulations(all_simulations, phase = 1):
    fragmentation_groups = []

    tokens_to_distribute = 0

    for cur_sim in all_simulations:
        cur_fragmentations = cur_sim.get_fragmentation()

        cur_fragmentations.sort(key=lambda x: len(x), reverse=True)

        fragmentation_groups.append(cur_fragmentations[0])

        for cur_group in cur_fragmentations[1:]:
            for cur_par in cur_group:
                tokens_to_distribute += cur_par.token
                cur_par.token = 0

    all_particles = []

    for cur_sim in all_simulations:

        np.random.shuffle(cur_sim.particles)

        for cur_par in cur_sim.particles.copy():
            if phase == 1:
                cur_par.check_death_after_phase_1(sim_options=cur_sim.sim_options, all_links=cur_sim.links,
                                    all_particles=cur_sim.particles, data=cur_sim.data,
                                    dead_particles=cur_sim.dead_particles,
                                    dead_links=cur_sim.dead_links)
            else:

                cur_par.check_death(sim_options=cur_sim.sim_options, all_links=cur_sim.links,
                                    all_particles=cur_sim.particles, data=cur_sim.data,
                                    dead_particles=cur_sim.dead_particles,
                                    dead_links=cur_sim.dead_links)

        all_particles += cur_sim.particles

    if tokens_to_distribute > 0:

        for _ in range(int(tokens_to_distribute)):
            chosen_particle = np.random.choice(all_particles)
            chosen_particle.token += 1

    for cur_sim in all_simulations.copy():
        if len(cur_sim.particles) == 0:
            all_simulations.remove(cur_sim)


while True:

    print("Start of New Competition")

    all_simulations = []
    iteration = 0
    TOKEN_PER_SIMULATION = ALL_TOKEN_AMOUNT_INIT/10
    ALL_TOKEN_AMOUNT = ALL_TOKEN_AMOUNT_INIT

    name = f"Competition_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    parent_dir = r"C:\Users\stefa\GameOfEvolution\outputs\Competition"

    path = os.path.join(parent_dir, name)
    os.makedirs(path, exist_ok=True)

    iterations_per_competition = 1000

    for i in range(iterations_per_competition):


        while ALL_TOKEN_AMOUNT >= TOKEN_PER_SIMULATION:



            run_options_dict = {}
            run_options_dict[RunOptionsEnum.SAVE_FOLDER] = path
            run_options_dict[RunOptionsEnum.INIT_AGENT_AMOUNT] = 25
            run_options_dict[RunOptionsEnum.KEEP_X_GROUPS] = 1
            run_options_dict[RunOptionsEnum.PERCENTAGE_OF_TOKENS_FOR_NEW_SIMULATIONS] = 0.00
            run_options_dict[RunOptionsEnum.PLOTTING] = False
            run_options_dict[RunOptionsEnum.RANDOM_VIS_OPTIONS] = True
            run_options_dict[RunOptionsEnum.PLOT_WITH_DIAGRAMS] = False
            run_options_dict[RunOptionsEnum.SPRING_MODEL_ITERATIONS] = 10
            run_options_dict[RunOptionsEnum.TOKEN_AMOUNT] = TOKEN_PER_SIMULATION
            run_options_dict[RunOptionsEnum.ANGLE_SHIFT] = 0.5 * np.pi / 180
            run_options_dict[RunOptionsEnum.SMOOTH_PLOTS] = 1
            run_options_dict[RunOptionsEnum.PLOT_EVERY_X_ITERATIONS] = 10
            run_options_dict[RunOptionsEnum.ITERATION_AMOUNT] = None

            simulation_options = NewSimOptions()

            #simulation_options.load(r"Engineered 4 Probabilistic except Blotto.csv",r"C:\Users\stefa\GameOfEvolution\Promising SimOptions Candidates")
            simulation_options.set_random_settings(change_settings_probability=1.0)
            
            simulation = Simulation(sim_options=simulation_options, run_options_dict=run_options_dict)
            all_simulations.append(simulation)
            ALL_TOKEN_AMOUNT -= TOKEN_PER_SIMULATION

        for cur_sim in all_simulations:
            cur_sim.run_single_iteration_phase1(fragmentation_redistribution=False)

        redistribute_tokens_across_simulations(all_simulations=all_simulations, phase=1)

        for cur_sim in all_simulations:
            cur_sim.run_single_iteration_phase2(fragmentation_redistribution=False)

        redistribute_tokens_across_simulations(all_simulations=all_simulations, phase=2)

        print(f"Next Iteration {i}")
        print("All tokens", ALL_TOKEN_AMOUNT + sum([sum([cur_par.token for cur_par in cur_sim.particles]) for cur_sim in all_simulations]))

        if len(all_simulations) == 1:
            break

            