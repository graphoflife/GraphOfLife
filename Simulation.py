import numpy as np

from Particle import *
from Link import *
from VisOptions import *
from SimOptions import *
from Data import *
from Behavior import *
import time
import math
import networkx as nx
import copy
from enum import Enum, auto
import datetime as datetime
import matplotlib
import logging
from numbers import Number
matplotlib.use('Agg')

VERSION = "v015a_graph_of_life"


class RunOptionsEnum(Enum):
    TOKEN_PER_SIMULATION = auto()
    PERCENTAGE_OF_TOKENS_FOR_NEW_SIMULATIONS = auto()
    KEEP_X_GROUPS = auto()
    TOKEN_AMOUNT = auto()
    ANGLE_SHIFT = auto()
    SMOOTH_PLOTS = auto()
    PLOT_EVERY_X_ITERATIONS = auto()
    ITERATION_AMOUNT = auto()
    SAVE_FOLDER = auto()
    PLOTTING = auto()
    RANDOM_VIS_OPTIONS = auto()
    INIT_AGENT_AMOUNT = auto()
    CAN_EXIT = auto()
    SPRING_MODEL_ITERATIONS = auto()
    PLOT_WITH_DIAGRAMS = auto()


class Simulation:
    """
    The main simulation class.
    """

    iterations_until_next_plot: int = 0
    current_plot: int = 0
    current_iteration = None

    sim_options_original: NewSimOptions = None
    sim_options: NewSimOptions = None
    vis_options: VisOptions = None
    particles: [Particle] = None
    links: [Link] = None

    name: str = None
    data = None

    # For Plotting
    g = None
    angle = None
    pos = None
    vel = None
    vis_max_node_color_value = None
    vis_max_edge_color_value = None

    dead_links = None
    dead_particles = None


    # For RPS game
    win_lose_matrix = None

    def __init__(self, sim_options, run_options_dict:dict):
        """
        Initialization of the simulation. Create neccesary folders. Create initial particles and links.
        Create evaluation matrix for the game evaluation.
        :param sim_options: Options of Simulation
        """

        self.run_options_dict = run_options_dict

        self.particles = []
        self.links = []
        self.sim_options_original = sim_options
        self.sim_options = copy.deepcopy(self.sim_options_original)
        self.data = Data()
        self.current_iteration = 0
        self.angle = 0
        self.cur_behavior_index = 0

        self.dead_links = []
        self.dead_particles = []

        self.node_sizes_array = None
        self.edge_sizes_array = None
        self.node_colors_array = None
        self.edge_colors_array = None

        self.vis_max_node_color_value = 1.0
        self.vis_max_edge_color_value = 1.0

        # Create Folder and Save Simulation Options
        self.name = f"{VERSION}_"+ datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        parent_dir = os.path.dirname(os.path.realpath(__file__))
        output_dir = os.path.join(parent_dir, "outputs")
        every_option_dir = os.path.join(parent_dir, "every_option")

        path = os.path.join(self.run_options_dict[RunOptionsEnum.SAVE_FOLDER], self.name)

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(every_option_dir, exist_ok=True)
        if self.run_options_dict[RunOptionsEnum.PLOTTING]:
            os.makedirs(path, exist_ok=True)
            self.sim_options.save(self.name, path)
            log_path = os.path.join(path, "log")

            handler = logging.FileHandler(log_path, mode='a')  # append mode
            formatter = logging.Formatter('%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s', '%H:%M:%S')
            handler.setFormatter(formatter)

            logger = logging.getLogger(log_path)
            logger.setLevel(logging.DEBUG)
            logger.addHandler(handler)

            # This prevents the log messages from being duplicated in the python output
            logger.propagate = False
            self.logger = logger
            self.logger.warning("init")

        self.sim_options.save(self.name, every_option_dir)

        self.vis_options = VisOptions()
        if self.run_options_dict[RunOptionsEnum.RANDOM_VIS_OPTIONS]:
            self.vis_options.randomize()

        self.iterations_until_next_plot = self.run_options_dict[RunOptionsEnum.PLOT_EVERY_X_ITERATIONS] + 1



        print("Iterations Amount:", self.run_options_dict[RunOptionsEnum.ITERATION_AMOUNT])

        # Particle and Link Initialization
        init_token_per_particle = int(self.run_options_dict[RunOptionsEnum.TOKEN_AMOUNT]/self.run_options_dict[RunOptionsEnum.INIT_AGENT_AMOUNT])
        for i in range(self.run_options_dict[RunOptionsEnum.INIT_AGENT_AMOUNT]):
            new_behavior = Behavior(sim_options=sim_options, indexer_f=self.index_behavior, data=self.data)
            self.particles.append(Particle(token=init_token_per_particle, behavior=new_behavior))
        if self.run_options_dict[RunOptionsEnum.INIT_AGENT_AMOUNT] == 2:
            self.links.append(Link(particle1=self.particles[0], particle2=self.particles[1]))
        elif self.run_options_dict[RunOptionsEnum.INIT_AGENT_AMOUNT] > 2:

            for i in range(len(self.particles)-1):
                for j in range(i+1, len(self.particles)):
                    self.links.append(Link(particle1=self.particles[i], particle2=self.particles[j]))
            print(len(self.links), len(self.particles)*(len(self.particles) - 1)/2)


    def kill_duplicate_links(self, sim_options, all_links, dead_links):
        for cur_par in self.particles:
            already_connected = []
            for cur_node in cur_par.nodes.copy():
                if cur_node.other_node.particle is not None:
                    if cur_node.other_node.particle in already_connected or cur_node.other_node.particle is cur_par:
                        cur_node.link.kill_link(sim_options, all_links, dead_links)
                    else:
                        already_connected.append(cur_node.other_node.particle)

    def run_single_iteration(self, redistribute = False):

        self.sim_options = copy.deepcopy(self.sim_options_original)

        if len(self.particles) == 0:
            return


        # Observation Iteration Preparation
        self.prepare()
        for cur_link in self.links:
            cur_link.prepare(sim_options=self.sim_options)
        for cur_par in self.particles:
            cur_par.prepare(sim_options=self.sim_options)

        # Reproduction and Geometric Phase

        # Observation Phase 1
        for cur_par in self.particles:
            cur_par.process_information_and_make_decisions_repro_geo_phase(sim_options=self.sim_options, data=self.data,
                                                                           all_particles=self.particles)

        """
        result_repro = 0
        result_repro_percentage = 0
        for cur_par in self.particles:
            result_repro += cur_par.reproduction_tokens
            if cur_par.reproduction_tokens > 0:
                result_repro_percentage += 1
        if result_repro != self.data.kept_repro_tokens_history[-1]:
            self.logger.warning("Shouldnt happen 1")
        """

        # Swap
        for cur_link in self.links:
            cur_link.try_swap(sim_options=self.sim_options, data=self.data)

        # Reconnection
        if self.sim_options.get(SimOptionsEnum.CAN_RECONNECT):

            for cur_par in self.particles:
                cur_par.do_reconnect(data=self.data)

            self.kill_duplicate_links(sim_options=self.sim_options, all_links=self.links, dead_links=self.dead_links)

        # New_link
        for cur_par in self.particles:
            cur_par.try_making_new_link(sim_options=self.sim_options, all_links=self.links, data=self.data,
                                        pos=self.pos)

        # Move Walker
        for cur_par in self.particles:
            cur_par.move()


        # Reproduction
        for cur_par in self.particles.copy():
            cur_par.reproduce(sim_options=self.sim_options, pos=self.pos,
                              all_particles=self.particles, all_links=self.links, data=self.data,
                              vel=self.vel, indexer_f=self.index_behavior, logger=self.logger)


        """
        if result_repro_percentage != self.data.planted_particles_history[-1] + self.data.reproduced_particles_history[-1]:
            self.logger.warning("Shouldnt happen 2")
        """

        # Game Phase
        self.prepare_game_phase()
        for cur_par in self.particles:
            cur_par.prepare_information(sim_options=self.sim_options)

        # Observation Phase 2
        for cur_par in self.particles:
            cur_par.process_information_and_make_decisions_game_phase(sim_options=self.sim_options, data=self.data,
                                                                      all_particles=self.particles)



        # Eval Game
        for cur_par in self.particles:
            cur_par.eval_game(sim_options=self.sim_options, data=self.data, indexer_f=self.index_behavior)


        # Apply Behavior
        for cur_par in self.particles:
            cur_par.apply_new_behavior(sim_options=self.sim_options, data=self.data, indexer_f=self.index_behavior)

        # TODO makes probably no sense because they have a disadvantage
        # Check Activity
        for cur_link in self.links:
            if cur_link.active_value > 0:
                cur_link.is_active = True
            else:
                cur_link.is_active = False

        # Check Death of Links
        if self.sim_options.get(SimOptionsEnum.KILL_INACTIVE_LINKS):
            for cur_link in self.links:
                cur_link.check_inactivity(sim_options=self.sim_options, all_links=self.links,
                                          data=self.data, dead_links=self.dead_links)


        # self.kill_duplicate_links(sim_options=self.sim_options,all_links=self.links,dead_links=self.dead_links)
        # Check Death
        for cur_par in self.particles.copy():
            cur_par.check_death(sim_options=self.sim_options, all_links=self.links,
                                all_particles=self.particles, data=self.data, dead_particles=self.dead_particles,
                                dead_links=self.dead_links)


        # 8 Mutate
        mutate_all_each_iteration = self.sim_options.get(SimOptionsEnum.MUTATE_ALL_EACH_ITERATION)
        for cur_par in self.particles:
            if cur_par.to_mutate or mutate_all_each_iteration:
                cur_par.behavior.mutate(sim_options=self.sim_options, vis_options=self.vis_options)

        if redistribute:
            # Redistribute
            fragmentation_groups = self.get_fragmentation()
            fragmentation_groups.sort(key=lambda x: len(x), reverse=True)
            tokens_to_distribute = 0
            if len(fragmentation_groups) > self.run_options_dict[RunOptionsEnum.KEEP_X_GROUPS]:
                for cur_group in fragmentation_groups[self.run_options_dict[RunOptionsEnum.KEEP_X_GROUPS]:]:
                    for cur_par in cur_group:
                        tokens_to_distribute += cur_par.token
                        cur_par.token = 0

            # Check Death
            for cur_par in self.particles.copy():
                cur_par.check_death(sim_options=self.sim_options, all_links=self.links,
                                    all_particles=self.particles, data=self.data, dead_particles=self.dead_particles,
                                    dead_links=self.dead_links)

            for _ in range(int(tokens_to_distribute)):

                chosen_particle = np.random.choice(self.particles)
                chosen_particle.token += 1

            # redistribution end


        self.data.sort_birth_layer()

        # Plotting
        token_amount = sum([cur_par.token for cur_par in self.particles])
        self.data.renormalize(token_amount=token_amount)

        """
        if self.sim_options.get(SimOptionsEnum.META_EVOLUTION_DEC) or self.sim_options.get(SimOptionsEnum.META_EVOLUTION_USE_X_BEH) :
            result = {}
            for cur_par in self.particles:
                for cur_beh in cur_par.behavior.meta_options.keys():
                    if cur_beh not in result.keys():
                        result[cur_beh] = {}
                    option = cur_par.behavior.meta_options[cur_beh]
                    if option not in result[cur_beh].keys():
                        result[cur_beh][option] = 1
                    else:
                        result[cur_beh][option] += 1
            for cur_key in result:
                result[cur_key] = {key: result[cur_key][key] for key in sorted(result[cur_key])}

            print(result)
        """

        if self.run_options_dict[RunOptionsEnum.PLOTTING]:
            self.plot_network()

        """            
        crit1 = len(self.particles) < 10 and self.current_iteration > 20
        crit12 = len(self.particles) < 5
        if crit1 or crit12:
            break

        if self.run_options_dict[RunOptionsEnum.CAN_EXIT]:
            hastocrit = self.current_iteration > 30
            crit2 = all(np.array(self.data.shannon_entropy[-10:]) < 0.25) and hastocrit
            crit4 = False # all(np.array(self.data.died_particles_history[-10:]) < 0.01) and hastocrit
            crit5 = False # all(np.array(self.data.reproduced_particles_history[-5:]) + np.array(self.data.planted_particles_history[-5:]) < 0.01) and hastocrit
            crit6 = all(np.array(self.data.links_per_particle_history[-5:]) < 1.01) and hastocrit
            crit9 = all(np.array(self.data.links_per_particle_history[-5:]) > 5) and hastocrit
            crit10 = all(np.array(self.data.token_other_invested_history[-5:]) < 0.01) and hastocrit
            link_amounts = [cur_par.link_amount() for cur_par in self.particles]
            self.data.link_amount_crit_array.append(max(link_amounts) > 30*np.quantile(link_amounts, 0.9))
            self.data.link_amount_crit_array2.append(max(link_amounts) > 250)

            crit11 = all(self.data.link_amount_crit_array[-5:])
            crit12 = all(self.data.link_amount_crit_array2[-5:])
            if crit11:
                print("max link amount too large", self.name)
            if crit12:
                print("max link amount too large > 250", self.name)
            if crit1 or crit2 or crit4 or crit5 or crit6 or crit9 or crit10:

                break

        """

        self.current_iteration += 1

    def run_main_loop(self, redistribute = False):
        """
        The main loop of the simulation.
        :param iterations: The amount of iterations that are executed
        """

        # Main loop
        iterations = self.run_options_dict[RunOptionsEnum.ITERATION_AMOUNT]
        print("total iterations:", iterations)

        print("meta_evo, use_x_beh, dec", self.sim_options.get(SimOptionsEnum.META_EVOLUTION_USE_X_BEH), self.sim_options.get(SimOptionsEnum.META_EVOLUTION_DEC))

        self.data.sort_birth_layer()

        while iterations > 0:
            if len(self.particles) == 0:
                break
            self.run_single_iteration(redistribute)
            iterations -= 1

            
    def index_behavior(self):
        self.cur_behavior_index += 1
        return self.cur_behavior_index

    def shannon_entropy(self, G):
        vk = dict(G.degree())
        vk = list(vk.values())  # we get only the degree values
        maxk = np.max(vk)
        # mink = np.min(min)
        kvalues = np.arange(0, maxk + 1)  # possible values of k
        Pk = np.zeros(maxk + 1)  # P(k)
        for k in vk:
            Pk[k] = Pk[k] + 1
        Pk = Pk / sum(Pk)  # the sum of the elements of P(k) must to be equal to one
        H = 0
        for p in Pk:
            if (p > 0):
                H = H - p * math.log(p, 2)
        return H


    def prepare_game_phase(self):

        self.data.particle_amount_history_after_phase1.append(len(self.particles))

        self.data.link_amount_history_after_phase1.append(len(self.links))
    def prepare(self):
        """
        Prepare iteration information for data analysis
        """

        # Randomize the order of execution so that no systemic advantages arise
        np.random.shuffle(self.particles)
        np.random.shuffle(self.links)

        self.dead_links = []
        self.dead_particles = []

        self.data.prepare(particles=self.particles, links=self.links)







    def plot_birth_tree(self, ax, cur_smooth_plot):
        ax.set_ylim([0.0, 1.0])
        smooth_plots = self.run_options_dict[RunOptionsEnum.SMOOTH_PLOTS]
        offset = (1+cur_smooth_plot)/smooth_plots

        G = nx.Graph()
        pos = {}
        color = []
        edge_colors = []

        plot_how_many = 150
        self.data.birth_tree = self.data.birth_tree[-plot_how_many:]
        if len(self.data.birth_tree) == plot_how_many:
            start_offset = offset
        else:
            start_offset = 0.0
        ax.set_xlim([start_offset, len(self.data.birth_tree)-1.9999  + offset])

        for i, cur_birth_layer in enumerate(self.data.birth_tree):
            particle_amount = len(cur_birth_layer)
            dx = 1.0
            dy = 1.0/particle_amount
            if i > 0:
                indexes_of_last_layer = [cur_par_entry[0] for cur_par_entry in self.data.birth_tree[i-1]]
            indexes_of_this_layer = [cur_par_entry[0] for cur_par_entry in self.data.birth_tree[i]]
            for j, cur_par_entry in enumerate(cur_birth_layer):
                x = i*dx
                y = j*dy
                G.add_node(cur_par_entry[0])
                pos[cur_par_entry[0]] = np.array([x, y])
                cur_color = np.maximum(np.minimum(cur_par_entry[4], 0.999999), 0.00000001)
                color.append(cur_color)

                if i > 0:
                    # has parent: plot edge
                    if cur_par_entry[1] in indexes_of_last_layer:
                        G.add_edge(cur_par_entry[1], cur_par_entry[0])
                        edge_colors.append(cur_color)

                    else:
                        if cur_par_entry[2] in indexes_of_last_layer:
                            G.add_edge(cur_par_entry[2], cur_par_entry[0])
                            edge_colors.append(cur_color)

                        else:
                            if cur_par_entry[3] in indexes_of_last_layer:
                                G.add_edge(cur_par_entry[3], cur_par_entry[0])
                                edge_colors.append(cur_color)

                            else:
                                print("why")
                                if cur_par_entry[1] in indexes_of_this_layer:
                                    print("why 1")
                                if cur_par_entry[2] in indexes_of_this_layer:
                                    print("why 2")
                                if cur_par_entry[3] in indexes_of_this_layer:
                                    print("why 3")

        options = {
            'node_shape': ".",
            'node_size': 0,
            'node_color': color,
            'linewidths': 0.0,
            'with_labels': False,

            #'connectionstyle': "arc3,rad=0.1",
            #'arrowstyle': "-",
            'width': 3,
            'edge_color': edge_colors
        }

        nx.draw_networkx(G=G, pos=pos, ax=ax, **options)

    def get_fragmentation(self):

        assigned_particles = []
        fragmentation_groups = []

        for cur_par in self.particles:
            if cur_par not in assigned_particles:

                particles_reached = [cur_par]
                for cur_node in cur_par.nodes:
                    particles_reached.append(cur_node.other_node.particle)
                reached = 0
                while len(particles_reached) > reached:
                    reached = len(particles_reached)
                    for cur_par1 in particles_reached.copy():
                        for cur_node in cur_par1.nodes:
                            particles_reached.append(cur_node.other_node.particle)
                    particles_reached = list(set(particles_reached))

                assigned_particles += particles_reached
                assigned_particles = list(set(assigned_particles))
                fragmentation_groups.append(particles_reached)

        return fragmentation_groups


    def set_test(self, counter):
        # debug Test
        par_set = []
        link_set = []
        for cur_par in self.particles:
            par_set.append(cur_par)
        for cur_link in self.links:
            link_set.append(cur_link.node1.particle)
            link_set.append(cur_link.node2.particle)



    def plot_network(self):
        """
        Plots the network and data, saves images into the outputs folder.
        :param background_color: "black" or "white"
        """
        token_amount_particles = sum([cur_par.token for cur_par in self.particles])
        print("Tokens", token_amount_particles,
              ", Particles", len(self.particles),
              ", Links", len(self.links),
              ", Iteration", self.current_iteration, self.name)

        self.g = nx.Graph()
        for cur_par in self.particles:
            self.g.add_node(cur_par)
        for cur_link in self.links:
            self.g.add_edge(cur_link.node1.particle, cur_link.node2.particle, weight=1)


        tokens = [cur_par.token for cur_par in self.particles]
        self.data.max_token_track.append(max(tokens))
        self.data.max_attack_track.append(max(self.data.attacked_with_x_tokens))
        tokens_var = float(np.var(tokens))
        particles_at_position = [len(cur_par.walker_position.particles_at_this_position) for cur_par in
                                 self.particles]
        particles_at_position_var = float(np.var(particles_at_position))
        link_amounts = [cur_par.link_amount() for cur_par in self.particles]
        link_amounts_var = float(np.var(link_amounts))
        beh_ages = [cur_par.behavior_age for cur_par in self.particles]
        beh_ages_var = float(np.var(beh_ages))

        ages = [cur_par.age for cur_par in self.particles]
        ages_var = float(np.var(ages))
        link_ages = [cur_link.age for cur_link in self.links]
        link_ages_var = float(np.var(link_ages))

        self.data.shannon_entropy.append(self.shannon_entropy(self.g))

        # Dimensionality calculation
        dims = []
        dims_max = []
        dims_mean = []
        dims_amounts = []
        for cur_par in np.random.choice(self.particles, 30):
            cur_dims = cur_par.get_dimension_list(all_particles=self.particles)
            if len(cur_dims) > 0:
                dims_max.append(np.max(cur_dims))
                dims_mean.append(np.mean(cur_dims))
                dims_amounts.append(len(cur_dims))
                dims.append(cur_dims)
        mean_steps = np.mean([len(cur_dim) for cur_dim in dims])
        self.data.mean_steps_history.append(mean_steps)
        self.data.dims = dims
        self.data.dim_max_history.append(np.average(dims_max))
        self.data.dim_mean_history.append(np.average(dims_mean))

        self.data.dim_max_stddev_history.append(np.std(dims_max))
        self.data.dim_mean_stddev_history.append(np.std(dims_mean))

        dead_par_pos = []
        dead_par_vel = []
        dead_par_temp = []
        if self.pos is not None:
            for cur_par in self.dead_particles:
                if cur_par in self.pos:
                    dead_par_pos.append(self.pos[cur_par])
                    dead_par_temp.append(cur_par)
                    dead_par_vel.append(self.vel[cur_par])
        self.dead_particles = dead_par_temp

        if self.pos is not None:

            temp_pos = {}
            for cur_par in self.particles:
                temp_pos[cur_par] = self.pos[cur_par]
            self.pos = temp_pos

            temp_vel = {}
            for cur_par in self.particles:
                temp_vel[cur_par] = self.vel[cur_par]
            self.vel = temp_vel

        self.iterations_until_next_plot -= 1
        if self.iterations_until_next_plot <= 0:    # OR first plot TODO
            self.iterations_until_next_plot = self.run_options_dict[RunOptionsEnum.PLOT_EVERY_X_ITERATIONS]
        else:
            return



        if self.pos is None:
            self.pos = nx.fruchterman_reingold_layout(self.g, pos=self.pos, scale=1, iterations=100, dim=3)
            self.vel = {}
            for cur_par in self.particles:
                self.vel[cur_par] = np.zeros(3)

            for cur_link in self.links:
                cur_link.node1.vis_pos = self.pos[cur_link.node1.particle]
                cur_link.node1.vis_vel = self.vel[cur_link.node1.particle]

                cur_link.node2.vis_pos = self.pos[cur_link.node2.particle]
                cur_link.node2.vis_vel = self.vel[cur_link.node2.particle]
        else:
            for cur_link in self.links:
                if cur_link.node1.vis_pos is None:
                    cur_link.node1.vis_pos = self.pos[cur_link.node1.particle]
                cur_link.node1.vis_vel = self.vel[cur_link.node1.particle]

                if cur_link.node2.vis_pos is None:
                    cur_link.node2.vis_pos = self.pos[cur_link.node2.particle]
                cur_link.node2.vis_vel = self.vel[cur_link.node2.particle]
        for cur_link in self.dead_links:
            if cur_link.node1.vis_pos is None:
                cur_link.node1.vis_pos = np.zeros(3)
            if cur_link.node1.vis_vel is None:
                cur_link.node1.vis_vel = np.zeros(3)

            if cur_link.node2.vis_pos is None:
                cur_link.node2.vis_pos = np.zeros(3)
            if cur_link.node2.vis_vel is None:
                cur_link.node2.vis_vel = np.zeros(3)

        new_pos = nx.fruchterman_reingold_layout(self.g, pos=self.pos, scale=1, iterations=self.run_options_dict[RunOptionsEnum.SPRING_MODEL_ITERATIONS], dim=3)

        SMOOTH_PLOTS = self.run_options_dict[RunOptionsEnum.SMOOTH_PLOTS]

        for i in range(SMOOTH_PLOTS):
            projected_pos = {}
            depth = {}
            self.angle += self.run_options_dict[RunOptionsEnum.ANGLE_SHIFT]
            c, s = np.cos(self.angle), np.sin(self.angle)
            rotation_matrix = np.array([[c, 0, -s], [0, 1.0, 0], [s, 0, c]])

            for cur_par in self.particles:
                difference = (new_pos[cur_par] - self.pos[cur_par])/(SMOOTH_PLOTS-i)
                diff_per = ((i+1)/SMOOTH_PLOTS)
                vel_per = 1 - diff_per
                vel = diff_per*difference + vel_per*self.vel[cur_par]
                self.vel[cur_par] = vel
                self.pos[cur_par] = self.pos[cur_par] + vel
                new_coord = np.matmul(rotation_matrix, self.pos[cur_par])
                projected_pos[cur_par] = new_coord[0:2]
                depth[cur_par] = new_coord[-1]

            for cur_link in self.links:
                # node1
                cur_par = cur_link.node1.particle
                difference = (new_pos[cur_par] - cur_link.node1.vis_pos)/(SMOOTH_PLOTS-i)
                diff_per = ((i+1)/SMOOTH_PLOTS)
                vel_per = 1 - diff_per
                vel = diff_per*difference + vel_per*cur_link.node1.vis_vel
                cur_link.node1.vis_vel = vel
                cur_link.node1.vis_pos = cur_link.node1.vis_pos + vel
                # node2
                cur_par = cur_link.node2.particle
                difference = (new_pos[cur_par] - cur_link.node2.vis_pos) / (SMOOTH_PLOTS - i)
                diff_per = ((i + 1) / SMOOTH_PLOTS)
                vel_per = 1 - diff_per
                vel = diff_per * difference + vel_per * cur_link.node2.vis_vel
                cur_link.node2.vis_vel = vel
                cur_link.node2.vis_pos = cur_link.node2.vis_pos + vel

            for cur_link in self.dead_links:
                # node1
                cur_link.node1.vis_pos = cur_link.node1.vis_pos + cur_link.node1.vis_vel/2
                # node2
                cur_link.node2.vis_pos = cur_link.node2.vis_pos + cur_link.node2.vis_vel/2

            for j in range(len(self.dead_particles)):
                dead_par_pos[j] += dead_par_vel[j]/2

            lim = 1.0
            edge_depth = []
            edge_sizes = []
            max_size = self.vis_options.max_size_edge
            min_size = self.vis_options.min_size_edge
            for cur_link in self.links:
                cur_depth = (cur_link.node1.vis_pos[-1] + cur_link.node2.vis_pos[-1])/2
                cur_depth = min(max(cur_depth, -lim), lim)
                cur_size = min_size + (max_size-min_size)*(cur_depth+lim)/(2*lim)
                edge_depth.append(cur_depth)
                edge_sizes.append(cur_size)

            dead_edge_depth = []
            dead_edge_sizes = []
            dead_link_alpha_values = []
            old_dead_link_tokens = []
            for cur_link in self.dead_links:
                cur_depth = (cur_link.node1.vis_pos[-1] + cur_link.node2.vis_pos[-1])/2
                cur_depth = min(max(cur_depth, -lim), lim)
                cur_size = min_size + (max_size-min_size)*(cur_depth+lim)/(2*lim)
                dead_edge_depth.append(cur_depth)
                dead_edge_sizes.append(cur_size)
                dead_link_alpha_values.append( 1 - (i + 1) / SMOOTH_PLOTS)
                old_dead_link_tokens.append(cur_link.vis_max_color_value)

            node_depth = []
            node_sizes = []
            max_size = self.vis_options.max_size_node
            min_size = self.vis_options.min_size_node
            for curPar in self.particles:
                cur_depth = depth[curPar]
                cur_depth = min(max(cur_depth, -lim), lim)
                cur_size = min_size + (max_size-min_size)*(cur_depth+lim)/(2*lim)
                node_depth.append(cur_depth)
                node_sizes.append(cur_size)

            dead_node_sizes = []
            dead_node_alpha_values = []
            old_dead_particle_tokens = []
            for j in range(len(self.dead_particles)):
                cur_depth = dead_par_pos[j][-1]
                cur_depth = min(max(cur_depth, -lim), lim)
                cur_size = min_size + (max_size-min_size)*(cur_depth+lim)/(2*lim)
                dead_node_sizes.append(cur_size)
                dead_node_alpha_values.append(1 - (i + 1) / SMOOTH_PLOTS)
                old_dead_particle_tokens.append(self.dead_particles[j].vis_max_color_value)

            sorted_edges = sorted(self.g.edges, key=lambda edge: edge_depth[list(self.g.edges).index(edge)])
            sorted_nodes = sorted(self.g.nodes, key=lambda node: node_depth[list(self.g.nodes).index(node)])
            node_depth = sorted(node_depth)
            edge_depth = sorted(edge_depth)
            edge_sizes = sorted(edge_sizes)
            node_sizes = sorted(node_sizes)

            g_plot = nx.DiGraph()
            for cur_par in sorted_nodes:
                g_plot.add_node(cur_par)
            for cur_edge in sorted_edges:
                g_plot.add_edge(cur_edge[0], cur_edge[1], weight=1)

            particle_tokens = np.array([cur_node.token for cur_node in g_plot.nodes])
            old_particle_tokens = []
            for cur_par in g_plot.nodes:
                old_particle_tokens.append(cur_par.vis_max_color_value)


            link_ages = []
            link_tokens = []
            old_link_tokens = []
            link_alpha_values = []
            for cur_edge in g_plot.edges:
                for cur_node in cur_edge[0].nodes:
                    if cur_node.other_node.particle is cur_edge[1]:
                        link_ages.append(cur_node.link.age)
                        link_tokens.append(cur_node.link.active_value)
                        old_link_tokens.append(cur_node.link.vis_max_color_value)
                        if cur_node.link.is_new_link:
                            link_alpha_values.append((i + 1) / SMOOTH_PLOTS)
                        else:
                            link_alpha_values.append(1.0)

                        break
            link_tokens = np.array(link_tokens)
            old_link_tokens = np.array(old_link_tokens)

            # Node Color

            vmin = 1  # Your vmin
            max_value = max(particle_tokens)
            vmax = self.vis_max_node_color_value + (max_value - self.vis_max_node_color_value) * (i + 1) / SMOOTH_PLOTS
            if i == SMOOTH_PLOTS - 1:
                self.vis_max_node_color_value = max_value

            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            interpolated_node_tokens = old_particle_tokens + (particle_tokens - old_particle_tokens) * (
                        i + 1) / SMOOTH_PLOTS
            normalized_tokens = norm(interpolated_node_tokens)
            normalized_dead_tokens = norm(old_dead_particle_tokens)
            rgb_node_colors = self.vis_options.cmap(normalized_tokens)
            rgb_dead_node_colors = self.vis_options.cmap(normalized_dead_tokens)

            rgba_dead_node_colors = [(r, g, b, alpha) for (r, g, b, _), alpha in
                                     zip(rgb_dead_node_colors, dead_node_alpha_values)]

            # Edge Color
            vmin = 0  # Your vmin
            if len(link_tokens) > 0:
                max_value = max(link_tokens)
            else:
                max_value = 1
            vmax = self.vis_max_edge_color_value + (max_value - self.vis_max_edge_color_value) * (i + 1) / SMOOTH_PLOTS
            if i == SMOOTH_PLOTS - 1:
                self.vis_max_edge_color_value = max_value
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            interpolated_link_tokens = old_link_tokens + (link_tokens - old_link_tokens) * (i + 1) / SMOOTH_PLOTS
            normalized_tokens = norm(interpolated_link_tokens)
            normalized_dead_tokens = norm(old_dead_link_tokens)

            rgb_colors = self.vis_options.cmap_edge(normalized_tokens)
            rgb_dead_colors = self.vis_options.cmap_edge(normalized_dead_tokens)
            rgba_edge_colors = [(r, g, b, alpha) for (r, g, b, _), alpha in zip(rgb_colors, link_alpha_values)]

            rgba_dead_edge_colors = [(r, g, b, alpha) for (r, g, b, _), alpha in
                                     zip(rgb_dead_colors, dead_link_alpha_values)]

            options_node = {
                'node_shape': ".",
                'node_size': node_sizes,
                'node_color': rgb_node_colors,
                'linewidths': 0.0,
            }

            options_dead_node = {
                'node_shape': ".",
                'node_size': dead_node_sizes,
                'node_color': rgba_dead_node_colors,
                'linewidths': 0.0,

            }

            options_edge = {
                'node_size': node_sizes,
                'connectionstyle': "arc3,rad=0.1",
                'arrowstyle': "-",
                'width': edge_sizes,
                'edge_color': rgba_edge_colors

            }
            ref_node_size = self.vis_options.max_size_node
            options_dead_edge = {
                'node_size': ref_node_size,
                'connectionstyle': "arc3,rad=0.1",
                'arrowstyle': "-",
                'width': dead_edge_sizes,
                'edge_color': rgba_dead_edge_colors
            }

            if self.run_options_dict[RunOptionsEnum.PLOT_WITH_DIAGRAMS]:
                res_x = 16*160
                ratio = 2.0
                plt.figure(figsize=(16*ratio, 9*ratio), dpi=res_x/(16*ratio))
                if self.vis_options.backgroundcolor[0] < 0.5:
                    plt.style.use('dark_background')
                    linecolor = "white"
                    edgecolor = "black"
                else:
                    plt.style.use('default')
                    linecolor = "black"
                    edgecolor = "white"

                col_amount_main = 4

                row_amount_tot = 9
                col_amount_diagrams = 3

                col_amount_tot = col_amount_main + col_amount_diagrams

                ax1 = plt.subplot2grid((row_amount_tot, col_amount_tot), (0, 0), colspan=col_amount_main, rowspan=row_amount_tot-2)
                ax2 = plt.subplot2grid((row_amount_tot, col_amount_tot), (row_amount_tot-2, 0), colspan=col_amount_main, rowspan=2)

                axs = []

                for jndex in range(col_amount_main, col_amount_tot):
                    for index in range(row_amount_tot):
                        axs.append(plt.subplot2grid((row_amount_tot, col_amount_tot), (index, jndex), colspan=1, rowspan=1))
                ax_index = 0

                plt.subplots_adjust(wspace=0.35, hspace=0.35)
                ax1.set_title(f"Game of Evolution, {self.name}, AT={token_amount_particles}, "
                              f"{self.sim_options.get_description()}")

                self.plot_birth_tree(ax=ax2, cur_smooth_plot=i)

                backlash = 0.96

                self.data.max_token_amount_history = max(int(self.data.max_token_amount_history * backlash), max(tokens))
                bins = np.linspace(0.0, max(tokens), 30)
                axs[ax_index].hist(tokens, rwidth=1, bins=bins, color=linecolor, edgecolor=edgecolor)
                axs[ax_index].set_title(f"Particle Tokens, var={round(tokens_var, 3)}")
                # axs[ax_index].set_yscale('log')
                ax_index += 1

                self.data.max_par_conc_history = max(int(self.data.max_par_conc_history * backlash),
                                                     max(particles_at_position))
                bins = np.linspace(0.0, max(particles_at_position), 30)
                axs[ax_index].hist(particles_at_position, rwidth=1, bins=bins, color=linecolor, edgecolor=edgecolor)
                axs[ax_index].set_title(f"Particle Concentration, var={round(particles_at_position_var, 3)}")
                # axs[ax_index].set_yscale('log')
                self.data.max_link_amount_history = max(int(self.data.max_link_amount_history * backlash), max(link_amounts))
                ax_index += 1

                bins = np.linspace(0.0, self.data.max_link_amount_history, 30)
                axs[ax_index].hist(link_amounts, rwidth=1, bins=bins, color=linecolor, edgecolor=edgecolor)
                axs[ax_index].set_title(f"Link Amount Distribution, var={round(link_amounts_var, 3)}")
                # axs[ax_index].set_yscale('log')
                ax_index += 1

                self.data.max_age_history = max(int(self.data.max_age_history * backlash), max(ages))
                bins = np.linspace(0.0, max(ages), 30)
                axs[ax_index].hist(ages, rwidth=1, bins=bins, color=linecolor, edgecolor=edgecolor)
                axs[ax_index].set_title(f"Particle Age Distribution, var={round(ages_var, 3)}")
                # axs[ax_index].set_yscale('log')
                ax_index += 1

                self.data.max_beh_age_history = max(int(self.data.max_beh_age_history * backlash), max(beh_ages))
                bins = np.linspace(0.0, max(beh_ages), 30)
                axs[ax_index].hist(beh_ages, rwidth=1, bins=bins, color=linecolor, edgecolor=edgecolor)
                axs[ax_index].set_title(f"Particle Behavior Age Distribution, var={round(beh_ages_var, 3)}")
                # axs[ax_index].set_yscale('log')
                ax_index += 1

                """
                self.data.max_link_age_history = max(int(self.data.max_link_age_history * backlash), max(link_ages))
                bins = np.linspace(0.0, self.data.max_link_age_history, 30)
                axs[ax_index].hist(link_ages, rwidth=1, bins=bins, color=linecolor, edgecolor=edgecolor)
                axs[ax_index].set_title(f"Link Age Distribution, var={round(link_ages_var, 3)}")
                axs[ax_index].set_yscale('log')
                ax_index += 1
                """

                if len(link_tokens) > 0:
                    temp_max_value =  max(link_tokens)
                else:
                    temp_max_value = 1

                self.data.max_link_active_value_history = max(int(self.data.max_link_active_value_history * backlash), temp_max_value)
                bins = np.linspace(0.0, temp_max_value, 30)
                axs[ax_index].hist(link_tokens, rwidth=1, bins=bins, color=linecolor, edgecolor=edgecolor)
                axs[ax_index].set_title(f"Link Tokens switching Side")
                # axs[ax_index].set_yscale('log')
                ax_index += 1

                axs[ax_index].set_title(f"Dimensionality Curves s={round(self.data.mean_steps_history[-1], 3)}")
                for j in range(len(self.data.dims)):
                    axs[ax_index].plot(np.linspace(0.0, 1.0, dims_amounts[j]), dims[j], color=linecolor, alpha=0.15)
                axs[ax_index].plot(np.linspace(0.0, 1.0, 3), [self.data.dim_max_history[-1]] * 3,
                         color="cornflowerblue", alpha=1)
                axs[ax_index].plot(np.linspace(0.0, 1.0, 3), [self.data.dim_mean_history[-1]] * 3,
                         color="cornflowerblue", alpha=1)
                axs[ax_index].set_xticks([])
                axs[ax_index].set_ylim(0.0, 6)
                ax_index += 1

                axs[ax_index].set_title(f"Dimensionality {round(self.data.dim_max_history[-1], 3)} {round(self.data.dim_mean_history[-1], 3)}")
                x_data = list(range(len(self.data.dim_max_history)))
                y1 = np.maximum(np.array(self.data.dim_max_history) - np.array(self.data.dim_max_stddev_history), 0.0)
                y2 = np.array(self.data.dim_max_history) + np.array(self.data.dim_max_stddev_history)
                axs[ax_index].fill_between(x_data, y1, y2, color=linecolor, alpha=0.25)
                axs[ax_index].plot(x_data, self.data.dim_max_history, color=linecolor, alpha=1)
                y1 = np.maximum(np.array(self.data.dim_mean_history) - np.array(self.data.dim_mean_stddev_history), 0.0)
                y2 = np.array(self.data.dim_mean_history) + np.array(self.data.dim_mean_stddev_history)
                axs[ax_index].fill_between(x_data, y1, y2, color="cornflowerblue", alpha=0.25)
                axs[ax_index].plot(x_data, self.data.dim_mean_history, color="cornflowerblue", alpha=1)
                axs[ax_index].plot(x_data, [3.0] * len(self.data.dim_max_history), color="coral", alpha=1)
                axs[ax_index].set_ylim(0.0, 5.5)
                ax_index += 1

                x_iter = np.linspace(0, self.current_iteration, len(self.data.percent_home_history))

                axs[ax_index].set_title(f"Steps Diameter {round(self.data.mean_steps_history[-1], 4)}")
                axs[ax_index].plot(x_iter, self.data.mean_steps_history, color=linecolor)
                ax_index += 1

                axs[ax_index].set_title(f"Particles at Home {round(self.data.percent_home_history[-1], 4)}")
                axs[ax_index].plot(x_iter, self.data.percent_home_history, color=linecolor)
                ax_index += 1

                axs[ax_index].set_title(
                    f"Particle and Link Amount {self.data.particle_amount_history_init[-1]} {self.data.link_amount_history_init[-1]}")
                axs[ax_index].plot(x_iter, self.data.particle_amount_history_init, color=linecolor)
                axs[ax_index].plot(x_iter, self.data.link_amount_history_init, color=linecolor)
                ax_index += 1

                axs[ax_index].set_title(f"Links per Particle  {round(self.data.links_per_particle_history[-1], 4)}")
                axs[ax_index].plot(x_iter, self.data.links_per_particle_history, color=linecolor)
                ax_index += 1

                axs[ax_index].set_title(f"Shannon Entropy {round(self.data.shannon_entropy[-1], 3)}")
                axs[ax_index].plot(x_iter, self.data.shannon_entropy, color=linecolor)
                ax_index += 1

                axs[ax_index].set_title(
                    f"New Particles W=R  {round(self.data.reproduced_particles_history[-1], 4)}, "
                    f"B=P  {round(self.data.planted_particles_history[-1], 4)}")
                axs[ax_index].plot(x_iter, self.data.reproduced_particles_history, color=linecolor)
                axs[ax_index].plot(x_iter, self.data.planted_particles_history, color="cornflowerblue")
                ax_index += 1

                axs[ax_index].set_title(f"Reproduction Tokens ")
                axs[ax_index].plot(x_iter, self.data.kept_repro_tokens_history, color=linecolor)
                ax_index += 1


                axs[ax_index].set_title(f"Died Particles {round(self.data.died_particles_history[-1], 4)}")
                axs[ax_index].plot(x_iter, self.data.died_particles_history, color=linecolor)
                ax_index += 1

                axs[ax_index].set_title(
                    f"New Spawned Links {round(self.data.new_links_spawned_history[-1], 4)}, "
                    f"{round(self.data.declined_new_links_history[-1], 4)}")
                axs[ax_index].plot(x_iter, self.data.new_links_spawned_history, color=linecolor)
                axs[ax_index].plot(x_iter, self.data.declined_new_links_history, color="cornflowerblue")
                ax_index += 1

                axs[ax_index].set_title(f"Inactive Links {round(self.data.inactive_links_history[-1], 4)}")
                axs[ax_index].plot(x_iter, self.data.inactive_links_history, color=linecolor)
                ax_index += 1

                """
                if self.sim_options.get(SimOptionsEnum.CAN_SWAP):
                    axs[ax_index].set_title(f"Swapped Particles {round(self.data.swap_percentage_history[-1], 4)}")
                    axs[ax_index].plot(x_iter, self.data.swap_percentage_history, color=linecolor)
                    ax_index += 1

                if self.sim_options.get(SimOptionsEnum.CAN_RECONNECT):
                    axs[ax_index].set_title(f"Reconnections {round(self.data.reconnection_history[-1], 4)}")
                    axs[ax_index].plot(x_iter, self.data.reconnection_history, color=linecolor)
                    ax_index += 1
                """

                axs[ax_index].set_title(f"Token Invested Self")
                axs[ax_index].plot(x_iter, self.data.token_self_invested_history, color=linecolor)
                ax_index += 1


                axs[ax_index].set_title(f"Win Token Percentage B=REPRO")
                axs[ax_index].plot(x_iter, self.data.win_token_percentage_history, color=linecolor)
                ax_index += 1

                axs[ax_index].set_title(f"Win Token Relative Percentage")
                y1 = np.maximum(np.array(self.data.relative_win_token_percentage_history) - np.array(self.data.relative_win_token_percentage_stddev_history), 0.0)
                y2 = np.minimum(np.array(self.data.relative_win_token_percentage_history) + np.array(self.data.relative_win_token_percentage_stddev_history), 1.0)
                axs[ax_index].fill_between(x_data, y1, y2, color=linecolor, alpha=0.25)
                axs[ax_index].plot(x_data, self.data.relative_win_token_percentage_history, color=linecolor, alpha=1)
                ax_index += 1

                bins = np.linspace(0.0, 1.0, 30)
                axs[ax_index].hist(self.data.relative_win_token_percentage_array_history, rwidth=1, bins=bins, color=linecolor, edgecolor=edgecolor)
                axs[ax_index].set_title(f"Win Token Relative Percentage Distribution Total")
                ax_index += 1

                bins = np.linspace(0.0, max(self.data.attacked_with_x_tokens), 30)
                axs[ax_index].hist(self.data.attacked_with_x_tokens, rwidth=1, bins=bins, color=linecolor, edgecolor=edgecolor)
                axs[ax_index].set_title(f"Tokens used to attack (self or other)")
                ax_index += 1

                """
                temp = 1.0
                if len(self.data.each_relative_win_token_percentage_array_history) > 0:
                    temp =max(self.data.each_relative_win_token_percentage_array_history)
                self.data.max_each_relative_win_token_percentage_array_history = max(self.data.max_each_relative_win_token_percentage_array_history * backlash, temp)
                bins = np.linspace(0.0, max(1.0, self.data.max_each_relative_win_token_percentage_array_history), 30)
                axs[ax_index].hist(self.data.each_relative_win_token_percentage_array_history, rwidth=1, bins=bins,
                                   color=linecolor, edgecolor=edgecolor)
                axs[ax_index].set_title(f"Win Token Relative Percentage Distribution Each")
                ax_index += 1
                """


                axs[ax_index].set_title(f"Win Behavior is Self")
                axs[ax_index].plot(x_iter, self.data.win_behavior_is_self_history, color=linecolor)
                ax_index += 1

                axs[ax_index].set_title(f"Max Token Track")
                axs[ax_index].plot(x_iter, self.data.max_token_track, color=linecolor)
                ax_index += 1

                axs[ax_index].set_title(f"Max attack value track")
                axs[ax_index].plot(x_iter, self.data.max_attack_track, color=linecolor)
                ax_index += 1



                for ax in axs:
                    ax.grid(axis="y")

                if len(dead_par_pos) > 0:
                    self.my_draw_networkx_dead_nodes(node_pos=dead_par_pos, rotation_matrix=rotation_matrix, ax=ax1, **options_dead_node)
                if len(self.dead_links) > 0:
                    self.my_draw_networkx_dead_edges(dead_links=self.dead_links, ax=ax1, rotation_matrix=rotation_matrix, **options_dead_edge)
                nx.draw_networkx_nodes(G=g_plot, pos=projected_pos, ax=ax1, **options_node)
                if len(self.links) > 0:
                    self.my_draw_networkx_edges(G=g_plot, ax=ax1, rotation_matrix=rotation_matrix, **options_edge)
                ax1.margins(x=0.1)
                ax1.set_axis_off()
                ax1.set_xlim(-lim, lim)
                ax1.set_ylim(-lim, lim)
                ax2.set_title("Phylogenetic Tree")
                ax2.set_axis_off()


                plt.savefig(rf"{self.run_options_dict[RunOptionsEnum.SAVE_FOLDER]}\{self.name}\Plot {int(self.current_plot)}, i = {self.current_iteration}.png",
                            facecolor=self.vis_options.backgroundcolor, edgecolor='none')
                self.current_plot += 1
                plt.close()
            else:
                res_x = 1920
                aspect_ratio = 9 / 16  # Adjusted to 9/16 for a vertical orientation
                fig_width = 9
                fig_height = fig_width * aspect_ratio
                dpi = res_x / fig_width

                fig = plt.figure(dpi = dpi) #figsize=(fig_height, fig_width), dpi=dpi
                fig.set_size_inches(fig_height, fig_width, forward=False)
                """
                if background_color == "black":
                    plt.style.use('dark_background')
                else:
                    plt.style.use('default')
                """
                # ax1 = plt.subplot2grid((1, 1), (0, 0), colspan=1, rowspan=1)
                ax1 = plt.Axes(fig, [0,0,1.0,1.0])
                xmin, xmax = xlim = -1, 1
                ymin, ymax = ylim = -16/9, 16/9

                X = [[0, 0], [1, 1]]
                ax1.set_axis_off()
                fig.add_axes(ax1)



                cmap = mcolors.LinearSegmentedColormap.from_list("black_colormap", [self.vis_options.backgroundcolor, self.vis_options.backgroundcolor2], N=256)
                ax1.imshow(X, interpolation='bicubic', cmap=cmap,
                          extent=(xmin, xmax, ymin, ymax), alpha=1)


                if len(dead_par_pos) > 0:
                    self.my_draw_networkx_dead_nodes(node_pos=dead_par_pos, rotation_matrix=rotation_matrix, ax=ax1,
                                                     **options_dead_node)
                if len(self.dead_links) > 0:
                    self.my_draw_networkx_dead_edges(dead_links=self.dead_links, ax=ax1,
                                                     rotation_matrix=rotation_matrix, **options_dead_edge)
                nx.draw_networkx_nodes(G=g_plot, pos=projected_pos, ax=ax1, **options_node)
                if len(self.links) > 0:
                    self.my_draw_networkx_edges(G=g_plot, ax=ax1, rotation_matrix=rotation_matrix, **options_edge)


                square_lim_ratio = fig_width / fig_height
                ax1.set_xlim(-lim*0.8, lim*0.8)
                ax1.set_ylim(-lim*square_lim_ratio*0.8, lim*square_lim_ratio*0.8)

                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                plt.margins(0, 0)
                ax1.xaxis.set_major_locator(plt.NullLocator())
                ax1.yaxis.set_major_locator(plt.NullLocator())

                plt.savefig(rf"{self.run_options_dict[RunOptionsEnum.SAVE_FOLDER]}\{self.name}\Plot {int(self.current_plot)}, i = {self.current_iteration}.png")
                self.current_plot += 1
                plt.close()

    def my_draw_networkx_dead_nodes(
            self,
            node_pos,
            rotation_matrix = None,
            node_size=300,
            node_color="#1f78b4",
            node_shape="o",
            alpha=None,
            ax=None,
            linewidths=None,
            edgecolors=None,
    ):

        if len(node_pos) == 0:  # empty nodelist, no drawing
            return matplotlib.collections.PathCollection(None)

        try:
            xy = np.asarray([np.matmul(rotation_matrix, cur_node_pos) for cur_node_pos in node_pos])
        except KeyError as e:
            raise nx.NetworkXError(f"Node {e} has no position.") from e
        except ValueError as e:
            raise nx.NetworkXError("Bad value in node positions.") from e
        node_collection = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=node_size,
            c=node_color,
            marker=node_shape,
            alpha=alpha,
            linewidths=linewidths,
            edgecolors=edgecolors,
        )
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )

        node_collection.set_zorder(2)

    def my_draw_networkx_edges(self,
    G,
    width=1.0,
    rotation_matrix= None,
    edge_color="k",
    style="solid",
    alpha=None,
    arrowstyle="-|>",
    arrowsize=10,
    edge_cmap=None,
    edge_vmin=None,
    edge_vmax=None,
    ax=None,
    node_size=300,
    node_shape="o",
    connectionstyle=None,
    min_source_margin=0,
    min_target_margin=0,
    ):
        edgelist = list(G.edges())
        nodelist = list(G.nodes())
        edge_pos = np.asarray([(np.matmul(rotation_matrix, e[0].get_own_node_to_other_particle(e[1]).vis_pos), np.matmul(rotation_matrix, e[1].get_own_node_to_other_particle(e[0]).vis_pos)) for e in edgelist])

        if (
                np.iterable(edge_color)
                and (len(edge_color) == len(edge_pos))
                and np.alltrue([isinstance(c, Number) for c in edge_color])
        ):
            if edge_cmap is not None:
                assert isinstance(edge_cmap, mcolors.Colormap)
            else:
                edge_cmap = plt.get_cmap()
            if edge_vmin is None:
                if len(edge_color) > 0:
                    edge_vmin = min(edge_color)
                else:
                    edge_vmin = 0
            if edge_vmax is None:
                if len(edge_color) > 0:
                    edge_vmax = max(edge_color)
                else:
                    edge_vmax = 1
            color_normal = mcolors.Normalize(vmin=edge_vmin, vmax=edge_vmax)
            edge_color = [edge_cmap(color_normal(e)) for e in edge_color]

        def to_marker_edge(marker_size, marker):
            if marker in "s^>v<d":  # `large` markers need extra space
                return np.sqrt(2 * marker_size) / 2
            else:
                return np.sqrt(marker_size) / 2

        # Draw arrows with `matplotlib.patches.FancyarrowPatch`
        arrow_collection = []
        mutation_scale = arrowsize  # scale factor of arrow head

        # FancyArrowPatch doesn't handle color strings
        arrow_colors = mcolors.colorConverter.to_rgba_array(edge_color, alpha)
        for i, (src, dst) in enumerate(edge_pos):
            x1, y1 = src[0:2]
            x2, y2 = dst[0:2]
            shrink_source = 0  # space from source to tail
            shrink_target = 0  # space from  head to target
            if np.iterable(node_size):  # many node sizes
                source, target = edgelist[i][:2]
                source_node_size = node_size[nodelist.index(source)]
                target_node_size = node_size[nodelist.index(target)]
                shrink_source = to_marker_edge(source_node_size, node_shape)
                shrink_target = to_marker_edge(target_node_size, node_shape)
            else:
                shrink_source = shrink_target = to_marker_edge(node_size, node_shape)

            if shrink_source < min_source_margin:
                shrink_source = min_source_margin

            if shrink_target < min_target_margin:
                shrink_target = min_target_margin

            if len(arrow_colors) == len(edge_pos):
                arrow_color = arrow_colors[i]
            elif len(arrow_colors) == 1:
                arrow_color = arrow_colors[0]
            else:  # Cycle through colors
                arrow_color = arrow_colors[i % len(arrow_colors)]

            if np.iterable(width):
                if len(width) == len(edge_pos):
                    line_width = width[i]
                else:
                    line_width = width[i % len(width)]
            else:
                line_width = width

            arrow = matplotlib.patches.FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle=arrowstyle,
                shrinkA=shrink_source,
                shrinkB=shrink_target,
                mutation_scale=mutation_scale,
                color=arrow_color,
                linewidth=line_width,
                connectionstyle=connectionstyle,
                linestyle=style,
                zorder=1,
            )  # arrows go behind nodes

            # There seems to be a bug in matplotlib to make collections of
            # FancyArrowPatch instances. Until fixed, the patches are added
            # individually to the axes instance.
            arrow_collection.append(arrow)
            ax.add_patch(arrow)

        # update view
        minx = np.amin(np.ravel(edge_pos[:, :, 0]))
        maxx = np.amax(np.ravel(edge_pos[:, :, 0]))
        miny = np.amin(np.ravel(edge_pos[:, :, 1]))
        maxy = np.amax(np.ravel(edge_pos[:, :, 1]))

        w = maxx - minx
        h = maxy - miny
        padx, pady = 0.05 * w, 0.05 * h
        corners = (minx - padx, miny - pady), (maxx + padx, maxy + pady)
        ax.update_datalim(corners)
        ax.autoscale_view()

        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )


    def my_draw_networkx_dead_edges(self,
    dead_links,
    width=1.0,
    rotation_matrix= None,
    edge_color="k",
    style="solid",
    alpha=None,
    arrowstyle="-|>",
    arrowsize=10,
    edge_cmap=None,
    edge_vmin=None,
    edge_vmax=None,
    ax=None,
    node_size=300,
    node_shape="o",
    connectionstyle=None,
    min_source_margin=0,
    min_target_margin=0,
    ):
        edge_pos = np.asarray([(np.matmul(rotation_matrix, cur_link.node1.vis_pos), np.matmul(rotation_matrix, cur_link.node2.vis_pos)) for cur_link in dead_links])

        if (
                np.iterable(edge_color)
                and (len(edge_color) == len(edge_pos))
                and np.alltrue([isinstance(c, Number) for c in edge_color])
        ):
            if edge_cmap is not None:
                assert isinstance(edge_cmap, mcolors.Colormap)
            else:
                edge_cmap = plt.get_cmap()
            if edge_vmin is None:
                edge_vmin = min(edge_color)
            if edge_vmax is None:
                edge_vmax = max(edge_color)
            color_normal = mcolors.Normalize(vmin=edge_vmin, vmax=edge_vmax)
            edge_color = [edge_cmap(color_normal(e)) for e in edge_color]

        def to_marker_edge(marker_size, marker):
            if marker in "s^>v<d":  # `large` markers need extra space
                return np.sqrt(2 * marker_size) / 2
            else:
                return np.sqrt(marker_size) / 2

        # Draw arrows with `matplotlib.patches.FancyarrowPatch`
        arrow_collection = []
        mutation_scale = arrowsize  # scale factor of arrow head

        # FancyArrowPatch doesn't handle color strings
        arrow_colors = mcolors.colorConverter.to_rgba_array(edge_color, alpha)
        for i, (src, dst) in enumerate(edge_pos):
            x1, y1 = src[0:2]
            x2, y2 = dst[0:2]
            shrink_source = 0  # space from source to tail
            shrink_target = 0  # space from  head to target
            shrink_source = shrink_target = to_marker_edge(node_size, node_shape)

            if shrink_source < min_source_margin:
                shrink_source = min_source_margin

            if shrink_target < min_target_margin:
                shrink_target = min_target_margin

            if len(arrow_colors) == len(edge_pos):
                arrow_color = arrow_colors[i]
            elif len(arrow_colors) == 1:
                arrow_color = arrow_colors[0]
            else:  # Cycle through colors
                arrow_color = arrow_colors[i % len(arrow_colors)]

            if np.iterable(width):
                if len(width) == len(edge_pos):
                    line_width = width[i]
                else:
                    line_width = width[i % len(width)]
            else:
                line_width = width

            arrow = matplotlib.patches.FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle=arrowstyle,
                shrinkA=shrink_source,
                shrinkB=shrink_target,
                mutation_scale=mutation_scale,
                color=arrow_color,
                linewidth=line_width,
                connectionstyle=connectionstyle,
                linestyle=style,
                zorder=1,
            )  # arrows go behind nodes

            # There seems to be a bug in matplotlib to make collections of
            # FancyArrowPatch instances. Until fixed, the patches are added
            # individually to the axes instance.
            arrow_collection.append(arrow)
            ax.add_patch(arrow)

        # update view
        minx = np.amin(np.ravel(edge_pos[:, :, 0]))
        maxx = np.amax(np.ravel(edge_pos[:, :, 0]))
        miny = np.amin(np.ravel(edge_pos[:, :, 1]))
        maxy = np.amax(np.ravel(edge_pos[:, :, 1]))

        w = maxx - minx
        h = maxy - miny
        padx, pady = 0.05 * w, 0.05 * h
        corners = (minx - padx, miny - pady), (maxx + padx, maxy + pady)
        ax.update_datalim(corners)
        ax.autoscale_view()

        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )
