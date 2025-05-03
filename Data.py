import numpy as np
import json

class Serializable:
    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, data):
        obj = cls.__new__(cls)  # Create instance without calling __init__
        obj.__dict__.update(data)
        return obj

    def save_json(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

class SimulationIterationLogger:

    def __init__(self):

        # after phase 1
        self.nodes_p1 = [] #node_id
        self.edges_p1 = [] #edge_id, node1_id, node2_id
        self.behaviors_p1 = [] #node_id, corresponding_behavior_id
        self.node_walkers_p1 = [] #node_id, walker_node_id
        self.node_tokens_p1 = [] #node_id, token_amount

        self.behavior_swaps_p1 = [] #node1_id, node2_id
        self.link_reconnections_p1 = [] #node_id, edge_id, node_id_to_switch_to
        self.new_links_to_walker_p1 = [] #node1_id, node2_id
        self.declined_links = [] #node_id
        self.walker_movements_p1 = [] #node_id, node_id_walker_origin, node_id_walker_destination
        self.reproduced_nodes_p1 = [] #node_id_parent, node_id_child
        self.inherit_walker_position_p1 = [] #node_id_parent, node_id_child, node_id_walker
        self.reproduction_new_links_p1 = [] #node_id_parent, node_id_child, node_id_child_to_connect
        self.planted_children_p1 = [] #node_id_parent
        self.links_shifts_p1 = [] #node_id_that_shifts, edge_id_to_switch, node_id_switch_to
        self.dead_nodes_p1 = [] # node_id
        self.walker_movements_because_death_p1 = [] # node_id, node_id_walker_origin (the one that died), node_id_walker_destination

        # after phase 2
        self.nodes_p2 = [] #node_id
        self.edges_p2 = [] #edge_id, node1_id, node2_id
        self.behaviors_p2 = [] #node_id, corresponding_behavior_id
        self.node_walkers_p2 = [] #node_id, walker_node_id
        self.node_tokens_p2 = [] #node_id, token_amount

        self.token_attacks_p2 = [] #node_id_sender, node_id_receiver, token_amount
        self.walker_movements_because_death_p2 = [] #node_id, node_id_walker_origin (the one that died), node_id_walker_destination
        self.game_winner_p2 = [] #node_id_of_competition_place, node_id_of_winner
        self.game_behavior_transfer_p2 = []  # node_id, old_behavior_id, parent_behavior_id, new_behavior_id
        self.dead_nodes_p2 = []# node_id

        self.node_ages_p2 = [] #node_id, age
        self.behavior_ages_p2 = [] # behavior_id, age

        self.killed_links_tot = [] # edge_id (done in kill link)

    def store_simulation_iteration_information_p1(self, simulation):

        for cur_par in simulation.particles:
            self.nodes_p1.append(cur_par.id)
            self.behaviors_p1.append((cur_par.id, cur_par.behavior.id))
            self.node_walkers_p1.append((cur_par.id, cur_par.walker_position.id))
            self.node_tokens_p1.append((cur_par.id, int(cur_par.token)))
        for cur_link in simulation.links:
            self.edges_p1.append((cur_link.id, cur_link.node1.particle.id, cur_link.node2.particle.id))

    def store_simulation_iteration_information_p2(self, simulation):
        for cur_par in simulation.particles:
            self.nodes_p2.append(cur_par.id)
            self.behaviors_p2.append((cur_par.id, cur_par.behavior.id))
            self.node_walkers_p2.append((cur_par.id, cur_par.walker_position.id))
            self.node_tokens_p2.append((cur_par.id, int(cur_par.token)))

            self.node_ages_p2.append((cur_par.id, cur_par.age))
            self.behavior_ages_p2.append((cur_par.behavior.id, cur_par.behavior_age))

        for cur_link in simulation.links:
            self.edges_p2.append((cur_link.id, cur_link.node1.particle.id, cur_link.node2.particle.id))


class Data:
    """
    Data container for analysis of the simulation.
    """

    def get_new_node_index(self):
        self.current_node_index += 1
        return self.current_node_index

    def get_new_edge_index(self):
        self.current_edge_index += 1
        return self.current_edge_index

    def get_new_behavior_index(self):
        self.current_behavior_index += 1
        return self.current_behavior_index


    def __init__(self):
        """
        Initialize all all data variables
        """

        self.current_node_index = 0
        self.current_edge_index = 0
        self.current_behavior_index = 0

        self.simulation_iteration_logger = None

        self.particle_amount_history_init = []
        self.particle_amount_history_after_phase1 = []

        self.max_token_track = []
        self.max_attack_track = []
        self.max_age_track = []

        self.link_amount_history_init = []
        self.link_amount_history_after_phase1 = []
        self.percent_home_history = []
        self.dim_max_history = []
        self.dim_mean_history = []
        self.max_link_amount_history = 1
        self.max_par_conc_history = 1
        self.max_token_amount_history = 1
        self.max_token_link_amount_history = 1
        self.max_age_history = 1
        self.max_beh_age_history = 1
        self.max_link_age_history = 1
        self.max_link_active_value_history = 1

        self.dim_max_stddev_history = []
        self.dim_mean_stddev_history = []
        self.swap_percentage_history = []
        self.new_links_spawned_history = []
        self.declined_new_links_history = []
        self.planted_particles_history = []
        self.reproduced_particles_history = []
        self.died_particles_history = []
        self.links_per_particle_history = []
        self.kept_repro_tokens_history = []
        self.shannon_entropy = []
        self.mean_steps_history = []

        self.reconnection_history = []

        self.inactive_links_history = []

        self.token_self_invested_history = []
        self.token_other_invested_history = []
        self.win_token_percentage_history = []

        self.relative_win_token_percentage_history = []
        self.relative_win_token_percentage_stddev_history = []
        self.relative_win_token_percentage_array_history = []
        self.attacked_with_x_tokens = []

        self.win_behavior_is_self_history = []
        self.each_relative_win_token_percentage_array_history = []
        self.max_each_relative_win_token_percentage_array_history = 1.0

        self.link_amount_crit_array = []
        self.link_amount_crit_array2 = []

        self.birth_tree = []
        self.next_birth_layer = []

        self.token_power_law_exponent_history = []
        self.par_concentration_power_law_exponent_history = []
        self.link_amounts_power_law_exponent_history = []
        self.ages_power_law_exponent_history = []

    def prepare(self, particles, links):

        self.simulation_iteration_logger = SimulationIterationLogger()

        # Prepare data arrays for analysis
        self.particle_amount_history_init.append(len(particles))
        self.link_amount_history_init.append(len(links))
        self.links_per_particle_history.append(self.link_amount_history_init[-1]/self.particle_amount_history_init[-1])
        sum_x = sum([1 for cur_par in particles if cur_par is cur_par.walker_position])
        self.percent_home_history.append(sum_x / self.particle_amount_history_init[-1])
        self.swap_percentage_history.append(0.0)
        self.new_links_spawned_history.append(0.0)
        self.declined_new_links_history.append(0.0)
        self.planted_particles_history.append(0.0)
        self.reproduced_particles_history.append(0.0)
        self.died_particles_history.append(0.0)
        self.kept_repro_tokens_history.append(0.0)

        self.reconnection_history.append(0.0)

        self.inactive_links_history.append(0.0)

        self.token_self_invested_history.append(0.0)
        self.token_other_invested_history.append(0.0)
        self.win_token_percentage_history.append(0.0)

        self.relative_win_token_percentage_history.append([])



        self.each_relative_win_token_percentage_array_history = []
        self.attacked_with_x_tokens = []

        self.win_behavior_is_self_history.append(0.0)

        self.next_birth_layer = []

    def sort_birth_layer(self):

        if len(self.birth_tree) > 0:
            previous_layer = self.birth_tree[-1]

            # Mapping of parent index to its position in the previous layer
            parent_index_to_position = {entry[0]: pos for pos, entry in enumerate(previous_layer)}

            # Sorting the current layer based on the ancestry hierarchy
            self.next_birth_layer.sort(key=lambda x: parent_index_to_position.get(
                x[1], parent_index_to_position.get(
                    x[2], parent_index_to_position.get(x[3], -1)
                )
            ))
        self.birth_tree.append(self.next_birth_layer)
        self.next_birth_layer = []


    def renormalize(self, token_amount):


        self.swap_percentage_history[-1] = self.swap_percentage_history[-1] / self.particle_amount_history_init[-1]
        self.new_links_spawned_history[-1] = self.new_links_spawned_history[-1] / self.particle_amount_history_init[-1]
        self.declined_new_links_history[-1] = self.declined_new_links_history[-1] / self.particle_amount_history_init[-1]
        self.planted_particles_history[-1] = self.planted_particles_history[-1] / self.particle_amount_history_init[-1]
        self.reproduced_particles_history[-1] = self.reproduced_particles_history[-1] / self.particle_amount_history_init[-1]

        self.died_particles_history[-1] = self.died_particles_history[-1] / self.particle_amount_history_after_phase1[-1]
        self.kept_repro_tokens_history[-1] = self.kept_repro_tokens_history[-1] / token_amount

        self.reconnection_history[-1] /= self.particle_amount_history_init[-1]

        self.inactive_links_history[-1] /= max(self.link_amount_history_after_phase1[-1],1)

        self.token_self_invested_history[-1] /= token_amount
        self.token_other_invested_history[-1] /= token_amount
        self.win_token_percentage_history[-1] /= token_amount

        if len(self.relative_win_token_percentage_history[-1]) > 0:
            self.relative_win_token_percentage_array_history = self.relative_win_token_percentage_history[-1].copy()
            self.relative_win_token_percentage_stddev_history.append(np.std(self.relative_win_token_percentage_history[-1]))
            self.relative_win_token_percentage_history[-1] = np.mean(self.relative_win_token_percentage_history[-1])

        else:
            self.relative_win_token_percentage_array_history = [0.0]
            self.relative_win_token_percentage_stddev_history.append(0.0)
            self.relative_win_token_percentage_history[-1] = 0.0


        self.win_behavior_is_self_history[-1] /= self.particle_amount_history_after_phase1[-1]

