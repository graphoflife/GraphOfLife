import numpy as np

class Data:
    """
    Data container for analysis of the simulation.
    """

    def __init__(self):
        """
        Initialize all all data variables
        """
        self.particle_amount_history_init = []
        self.particle_amount_history_after_phase1 = []

        self.max_token_track = []
        self.max_attack_track = []

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

    def prepare(self, particles, links):

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

