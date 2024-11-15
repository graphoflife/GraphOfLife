from SimOptions import *
from Behavior import *
from Link import *

import numpy as np
import logging

class Particle:
    """
    The autonomous particle that is the core of the simulation.
    """

    def __init__(self, token, behavior):

        self.nodes = []
        self.token = token
        self.walker_position = self
        self.particles_at_this_position = [self]
        self.behavior = behavior
        self.new_behavior = None
        self.age = 0
        self.behavior_age = 1
        self.can_new_link = True
        self.to_mutate = False
        self.inherit_walker_position = False
        self.reproduction_tokens = 0
        self.vis_max_color_value = self.token

        self.information = None
        self.make_new_link = None
        self.go_particle = None
        self.reproduction_behavior = None
        self.reproduce_at_home = None
        self.plant_particle = None

        self.reconnection_node = None
        self.reconnection_particle = None

        self.vis_pos = None
        self.vis_vel = None

        self.blotto_competition_here = None

        self.blotto_winner_here = None
        self.blotto_winner_repro = None

        self.link_creation_particles_for_reproduction = None




    def do_reconnect(self, data):
        if self.reconnection_node is not None:
            self.reconnection_node.switch_to_particle(self.reconnection_particle)
            data.reconnection_history[-1] += 1.0

    def get_own_node_to_other_particle(self, other_particle):
        for cur_node in self.nodes:
            if cur_node.other_node.particle is other_particle:
                return cur_node
        return None

    def get_winner_and_amount_of_competition(self, sim_options, competition):
        items = list(competition.items())
        if sim_options.get(SimOptionsEnum.BLOTTO_PROB_WINNER) == False:
            np.random.shuffle(items)
            winner_particle = max(items, key=lambda x: x[1])[0]
        else:
            keys = [item[0] for item in items]
            weights = np.array([item[1] for item in items])
            if np.sum(weights) == 0.0:
                weights = np.ones_like(weights)
            normalized_weights = weights / np.sum(weights)

            #print("comp" ,competition)
            #print("w", normalized_weights)

            winner_particle = np.random.choice(keys, p=normalized_weights)
        return winner_particle, sum(competition.values())

    def eval_game(self, sim_options, data, indexer_f):
        """
        :param sim_options: Options of Simulation
        :param data: Data for analysis
        """
        winner_particle_here, sum_token_here = self.get_winner_and_amount_of_competition(sim_options=sim_options, competition=self.blotto_competition_here)

        if sim_options.get(SimOptionsEnum.WINNER_TOKEN_GO_TO_COMPETITION_PLACE):
            self.token += sum_token_here
        else:   # "WINNER_PLACE"
            winner_particle_here.token += sum_token_here

        if sim_options.get(SimOptionsEnum.WINNER_HERE_BEHAVIOR_WINS):
            self.new_behavior = winner_particle_here.behavior
            if winner_particle_here is not self:
                self.behavior_age = 1
        else:   # "SELF_PARTICLE"
            self.new_behavior = self.behavior

        if sim_options.get(SimOptionsEnum.MUTATE_WIN_BEHAVIOR):
            self.to_mutate = True

        # Data

        win_token_amount = self.blotto_competition_here[winner_particle_here]

        for key in self.blotto_competition_here:
            if key is not winner_particle_here:
                value = self.blotto_competition_here[key]
                if value > 0:
                    data.each_relative_win_token_percentage_array_history.append(value/win_token_amount)

        data.win_token_percentage_history[-1] += win_token_amount

        if sum_token_here != 0.0:
            data.relative_win_token_percentage_history[-1].append(win_token_amount/sum_token_here)

        if self.new_behavior is self.behavior:
            data.win_behavior_is_self_history[-1] += 1

    def apply_new_behavior(self, sim_options, indexer_f, data):

        self.behavior = Behavior(sim_options=sim_options, parent_behavior=self.new_behavior, indexer_f=indexer_f, data=data)

    def link_amount(self):
        """
        :return: Gets the amount of links this particle has.
        """
        return len(self.nodes)

    def prepare_information(self, sim_options):

        self.blotto_competition_here = {}
        self.blotto_winner_here = None
        self.blotto_winner_repro = None
        self.new_behavior = None

        if len(self.nodes) == 0:
            token_quantiles = np.zeros((sim_options.QUANTILE_AMOUNT, 1))
            link_quantiles = np.zeros((sim_options.QUANTILE_AMOUNT, 1))
            particle_quantiles = np.zeros((sim_options.QUANTILE_AMOUNT, 1))
        else:
            # Quantiles Information
            token_amounts = [cur_node.other_node.particle.token for cur_node in self.nodes]
            link_amounts = [cur_node.other_node.particle.link_amount() for cur_node in self.nodes]
            particle_amounts = [len(cur_node.other_node.particle.particles_at_this_position)+1 for cur_node in self.nodes]
            token_amounts.sort()
            link_amounts.sort()
            particle_amounts.sort()
            indexes = np.linspace(0, self.link_amount() - 1, sim_options.QUANTILE_AMOUNT)
            token_quantiles = []
            link_quantiles = []
            particle_quantiles = []
            for cur_index in indexes:
                first_index = int(cur_index)
                percent_of_second = cur_index % 1
                if percent_of_second != 0.0:
                    token_quantiles.append(token_amounts[first_index] * (1 - percent_of_second) + token_amounts[
                        first_index + 1] * percent_of_second)
                    link_quantiles.append(link_amounts[first_index] * (1 - percent_of_second) + link_amounts[
                        first_index + 1] * percent_of_second)
                    particle_quantiles.append(particle_amounts[first_index] * (1 - percent_of_second) + particle_amounts[
                        first_index + 1] * percent_of_second)
                else:
                    token_quantiles.append(float(token_amounts[first_index]))
                    link_quantiles.append(float(link_amounts[first_index]))
                    particle_quantiles.append(float(particle_amounts[first_index]))

            token_quantiles = np.array(token_quantiles).reshape(sim_options.QUANTILE_AMOUNT, 1)*sim_options.INPUT_SCALING
            link_quantiles = np.array(link_quantiles).reshape(sim_options.QUANTILE_AMOUNT, 1)*sim_options.INPUT_SCALING
            particle_quantiles = np.array(particle_quantiles).reshape(sim_options.QUANTILE_AMOUNT, 1)*sim_options.INPUT_SCALING

        self.information = np.vstack([self.token*sim_options.INPUT_SCALING,
                                      self.age*sim_options.INPUT_SCALING,
                                      self.behavior_age*sim_options.INPUT_SCALING,
                                      self.link_amount()*sim_options.INPUT_SCALING,
                                      (len(self.particles_at_this_position))*sim_options.INPUT_SCALING,
                                      token_quantiles, link_quantiles, particle_quantiles])

    def prepare(self, sim_options):
        """
        Prepares information for next iteration. This includes assembling a information vector about the state of
        this particle with some quantile information about its neighbors.
        :param sim_options: Options of Simulation
        """
        self.vis_max_color_value = self.token
        self.go_particle = None
        self.reproduction_tokens = 0
        self.age += 1
        self.behavior_age += 1
        self.to_mutate = False
        self.inherit_walker_position = False
        self.can_new_link = True
        self.reconnection_node = None
        self.reconnection_particle = None
        self.link_creation_particles_for_reproduction = []

        np.random.shuffle(self.nodes)

        self.prepare_information(sim_options=sim_options)

    def get_link_to_particle(self, to_particle):
        """
        :param to_particle: The link to this particle is wanted.
        :return: Returns the link that connects this particle to another particle.
        """
        for cur_node in self.nodes:
            if cur_node.other_node.particle is to_particle:
                return cur_node.link
        return None


    def process_information_and_make_decisions_game_phase(self, sim_options, data, all_particles):
        """
        Assembles input vector for each node, feeds input through own, other and position behaviors and makes decisions
        according to the outputs of this process.
        :param sim_options: Options of Simulation
        :param data: Data for analysis
        """

        if len(self.nodes) == 0:
            self.blotto_competition_here[self] = self.token
            self.token = 0
            for cur_particle in self.particles_at_this_position:
                cur_particle.go_particle = cur_particle
            return

        walker_at_home = self.walker_position is self
        pos_link = self.get_link_to_particle(to_particle=self.walker_position)
        if pos_link:
            pos_link_connection = 1
        else:
            pos_link_connection = 0

        output_arrays_self = []
        output_arrays_other = []
        output_arrays_walker = []

        outputs_for_walkers_at_this_position = [[] for _ in range(len(self.particles_at_this_position))]



        for i, cur_node in enumerate(self.nodes):
            other_walker_at_home = cur_node.other_node.particle.walker_position is cur_node.other_node.particle

            input_array = np.vstack([0.0, 0.0, 0.0, self.information,
                                     cur_node.other_node.particle.information,
                                     self.walker_position.information,
                                     cur_node.other_node.particle.walker_position.information,
                                     walker_at_home,
                                     other_walker_at_home,
                                     pos_link_connection
                                     ])
            output_arrays_self.append(self.behavior.get_output(input_array=input_array, sim_options=sim_options))
            input_array[0] = 1.0
            output_arrays_other.append(cur_node.other_node.particle.behavior.get_output(input_array=input_array, sim_options=sim_options))
            input_array[1] = 1.0
            output_arrays_walker.append(self.walker_position.behavior.get_output(input_array=input_array, sim_options=sim_options))
            input_array[0] = 0.0
            if sim_options.get(SimOptionsEnum.CAN_MOVE_WHEN_DEATH):
                for j, cur_par in enumerate(self.particles_at_this_position):
                    if sim_options.get(SimOptionsEnum.MOVE_ON_OWN_BY_SELF):
                        ref_particle = self
                    else:
                        ref_particle = self.particles_at_this_position[j]
                    if sim_options.get(SimOptionsEnum.MOVE_ON_OWN, particle=ref_particle):
                        outputs_for_walkers_at_this_position[j].append(cur_par.behavior.get_output(input_array=input_array, sim_options=sim_options))


        output_arrays_self = np.hstack(output_arrays_self)
        output_arrays_other = np.hstack(output_arrays_other)
        output_arrays_walker = np.hstack(output_arrays_walker)
        if sim_options.get(SimOptionsEnum.CAN_MOVE_WHEN_DEATH):

            for j, cur_par in enumerate(self.particles_at_this_position):
                if sim_options.get(SimOptionsEnum.MOVE_ON_OWN_BY_SELF):
                    ref_particle = self
                else:
                    ref_particle = self.particles_at_this_position[j]
                if sim_options.get(SimOptionsEnum.MOVE_ON_OWN, particle=ref_particle):
                    outputs_for_walkers_at_this_position[j] = np.hstack(outputs_for_walkers_at_this_position[j])

        index = 0

        # Token allocation
        output_arrays = self.get_correct_output(
            use_x=sim_options.get(SimOptionsEnum.USE_X_FOR_TOKEN_ALLOCATION, particle=self),
            output_arrays_self=output_arrays_self,
            output_arrays_other=output_arrays_other,
            output_arrays_walker=output_arrays_walker)
        token_outputs = output_arrays[index:index + 2, :]

        sumx = np.sum(token_outputs)
        if sumx == 0.0:
            token_outputs = np.ones(token_outputs.shape)

        sum_tokens_here = np.mean(token_outputs[0, :])
        token_outputs[0, :] *= 0.0
        token_outputs[0, 0] = sum_tokens_here

        token_outputs_percent = token_outputs / token_outputs.sum()
        token_outputs_exact = token_outputs_percent * self.token
        token_outputs = np.round(token_outputs_exact)
        token_outputs_diff = token_outputs_exact - token_outputs
        for _ in range(max(int(np.sum(token_outputs) - self.token), 0)):
            index_temp = np.unravel_index(np.argmin(token_outputs_diff), token_outputs_diff.shape)
            token_outputs_diff[index_temp] = 0.0
            token_outputs[index_temp] -= 1
        for _ in range(max(int(self.token - np.sum(token_outputs)), 0)):
            index_temp = np.unravel_index(np.argmax(token_outputs_diff), token_outputs_diff.shape)
            token_outputs_diff[index_temp] = 0.0
            token_outputs[index_temp] += 1

        data.token_self_invested_history[-1] += token_outputs[0, 0]

        self.blotto_competition_here[self] = token_outputs[0, 0]
        if token_outputs[0, 0] > 0:
            data.attacked_with_x_tokens.append(token_outputs[0, 0])

        for i, cur_node in enumerate(self.nodes):
            if token_outputs[1, i] > 0:
                data.attacked_with_x_tokens.append(token_outputs[1, i])
            cur_node.link.active_value += token_outputs[1, i]
            data.token_other_invested_history[-1] += token_outputs[1, i]
            cur_node.other_node.particle.blotto_competition_here[self] = token_outputs[1, i]
        self.token = 0
        index += 2

        """
        # Dying direction
        output_arrays = self.get_correct_output(use_x=sim_options.get(SimOptionsEnum.USE_X_FOR_DEATH_RATING, particle=self),
                                                output_arrays_self=output_arrays_self,
                                                output_arrays_other=output_arrays_other,
                                                output_arrays_walker=output_arrays_walker)
        heritage_outputs = output_arrays[index, :]
        for i, cur_node in enumerate(self.nodes):
            cur_node.heritage_rating = heritage_outputs[i]
        index += 1
        """

        # Move particles
        if sim_options.get(SimOptionsEnum.CAN_MOVE_WHEN_DEATH):

            for i, cur_particle in enumerate(self.particles_at_this_position):
                if sim_options.get(SimOptionsEnum.MOVE_ON_OWN_BY_SELF):
                    ref_particle = self
                else:
                    ref_particle = self.particles_at_this_position[i]
                if sim_options.get(SimOptionsEnum.MOVE_ON_OWN, particle=ref_particle):

                    choices = [cur_particle, self, self.walker_position, np.random.choice(all_particles), ] + [
                        cur_node.other_node.particle for cur_node in self.nodes] + [
                                  cur_node.other_node.particle.walker_position for cur_node in self.nodes]

                    home_option = np.mean(outputs_for_walkers_at_this_position[i][index, :])
                    self_option = 0
                    walker_option = np.mean(outputs_for_walkers_at_this_position[i][index + 2, :])
                    random_option = np.mean(outputs_for_walkers_at_this_position[i][index + 3, :])
                    neighbor_option = outputs_for_walkers_at_this_position[i][index + 4, :]
                    neighbor_walker_option = outputs_for_walkers_at_this_position[i][index + 5, :]

                    if not sim_options.get(SimOptionsEnum.CAN_MOVE_HOME):
                        home_option = 0.0
                    if not sim_options.get(SimOptionsEnum.CAN_NOT_MOVE):
                        self_option = 0.0
                    if not sim_options.get(SimOptionsEnum.CAN_MOVE_TO_WALKERS_WALKER):
                        walker_option = 0.0
                    if not sim_options.get(SimOptionsEnum.CAN_MOVE_TO_RANDOM):
                        random_option = 0.0
                    if not sim_options.get(SimOptionsEnum.CAN_MOVE_TO_NEIGBHBOR):
                        neighbor_option *= 0.0
                    if not sim_options.get(SimOptionsEnum.CAN_MOVE_TO_NEIGBHBORS_WALKER):
                        neighbor_walker_option *= 0.0
                    probability = np.hstack(
                        [home_option, self_option, walker_option, random_option, neighbor_option, neighbor_walker_option])
                    if probability.sum() == 0.0:
                        home_option = 1.0
                        self_option = 1.0
                        walker_option = 1.0
                        random_option = 1.0
                        neighbor_option = np.ones_like(neighbor_option)
                        neighbor_walker_option = np.ones_like(neighbor_walker_option)
                        if not sim_options.get(SimOptionsEnum.CAN_MOVE_HOME):
                            home_option = 0.0
                        if not sim_options.get(SimOptionsEnum.CAN_NOT_MOVE):
                            self_option = 0.0
                        if not sim_options.get(SimOptionsEnum.CAN_MOVE_TO_WALKERS_WALKER):
                            walker_option = 0.0
                        if not sim_options.get(SimOptionsEnum.CAN_MOVE_TO_RANDOM):
                            random_option = 0.0
                        if not sim_options.get(SimOptionsEnum.CAN_MOVE_TO_NEIGBHBOR):
                            neighbor_option *= 0.0
                        if not sim_options.get(SimOptionsEnum.CAN_MOVE_TO_NEIGBHBORS_WALKER):
                            neighbor_walker_option *= 0.0
                        probability = np.hstack(
                            [home_option, self_option, walker_option, random_option, neighbor_option,
                             neighbor_walker_option])

                    choice = self.make_choice(to_choose=choices, p=probability,
                                              choice_type=sim_options.get(SimOptionsEnum.MOVING_PARTICLES_DECISION,
                                                                          particle=self.particles_at_this_position[i]))

                    cur_particle.go_particle = choice

                else:
                    output_arrays = self.get_correct_output(
                        use_x=sim_options.get(SimOptionsEnum.USE_X_FOR_MOVING_PARTICLES,
                                              particle=self.particles_at_this_position[i]),
                        output_arrays_self=output_arrays_self,
                        output_arrays_other=output_arrays_other,
                        output_arrays_walker=output_arrays_walker)

                    choices = [cur_particle, self, self.walker_position, np.random.choice(all_particles), ] + [
                        cur_node.other_node.particle for cur_node in self.nodes] + [
                                  cur_node.other_node.particle.walker_position for cur_node in self.nodes]

                    home_option = np.mean(output_arrays[index, :])
                    self_option = 0
                    walker_option = np.mean(output_arrays[index + 2, :])
                    random_option = np.mean(output_arrays[index + 3, :])
                    neighbor_option = output_arrays[index + 4, :]
                    neighbor_walker_option = output_arrays[index + 5, :]

                    if not sim_options.get(SimOptionsEnum.CAN_MOVE_HOME):
                        home_option = 0.0
                    if not sim_options.get(SimOptionsEnum.CAN_NOT_MOVE):
                        self_option = 0.0
                    if not sim_options.get(SimOptionsEnum.CAN_MOVE_TO_WALKERS_WALKER):
                        walker_option = 0.0
                    if not sim_options.get(SimOptionsEnum.CAN_MOVE_TO_RANDOM):
                        random_option = 0.0
                    if not sim_options.get(SimOptionsEnum.CAN_MOVE_TO_NEIGBHBOR):
                        neighbor_option *= 0.0
                    if not sim_options.get(SimOptionsEnum.CAN_MOVE_TO_NEIGBHBORS_WALKER):
                        neighbor_walker_option *= 0.0
                    probability = np.hstack(
                        [home_option, self_option, walker_option, random_option, neighbor_option, neighbor_walker_option])
                    if probability.sum() == 0.0:
                        home_option = 1.0
                        self_option = 1.0
                        walker_option = 1.0
                        random_option = 1.0
                        neighbor_option = np.ones_like(neighbor_option)
                        neighbor_walker_option = np.ones_like(neighbor_walker_option)
                        if not sim_options.get(SimOptionsEnum.CAN_MOVE_HOME):
                            home_option = 0.0
                        if not sim_options.get(SimOptionsEnum.CAN_NOT_MOVE):
                            self_option = 0.0
                        if not sim_options.get(SimOptionsEnum.CAN_MOVE_TO_WALKERS_WALKER):
                            walker_option = 0.0
                        if not sim_options.get(SimOptionsEnum.CAN_MOVE_TO_RANDOM):
                            random_option = 0.0
                        if not sim_options.get(SimOptionsEnum.CAN_MOVE_TO_NEIGBHBOR):
                            neighbor_option *= 0.0
                        if not sim_options.get(SimOptionsEnum.CAN_MOVE_TO_NEIGBHBORS_WALKER):
                            neighbor_walker_option *= 0.0
                        probability = np.hstack(
                            [home_option, self_option, walker_option, random_option, neighbor_option,
                             neighbor_walker_option])

                    choice = self.make_choice(to_choose=choices, p=probability,
                                              choice_type=sim_options.get(SimOptionsEnum.MOVING_PARTICLES_DECISION,
                                                                          particle=self.particles_at_this_position[i]))
                    cur_particle.go_particle = choice

        index += 6


    def process_information_and_make_decisions_repro_geo_phase(self, sim_options, data, all_particles):
        """
        Assembles input vector for each node, feeds input through own, other and position behaviors and makes decisions
        according to the outputs of this process.
        :param sim_options: Options of Simulation
        :param data: Data for analysis
        """

        if len(self.nodes) == 0:

            for cur_particle in self.particles_at_this_position:
                cur_particle.go_particle = cur_particle

            repro_tokens = int(self.token/2)
            self.reproduction_tokens += repro_tokens
            self.token -= repro_tokens

            self.make_new_link = False
            self.can_new_link = False
            self.reproduce_at_home = True
            return

        walker_at_home = self.walker_position is self
        pos_link = self.get_link_to_particle(to_particle=self.walker_position)
        if pos_link:
            pos_link_connection = 1
        else:
            pos_link_connection = 0

        output_arrays_self = []
        output_arrays_other = []
        output_arrays_walker = []

        outputs_for_walkers_at_this_position = [[] for _ in range(len(self.particles_at_this_position))]

        for i, cur_node in enumerate(self.nodes):
            other_walker_at_home = cur_node.other_node.particle.walker_position is cur_node.other_node.particle

            input_array = np.vstack([0.0, 0.0, 1.0,
                                     self.information,
                                     cur_node.other_node.particle.information,
                                     self.walker_position.information,
                                     cur_node.other_node.particle.walker_position.information,
                                     walker_at_home,
                                     other_walker_at_home,
                                     pos_link_connection
                                     ])
            output_arrays_self.append(self.behavior.get_output(input_array=input_array, sim_options=sim_options))

            if sim_options.get(SimOptionsEnum.ONLY_USE_OWN_BEHAVIOR) == False:
                input_array[0] = 1.0
                output_arrays_other.append(cur_node.other_node.particle.behavior.get_output(input_array=input_array, sim_options=sim_options))
            if sim_options.get(SimOptionsEnum.CAN_DECLINE_NEW_LINK) == True or sim_options.get(SimOptionsEnum.ONLY_USE_OWN_BEHAVIOR) == False:
                input_array[1] = 1.0
                output_arrays_walker.append(self.walker_position.behavior.get_output(input_array=input_array, sim_options=sim_options))

            input_array[0] = 0.0
            for j, cur_par in enumerate(self.particles_at_this_position):
                if sim_options.get(SimOptionsEnum.MOVE_ON_OWN_BY_SELF):
                    ref_particle = self
                else:
                    ref_particle = self.particles_at_this_position[j]
                if sim_options.get(SimOptionsEnum.MOVE_ON_OWN, particle=ref_particle):
                    outputs_for_walkers_at_this_position[j].append(cur_par.behavior.get_output(input_array=input_array, sim_options=sim_options))

        output_arrays_self = np.hstack(output_arrays_self)
        if len(output_arrays_other) > 0:
            output_arrays_other = np.hstack(output_arrays_other)
        if len(output_arrays_walker) > 0:
            output_arrays_walker = np.hstack(output_arrays_walker)


        for j, cur_par in enumerate(self.particles_at_this_position):
            if sim_options.get(SimOptionsEnum.MOVE_ON_OWN_BY_SELF):
                ref_particle = self
            else:
                ref_particle = self.particles_at_this_position[j]
            if sim_options.get(SimOptionsEnum.MOVE_ON_OWN, particle=ref_particle):
                outputs_for_walkers_at_this_position[j] = np.hstack(outputs_for_walkers_at_this_position[j])

        index = 0

        # Token Allocation in Game Phase and not here
        index += 2 # Token Allocation


        # Move particles
        for i, cur_particle in enumerate(self.particles_at_this_position):
            if sim_options.get(SimOptionsEnum.MOVE_ON_OWN_BY_SELF):
                ref_particle = self
            else:
                ref_particle = self.particles_at_this_position[i]
            if sim_options.get(SimOptionsEnum.MOVE_ON_OWN, particle=ref_particle):

                choices = [cur_particle, self, self.walker_position, np.random.choice(all_particles), ] + [cur_node.other_node.particle for cur_node in self.nodes] + [cur_node.other_node.particle.walker_position for cur_node in self.nodes]

                home_option = np.mean(outputs_for_walkers_at_this_position[i][index, :])
                self_option = np.mean(outputs_for_walkers_at_this_position[i][index+1, :])
                walker_option = np.mean(outputs_for_walkers_at_this_position[i][index + 2, :])
                random_option = np.mean(outputs_for_walkers_at_this_position[i][index + 3, :])
                neighbor_option = outputs_for_walkers_at_this_position[i][index + 4, :]
                neighbor_walker_option = outputs_for_walkers_at_this_position[i][index + 5, :]

                if not sim_options.get(SimOptionsEnum.CAN_MOVE_HOME):
                    home_option = 0.0
                if not sim_options.get(SimOptionsEnum.CAN_NOT_MOVE):
                    self_option = 0.0
                if not sim_options.get(SimOptionsEnum.CAN_MOVE_TO_WALKERS_WALKER):
                    walker_option = 0.0
                if not sim_options.get(SimOptionsEnum.CAN_MOVE_TO_RANDOM):
                    random_option = 0.0
                if not sim_options.get(SimOptionsEnum.CAN_MOVE_TO_NEIGBHBOR):
                    neighbor_option *= 0.0
                if not sim_options.get(SimOptionsEnum.CAN_MOVE_TO_NEIGBHBORS_WALKER):
                    neighbor_walker_option *= 0.0
                probability = np.hstack(
                    [home_option, self_option, walker_option, random_option, neighbor_option, neighbor_walker_option])
                if probability.sum() == 0.0:
                    home_option = 1.0
                    self_option = 1.0
                    walker_option = 1.0
                    random_option = 1.0
                    neighbor_option = np.ones_like(neighbor_option)
                    neighbor_walker_option = np.ones_like(neighbor_walker_option)
                    if not sim_options.get(SimOptionsEnum.CAN_MOVE_HOME):
                        home_option = 0.0
                    if not sim_options.get(SimOptionsEnum.CAN_NOT_MOVE):
                        self_option = 0.0
                    if not sim_options.get(SimOptionsEnum.CAN_MOVE_TO_WALKERS_WALKER):
                        walker_option = 0.0
                    if not sim_options.get(SimOptionsEnum.CAN_MOVE_TO_RANDOM):
                        random_option = 0.0
                    if not sim_options.get(SimOptionsEnum.CAN_MOVE_TO_NEIGBHBOR):
                        neighbor_option *= 0.0
                    if not sim_options.get(SimOptionsEnum.CAN_MOVE_TO_NEIGBHBORS_WALKER):
                        neighbor_walker_option *= 0.0
                    probability = np.hstack(
                        [home_option, self_option, walker_option, random_option, neighbor_option,
                         neighbor_walker_option])

                choice = self.make_choice(to_choose=choices, p=probability,
                                          choice_type=sim_options.get(SimOptionsEnum.MOVING_PARTICLES_DECISION, particle=self.particles_at_this_position[i]))

                cur_particle.go_particle = choice


            else:
                output_arrays = self.get_correct_output(
                    use_x=sim_options.get(SimOptionsEnum.USE_X_FOR_MOVING_PARTICLES,
                                            particle=self.particles_at_this_position[i]),
                                            output_arrays_self=output_arrays_self,
                                            output_arrays_other=output_arrays_other,
                                            output_arrays_walker=output_arrays_walker)

                choices = [cur_particle, self, self.walker_position, np.random.choice(all_particles), ] + [cur_node.other_node.particle for cur_node in self.nodes] + [cur_node.other_node.particle.walker_position for cur_node in self.nodes]

                home_option = np.mean(output_arrays[index, :])
                self_option = np.mean(output_arrays[index + 1, :])
                walker_option = np.mean(output_arrays[index + 2, :])
                random_option = np.mean(output_arrays[index + 3, :])
                neighbor_option = output_arrays[index + 4, :]
                neighbor_walker_option = output_arrays[index + 5, :]

                if not sim_options.get(SimOptionsEnum.CAN_MOVE_HOME):
                    home_option = 0.0
                if not sim_options.get(SimOptionsEnum.CAN_NOT_MOVE):
                    self_option = 0.0
                if not sim_options.get(SimOptionsEnum.CAN_MOVE_TO_WALKERS_WALKER):
                    walker_option = 0.0
                if not sim_options.get(SimOptionsEnum.CAN_MOVE_TO_RANDOM):
                    random_option = 0.0
                if not sim_options.get(SimOptionsEnum.CAN_MOVE_TO_NEIGBHBOR):
                    neighbor_option *= 0.0
                if not sim_options.get(SimOptionsEnum.CAN_MOVE_TO_NEIGBHBORS_WALKER):
                    neighbor_walker_option *= 0.0
                probability = np.hstack(
                    [home_option, self_option, walker_option, random_option, neighbor_option, neighbor_walker_option])
                if probability.sum() == 0.0:
                    home_option = 1.0
                    self_option = 1.0
                    walker_option = 1.0
                    random_option = 1.0
                    neighbor_option = np.ones_like(neighbor_option)
                    neighbor_walker_option = np.ones_like(neighbor_walker_option)
                    if not sim_options.get(SimOptionsEnum.CAN_MOVE_HOME):
                        home_option = 0.0
                    if not sim_options.get(SimOptionsEnum.CAN_NOT_MOVE):
                        self_option = 0.0
                    if not sim_options.get(SimOptionsEnum.CAN_MOVE_TO_WALKERS_WALKER):
                        walker_option = 0.0
                    if not sim_options.get(SimOptionsEnum.CAN_MOVE_TO_RANDOM):
                        random_option = 0.0
                    if not sim_options.get(SimOptionsEnum.CAN_MOVE_TO_NEIGBHBOR):
                        neighbor_option *= 0.0
                    if not sim_options.get(SimOptionsEnum.CAN_MOVE_TO_NEIGBHBORS_WALKER):
                        neighbor_walker_option *= 0.0
                    probability = np.hstack(
                        [home_option, self_option, walker_option, random_option, neighbor_option,
                         neighbor_walker_option])

                choice = self.make_choice(to_choose=choices, p=probability,
                                          choice_type=sim_options.get(SimOptionsEnum.MOVING_PARTICLES_DECISION,
                                                                      particle=self.particles_at_this_position[i]))
                cur_particle.go_particle = choice

        index += 6



        # Decide How much percent to reproduce at self
        output_arrays = self.get_correct_output(
            use_x=sim_options.get(SimOptionsEnum.USE_X_FOR_KEEPING_REPRO_TOKEN, particle=self),
            output_arrays_self=output_arrays_self,
            output_arrays_other=output_arrays_other,
            output_arrays_walker=output_arrays_walker)
        token_ref = np.sum(output_arrays[index, :])
        repro_token_ref = np.sum(output_arrays[index + 1, :])
        denominator = repro_token_ref + token_ref
        if denominator > 0.0:
            repro_percentage = repro_token_ref / denominator
            repro_tokens = int(np.round(repro_percentage * self.token))
            self.reproduction_tokens += repro_tokens
            self.token -= repro_tokens
            if self.token == 0 and sim_options.get(SimOptionsEnum.KEEPS_ONE_TOKEN_WHEN_REPRODUCTION):
                self.token += 1
                self.reproduction_tokens -= 1
            data.kept_repro_tokens_history[-1] += self.reproduction_tokens


        index += 2

        # Make new link Decision
        output_arrays = self.get_correct_output(use_x=sim_options.get(SimOptionsEnum.USE_X_FOR_NEW_LINK, particle=self),
                                                output_arrays_self=output_arrays_self,
                                                output_arrays_other=output_arrays_other,
                                                output_arrays_walker=output_arrays_walker)
        make_new_link_sum = np.sum(output_arrays[index, :])
        make_no_link_sum = np.sum(output_arrays[index + 1, :])
        to_choose = [True, False]
        choice = self.make_choice(to_choose=to_choose, p=[make_new_link_sum, make_no_link_sum],
                                  choice_type=sim_options.get(SimOptionsEnum.NEW_LINK_DECISION, particle=self))
        self.make_new_link = choice
        index += 2

        # Accept new link
        if sim_options.get(SimOptionsEnum.CAN_DECLINE_NEW_LINK):
            output_arrays = self.get_correct_output(use_x=sim_options.get(SimOptionsEnum.USE_X_FOR_ACCEPTING_LINK, particle=self.walker_position),
                                                    output_arrays_self=output_arrays_self,
                                                    output_arrays_other=output_arrays_other,
                                                    output_arrays_walker=output_arrays_walker)
            accept_sum = np.sum(output_arrays[index, :])
            decline_sum = np.sum(output_arrays[index + 1, :])
            to_choose = [True, False]
            choice = self.make_choice(to_choose=to_choose, p=[accept_sum, decline_sum],
                                      choice_type=sim_options.get(SimOptionsEnum.ACCEPT_NEW_LINK_DECISION, particle=self.walker_position))
            self.can_new_link = choice
        index += 2

        # Swapping
        output_arrays = self.get_correct_output(use_x=sim_options.get(SimOptionsEnum.USE_X_FOR_SWAPPING, particle=self),
                                                output_arrays_self=output_arrays_self,
                                                output_arrays_other=output_arrays_other,
                                                output_arrays_walker=output_arrays_walker)
        swapping_outputs = output_arrays[index, :]
        no_swapping_outputs = output_arrays[index+1, :]
        if np.sum(swapping_outputs) > np.sum(no_swapping_outputs):
            choice = self.make_choice(to_choose=self.nodes, p=swapping_outputs,
                                      choice_type=sim_options.get(SimOptionsEnum.SWAPPING_DECISION, particle=self))
            choice.is_swapping = True
        index += 2

        # Reconnection
        output_arrays = self.get_correct_output(
                    use_x=sim_options.get(SimOptionsEnum.USE_X_FOR_RECONNECTION, particle=self),
                    output_arrays_self=output_arrays_self,
                    output_arrays_other=output_arrays_other,
                    output_arrays_walker=output_arrays_walker)
        reconnection_outputs = output_arrays[index, :]
        no_reconnection_outputs = output_arrays[index + 1, :]
        if np.sum(reconnection_outputs) > np.sum(no_reconnection_outputs):
            choice = self.make_choice(to_choose=self.nodes, p=reconnection_outputs,
                                      choice_type=sim_options.get(SimOptionsEnum.RECONNECTION_DECISION, particle=self))
            self.reconnection_node = choice
        index += 2

        # Reconnection Direction
        output_arrays = self.get_correct_output(
                use_x=sim_options.get(SimOptionsEnum.USE_X_FOR_RECONNECTION_DIRECTION, particle=self),
                output_arrays_self=output_arrays_self,
                output_arrays_other=output_arrays_other,
                output_arrays_walker=output_arrays_walker)
        reconnection_direction_outputs_self = output_arrays[index, :]
        reconnection_direction_outputs = output_arrays[index+1, :]

        p = [np.mean(reconnection_direction_outputs_self)] + list(reconnection_direction_outputs)
        choices = [self] + [cur_node.other_node.particle for cur_node in self.nodes]
        choice = self.make_choice(to_choose=choices, p=p,
                                  choice_type=sim_options.get(SimOptionsEnum.RECONNECTION_DIRECTION_DECISION, particle=self))
        self.reconnection_particle = choice


        # Shifting Decision
        output_arrays = self.get_correct_output(use_x=sim_options.get(SimOptionsEnum.USE_X_FOR_SHIFTING, particle=self),
                                                output_arrays_self=output_arrays_self,
                                                output_arrays_other=output_arrays_other,
                                                output_arrays_walker=output_arrays_walker)
        shifting1_outputs = output_arrays[index, :]
        shifting2_outputs = output_arrays[index + 1, :]

        for i in range(len(self.nodes)):
            p = [shifting1_outputs[i], shifting2_outputs[i]]
            to_choose = [True, False]
            choice = self.make_choice(to_choose=to_choose, p=p, choice_type=sim_options.get(SimOptionsEnum.SHIFTING_DECISION, particle=self))
            self.nodes[i].is_shifting = choice
        index += 2


        # Plant or not
        output_arrays = self.get_correct_output(use_x=sim_options.get(SimOptionsEnum.USE_X_FOR_PLANTING, particle=self),
                                                output_arrays_self=output_arrays_self,
                                                output_arrays_other=output_arrays_other,
                                                output_arrays_walker=output_arrays_walker)
        to_choose = [True, False]
        choice = self.make_choice(to_choose=to_choose, p=[np.sum(output_arrays[index, :]), np.sum(output_arrays[index + 1, :])],
                                  choice_type=sim_options.get(SimOptionsEnum.PLANTING_DECISION, particle=self))
        self.reproduce_at_home = choice
        if not sim_options.get(SimOptionsEnum.CAN_REPRODUCE_AT_HOME):
            self.reproduce_at_home = False
        index += 2


        # Plant on which
        output_arrays = self.get_correct_output(use_x=sim_options.get(SimOptionsEnum.USE_X_FOR_PLANTING_PARTICLE, particle=self),
                                                output_arrays_self=output_arrays_self,
                                                output_arrays_other=output_arrays_other,
                                                output_arrays_walker=output_arrays_walker)
        choices = [self, self.walker_position, np.random.choice(all_particles)] + [cur_node.other_node.particle for cur_node in self.nodes] + [cur_node.other_node.particle.walker_position for cur_node in self.nodes]
        self_option = np.mean(output_arrays[index, :])
        walker_option = np.mean(output_arrays[index + 1, :])
        random_option = np.mean(output_arrays[index + 2, :])
        neighbor_option = output_arrays[index + 3, :]
        neighbor_walker_option = output_arrays[index + 4, :]

        if not sim_options.get(SimOptionsEnum.CAN_PLANT_ON_SELF):
            self_option = 0.0
        if not sim_options.get(SimOptionsEnum.CAN_PLANT_ON_WALKER):
            walker_option = 0.0
        if not sim_options.get(SimOptionsEnum.CAN_PLANT_ON_RANDOM):
            random_option = 0.0
        if not sim_options.get(SimOptionsEnum.CAN_PLANT_ON_NEIGHBOR):
            neighbor_option *= 0.0
        if not sim_options.get(SimOptionsEnum.CAN_PLANT_ON_NEIGBHBORS_WALKER):
            neighbor_walker_option *= 0.0
        probability = np.hstack([self_option, walker_option, random_option, neighbor_option, neighbor_walker_option])
        if probability.sum() == 0.0:
            self_option = 1.0
            walker_option = 1.0
            random_option = 1.0
            neighbor_option = np.ones_like(neighbor_option)
            neighbor_walker_option = np.ones_like(neighbor_walker_option)
            if not sim_options.get(SimOptionsEnum.CAN_PLANT_ON_SELF):
                self_option = 0.0
            if not sim_options.get(SimOptionsEnum.CAN_PLANT_ON_WALKER):
                walker_option = 0.0
            if not sim_options.get(SimOptionsEnum.CAN_PLANT_ON_RANDOM):
                random_option = 0.0
            if not sim_options.get(SimOptionsEnum.CAN_PLANT_ON_NEIGHBOR):
                neighbor_option *= 0.0
            if not sim_options.get(SimOptionsEnum.CAN_PLANT_ON_NEIGBHBORS_WALKER):
                neighbor_walker_option *= 0.0
            probability = np.hstack([self_option, walker_option, random_option, neighbor_option, neighbor_walker_option])

        choice = self.make_choice(to_choose=choices, p=probability,
                         choice_type=sim_options.get(SimOptionsEnum.PLANTING_PARTICLE_DECISION, particle=self))
        self.plant_particle = choice
        index += 5

        # Walker Position Inheritage Decision
        output_arrays = self.get_correct_output(use_x=sim_options.get(SimOptionsEnum.USE_X_FOR_INHERITAGING_POSITION, particle=self),
                                                output_arrays_self=output_arrays_self,
                                                output_arrays_other=output_arrays_other,
                                                output_arrays_walker=output_arrays_walker)
        inherit_sum = np.sum(output_arrays[index, :])
        dont_inherit_sum = np.sum(output_arrays[index + 1, :])
        to_choose = [True, False]
        choice = self.make_choice(to_choose=to_choose, p=[inherit_sum, dont_inherit_sum],
                                  choice_type=sim_options.get(SimOptionsEnum.INHERITAGING_POSITION_DECISION, particle=self))
        self.inherit_walker_position = choice
        index += 2


        # Get particles to connect the newly reproduced particle to
        output_arrays = output_arrays_self  #TODO only self option available
        self_yes = np.sum(output_arrays[index, :])
        self_no = np.sum(output_arrays[index+1, :])
        walker_yes = np.sum(output_arrays[index+2, :])
        walker_no = np.sum(output_arrays[index+3, :])
        neighbor_yes = output_arrays[index + 4, :]
        neighbor_no = output_arrays[index + 5, :]



        to_choose = [True, False]
        choice_self = self.make_choice(to_choose=to_choose, p=[self_yes, self_no],
                                  choice_type=sim_options.get(SimOptionsEnum.REPRODUCTION_LINKS_DECISION, particle=self))
        if choice_self:
            self.link_creation_particles_for_reproduction.append(self)
        choice_walker = self.make_choice(to_choose=to_choose, p=[walker_yes, walker_no],
                                  choice_type=sim_options.get(SimOptionsEnum.REPRODUCTION_LINKS_DECISION, particle=self))
        if choice_walker:
            self.link_creation_particles_for_reproduction.append(self.walker_position)

        for i, cur_node in enumerate(self.nodes):
            choice_neighbor = self.make_choice(to_choose=to_choose, p=[neighbor_yes[i], neighbor_no[i]],
                                           choice_type=sim_options.get(SimOptionsEnum.REPRODUCTION_LINKS_DECISION, particle=self))
            if choice_neighbor:
                self.link_creation_particles_for_reproduction.append(cur_node.other_node.particle)
        self.link_creation_particles_for_reproduction = list(set(self.link_creation_particles_for_reproduction))
        index += 6

    @staticmethod
    def get_correct_output(use_x, output_arrays_self, output_arrays_other, output_arrays_walker):
        """
        Returns the output defined by use_x.
        :param use_x: Which output array should be used? (= SELF, OTHER, POSITION)
        :param output_arrays_self: output array of own behavior
        :param output_arrays_other: output array of neighbouring behavior
        :param output_arrays_walker: output array of position behavior
        :return: The output array defined by use_x
        """
        if use_x == "SELF":
            return output_arrays_self
        elif use_x == "NEIGHBOR":
            return output_arrays_other
        else:   # WALKER
            return output_arrays_walker

    @staticmethod
    def make_choice(to_choose, p, choice_type):
        """
        Chooses an Element of an array according to weights and choice type.
        :param to_choose: Array of elements to choose from.
        :param p: Weights for the elements of the to_choose array
        :param choice_type: How should an element be chosen? (= MAX, PROB, RANDOM)
        :return: Element of to_choose with weights p and selected choice type
        """
        p = np.array(p)
        p.resize(p.size)
        if choice_type == "MAX":
            index = int(np.argmax(p))
            return to_choose[index]
        elif choice_type == "PROB":
            sump = np.sum(p)
            if sump == 0.0:
                p = np.ones(p.shape)
                sump = np.sum(p)
            return np.random.choice(to_choose, p=p/sump)
        else:   # RANDOM
            return np.random.choice(to_choose)  #TODO this is wrong, should ony choose random for non zero, but doesnt work anyway

    def reproduce(self, sim_options, pos, all_particles, all_links, data, vel, indexer_f, logger):
        """
        Reproduces particle. First determine the reproduction particles according to settings and chosen actions by
        particle. Then decides how to pay for the cost of the new link. If enough tokens are present, reproduction
        happens. Either the new particle gets planted or an umbilical cord is created to the "self" particle
        (planted or reproduce_at_home).
        :param sim_options: Options of Simulation
        :param pos: Position Array for visualization
        :param all_links: Array of all links of simulation
        :param all_particles: Array of all particles of simulation
        :param data: Data for analysis
        """
        if self.reproduction_tokens == 0:
            return
        elif np.isclose(self.reproduction_tokens, 0):
            logger.warning("WARNING, shouldnt happen, repro token error")

        new_behavior = Behavior(sim_options=sim_options, parent_behavior=self.behavior, indexer_f=indexer_f, data=data)
        new_particle = Particle(token=self.reproduction_tokens, behavior=new_behavior)
        new_particle.to_mutate = True

        if self.inherit_walker_position and sim_options.get(SimOptionsEnum.CAN_INHERIT_WALKER_POSITION):
            new_particle.move_to_particle(self.walker_position)

        self.reproduction_tokens = 0
        all_particles.append(new_particle)

        if sim_options.get(SimOptionsEnum.NEW_REPRODUCTION_LINK_PRODUCTION):
            data.reproduced_particles_history[-1] += 1.0

            for cur_link_par in self.link_creation_particles_for_reproduction:
                new_link = Link(particle1=cur_link_par, particle2=new_particle)

                if pos is not None:
                    pos[new_particle] = pos[self].copy() * np.random.normal(1, 0.001)
                    vel[new_particle] = vel[self].copy()

                if pos is not None:
                    new_link.node1.vis_pos = pos[self].copy()
                    new_link.node2.vis_pos = pos[new_particle].copy()

                if sim_options.get(SimOptionsEnum.CAN_STILL_SHIFT_AT_NEW_REPRODUCTION):
                    for cur_node in self.nodes:
                        if cur_node.is_shifting:
                            cur_node.switch_to_particle(particle=new_particle)

                all_links.append(new_link)

        else:
            if self.reproduce_at_home or not sim_options.get(SimOptionsEnum.CAN_PLANT):
                data.reproduced_particles_history[-1] += 1.0

                if pos is not None:
                    pos[new_particle] = pos[self].copy() * np.random.normal(1, 0.001)
                    vel[new_particle] = vel[self].copy()

                for cur_node in self.nodes:
                    if cur_node.is_shifting:
                        cur_node.switch_to_particle(particle=new_particle)

                new_link = Link(particle1=self, particle2=new_particle)
                if pos is not None:
                    new_link.node1.vis_pos = pos[self].copy()
                    new_link.node2.vis_pos = pos[new_particle].copy()

                all_links.append(new_link)
            else:   # Plant
                data.planted_particles_history[-1] += 1.0

                if pos is not None:
                    pos[new_particle] = pos[self.plant_particle].copy() * np.random.normal(1, 0.001)
                    vel[new_particle] = np.zeros(3)

                new_link = Link(particle1=self.plant_particle, particle2=new_particle)
                if pos is not None:
                    new_link.node1.vis_pos = pos[self].copy()
                    new_link.node2.vis_pos = pos[new_particle].copy()

                all_links.append(new_link)

    def check_death(self, sim_options, all_links, all_particles, data, dead_particles, dead_links):
        """
        If particle has no tokens anymore, it vanishes. The particle that inherits the links is chosen by the heritage
        ratings of the particle.
        :param sim_options: Options of Simulation
        :param all_links: Array of all links of simulation
        :param all_particles: Array of all particles of simulation
        :param data: Data for analysis
        """
        if self.token == 0:
            data.died_particles_history[-1] += 1.0
            for cur_node in self.nodes.copy():
                cur_node.link.kill_link(sim_options=sim_options, all_links=all_links, dead_links=dead_links)

            self.walker_position.particles_at_this_position.remove(self)

            for cur_par in self.particles_at_this_position.copy():
                if sim_options.get(SimOptionsEnum.MOVE_RANDOMLY_WHEN_DEATH):
                    if len(self.nodes) == 0:
                        cur_par.move_to_particle(cur_par)
                    else:
                        choice = np.random.choice(self.nodes)
                        cur_par.move_to_particle(choice.other_node.particle)

                elif sim_options.get(SimOptionsEnum.CAN_MOVE_WHEN_DEATH):

                    if cur_par.go_particle is self or len(cur_par.go_particle.nodes) == 0:
                        cur_par.move_to_particle(cur_par)
                    else:
                        cur_par.move_to_particle(cur_par.go_particle)
                else:
                    cur_par.move_to_particle(cur_par)

            all_particles.remove(self)
            dead_particles.append(self)


    def move_to_particle(self, particle):
        """
        Moves the reference/position Particle to another particle
        :param particle: Move reference/position to this particle
        """
        self.walker_position.particles_at_this_position.remove(self)
        self.walker_position = particle
        self.walker_position.particles_at_this_position.append(self)

    def try_making_new_link(self, sim_options, all_links, data, pos):

        if self.make_new_link and self.walker_position is not self:
            for cur_node in self.walker_position.nodes:
                if cur_node.other_node.particle is self:
                    return  # walker position is already connected because it is a neighbor
            # if reached -> not connected
            if not self.can_new_link:
                data.declined_new_links_history[-1] += 1.0
                return  # new link has been declined

            data.new_links_spawned_history[-1] += 1.0
            new_link = Link(particle1=self, particle2=self.walker_position)
            if pos is not None:
                new_link.node1.vis_pos = pos[self]
                new_link.node2.vis_pos = pos[self]
            all_links.append(new_link)
            return

    def move(self):
        """
        Moves reference to new particle. If particle chose to go home it creates a new link between "self" and reference
        particle. This is only possible if the cost for the new link can be paid and if this move has been accepted.
        :param sim_options: Options of Simulation
        :param all_links: Array of all links of simulation
        :param data: Data for analysis
        """
        self.move_to_particle(self.go_particle)

    def get_dimension_list(self, all_particles):
        """
        Gets the dimensionality curve as described in the paper: flooding the network starting at "self" particle.
        :param all_particles: Array of all particles of simulation
        :return: Dimensionality array of network for each step.
        """
        particles_reached = [self]
        for cur_node in self.nodes:
            particles_reached.append(cur_node.other_node.particle)
        reached = [0, len(particles_reached)]
        dims = []
        steps = [1]
        while len(particles_reached) > reached[-2]:

            for cur_par in particles_reached.copy():
                for cur_node in cur_par.nodes:
                    particles_reached.append(cur_node.other_node.particle)
            particles_reached = list(set(particles_reached))
            reached.append(len(particles_reached))
            steps.append(steps[-1] + 1)
            dims.append((np.log(reached[-1]) - np.log(reached[-2]))/(np.log(steps[-1]) - np.log(steps[-2])))
        return np.array(dims)
