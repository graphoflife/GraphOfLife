# IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
import datetime as datetime
import os
import pickle
matplotlib.use('Agg')

"""
We present a new algorithm that explores the evolution 
of complex and organic networks through the behavior of autonomous 
particles, which have properties of living creatures. The algorithm 
utilizes different feedforward neural networks to govern the behavior 
of individual particles, which are linked together to form a graph. 
These particles interact with their neighbors and compete for scarce 
fungible tokens to survive and reproduce. Over time, natural selection 
sorts out fragile behaviors while promoting the growth of antifragile 
ones. The algorithm allows for a wide range of settings, leading to 
diverse incentive structures and macroscopic structures. Through 
testing various combinations of settings, we were able to observe 
the emergence of autonomous, decentralized, and three-dimensional 
networks which are always evolving
"""

# Setting options
LOADING_FOLDER = None   # "to_simulate"
LOADING_FILE = None
RANDOM_SETTINGS = False
EXPERIMENT_SERIES = False
ITERATION_AMOUNT = 1200

# General simulation settings
INIT_PARTICLE_AMOUNT = 50
TOKEN_AMOUNT = 5000

# Plotting options
ANGLE_SHIFT = 2 * np.pi / 180
SMOOTH_PLOTS = 3
SPRING_MODEL_ITERATIONS = 10
PLOT_EVERY_X_ITERATIONS = 1

# Analysis
CHECK_X_PARTICLES_FOR_DIM = 30


class SimOptions:
    """
    The options of the simulation.
    """

    USE_X_FOR_GOING_HOME = "OTHER"          # SELF, OTHER, POSITION
    USE_X_FOR_MOVING_PARTICLES = "POSITION" # SELF, OTHER, POSITION
    USE_X_FOR_DEATH_RATING = "POSITION"     # SELF, OTHER, POSITION
    USE_X_FOR_SWAPPING = "OTHER"            # SELF, OTHER, POSITION
    USE_X_FOR_SHIFTING = "SELF"             # SELF, OTHER, POSITION
    USE_X_FOR_RPS = "SELF"                  # SELF, OTHER, POSITION
    USE_X_FOR_TOKEN_ALLOCATION = "SELF"     # SELF, OTHER, POSITION
    USE_X_FOR_KEEP_TOKENS = "SELF"          # SELF, OTHER, POSITION
    USE_X_FOR_PLANTING = "POSITION"         # SELF, OTHER, POSITION
    USE_X_FOR_X_REPRODUCTION = "SELF"       # SELF, OTHER, POSITION
    USE_X_FOR_PLANTING_PARTICLE = "SELF"    # SELF, OTHER, POSITION
    USE_X_FOR_ACCEPTING_MOVE = "SELF"       # SELF, OTHER, POSITION

    REPRODUCE_X = "POSITION"                # SELF, OTHER, POSITION
    PLANT_ON_X = "POSITION"                 # SELF, OTHER, POSITION

    GOING_HOME_DECISION = "MAX"         # MAX, PROB, RANDOM
    MOVING_PARTICLES_DECISION = "MAX"   # MAX, PROB, RANDOM
    DEATH_RATING_DECISION = "PROB"      # MAX, PROB, RANDOM
    SWAPPING_DECISION = "MAX"           # MAX, PROB, RANDOM
    SHIFTING_DECISION = "MAX"           # MAX, PROB, RANDOM
    RPS_DECISION = "PROB"               # MAX, PROB, RANDOM
    PLANTING_DECISION = "MAX"           # MAX, PROB, RANDOM
    X_REPRODUCTION_DECISION = "MAX"     # MAX, PROB, RANDOM
    PLANTING_PARTICLE_DECISION = "MAX"  # MAX, PROB, RANDOM
    ACCEPTING_MOVE_DECISION = "RANDOM"  # MAX, PROB, RANDOM

    DRAW_REPRODUCES = True
    DRAW_FORWARDS = True
    ALWAYS_DRAW = True

    WIN_LOSER_REPRODUCES = False     # Matters only if WIN_WINS_EVERYTHING = False
    WIN_WINS_EVERYTHING = False
    WIN_DISTRIBUTION_TYPE = "BR"    # S"elf"R"eproduction", O"ther"R"eproduction", B"oth"T"oken", B"oth"R"eproduction"

    TAKE_TOKEN_FROM_X_FOR_MOVE = "BR"   # B"oth"T"oken", B"oth"R"eproduction", T"oken", R"eproduction"
    CAN_USE_LAST_TOKEN_FOR_MOVE = True

    MOVE_ON_OWN = False
    SHIFT_OTHER = True
    CAN_SWAP = True
    SWAP_SWAPS_TOKEN = False
    ADD_RANDOM_VALUES = True
    CAN_PLANT = True
    CAN_KEEP_TOKENS = False
    SINGLE_RPS = True
    USE_ALL_TOKEN_FOR_REPRODUCTION = False
    MUTATE_WHEN_TOKEN_NOT_ENOUGH = True
    REPRO_USES_TOKEN_FOR_LINK = True
    SEE_EVERYTHING = True
    CAN_DECLINE_MOVE = True
    MOVE_HOME_IF_DECLINED = True
    ACCEPT_MOVE_HOME_ON_OWN = True

    MUTATION_CHANCE = 0.75

    # Neural Network
    INPUT_EXPONENT = -1
    QUANTILE_AMOUNT = 3
    INPUT_AMOUNT = 39 + QUANTILE_AMOUNT * 3 * 4
    OUTPUT_AMOUNT = 19
    LAYER_SIZES = [int(INPUT_AMOUNT * 1.3)] * 2
    SEE = np.vstack([1] * INPUT_AMOUNT)

    def __init__(self, random_settings: bool):

        if random_settings:
            self.set_random_settings()
            p = np.array([1.0, 1.0, 1.0, 1.0])
            self.MUTATION_CHANCE = np.random.choice([0.25, 0.5, 0.75, 1.0], p=p/p.sum())

    def set_random_settings(self):
        """
        Sets the settings randomly according to preset probabilities.
        """

        p = np.array([1.0, 1.0, 1.0])
        self.USE_X_FOR_GOING_HOME = np.random.choice(["SELF", "OTHER", "POSITION"], p=p / p.sum())
        p = np.array([1.0, 1.0, 1.0])
        self.USE_X_FOR_MOVING_PARTICLES = np.random.choice(["SELF", "OTHER", "POSITION"], p=p / p.sum())
        p = np.array([1.0, 1.0, 1.0])
        self.USE_X_FOR_DEATH_RATING = np.random.choice(["SELF", "OTHER", "POSITION"], p=p / p.sum())
        p = np.array([1.0, 1.0, 1.0])
        self.USE_X_FOR_SWAPPING = np.random.choice(["SELF", "OTHER", "POSITION"], p=p / p.sum())
        p = np.array([1.0, 1.0, 1.0])
        self.USE_X_FOR_SHIFTING = np.random.choice(["SELF", "OTHER", "POSITION"], p=p / p.sum())
        p = np.array([1.0, 1.0, 1.0])
        self.USE_X_FOR_RPS = np.random.choice(["SELF", "OTHER", "POSITION"], p=p / p.sum())
        p = np.array([1.0, 1.0, 1.0])
        self.USE_X_FOR_TOKEN_ALLOCATION = np.random.choice(["SELF", "OTHER", "POSITION"], p=p / p.sum())
        p = np.array([1.0, 1.0, 1.0])
        self.USE_X_FOR_KEEP_TOKENS = np.random.choice(["SELF", "OTHER", "POSITION"], p=p / p.sum())
        p = np.array([1.0, 1.0, 1.0])
        self.USE_X_FOR_PLANTING = np.random.choice(["SELF", "OTHER", "POSITION"], p=p / p.sum())
        p = np.array([1.0, 1.0, 1.0])
        self.USE_X_FOR_X_REPRODUCTION = np.random.choice(["SELF", "OTHER", "POSITION"], p=p / p.sum())
        p = np.array([1.0, 1.0, 1.0])
        self.USE_X_FOR_PLANTING_PARTICLE = np.random.choice(["SELF", "OTHER", "POSITION"], p=p / p.sum())
        p = np.array([1.0, 1.0, 1.0])
        self.USE_X_FOR_ACCEPTING_MOVE = np.random.choice(["SELF", "OTHER", "POSITION"], p=p / p.sum())

        p = np.array([1.0, 1.0, 1.0])
        self.REPRODUCE_X = np.random.choice(["SELF", "OTHER", "POSITION"], p=p / p.sum())
        p = np.array([1.0, 1.0, 1.0])
        self.PLANT_ON_X = np.random.choice(["SELF", "OTHER", "POSITION"], p=p / p.sum())

        p = np.array([3.0, 1.0, 1.0])
        self.GOING_HOME_DECISION = np.random.choice(["MAX", "PROB", "RANDOM"], p=p / p.sum())
        p = np.array([2.0, 2.0, 1.0])
        self.MOVING_PARTICLES_DECISION = np.random.choice(["MAX", "PROB", "RANDOM"], p=p / p.sum())
        p = np.array([1.0, 1.0, 1.0])
        self.DEATH_RATING_DECISION = np.random.choice(["MAX", "PROB", "RANDOM"], p=p / p.sum())
        p = np.array([3.0, 1.0, 1.0])
        self.SWAPPING_DECISION = np.random.choice(["MAX", "PROB", "RANDOM"], p=p / p.sum())
        p = np.array([3.0, 1.0, 1.0])
        self.SHIFTING_DECISION = np.random.choice(["MAX", "PROB", "RANDOM"], p=p / p.sum())
        p = np.array([3.0, 2.0, 1.0])
        self.RPS_DECISION = np.random.choice(["MAX", "PROB", "RANDOM"], p=p / p.sum())
        p = np.array([3.0, 2.0, 1.0])
        self.PLANTING_DECISION = np.random.choice(["MAX", "PROB", "RANDOM"], p=p / p.sum())
        p = np.array([3.0, 1.0, 1.0])
        self.X_REPRODUCTION_DECISION = np.random.choice(["MAX", "PROB", "RANDOM"], p=p / p.sum())
        p = np.array([3.0, 2.0, 1.0])
        self.PLANTING_PARTICLE_DECISION = np.random.choice(["MAX", "PROB", "RANDOM"], p=p / p.sum())
        p = np.array([3.0, 2.0, 1.0])
        self.ACCEPTING_MOVE_DECISION = np.random.choice(["MAX", "PROB", "RANDOM"], p=p / p.sum())

        p = np.array([1.0, 1.0])
        self.DRAW_REPRODUCES = np.random.choice([True, False], p=p / p.sum())
        p = np.array([1.0, 1.0])
        self.DRAW_FORWARDS = np.random.choice([True, False], p=p / p.sum())
        p = np.array([1.0, 3.0])
        self.ALWAYS_DRAW = np.random.choice([True, False], p=p / p.sum())

        p = np.array([1.0, 1.0])
        self.WIN_LOSER_REPRODUCES = np.random.choice([True, False], p=p / p.sum())
        p = np.array([1.0, 1.0])
        self.WIN_WINS_EVERYTHING = np.random.choice([True, False], p=p / p.sum())
        p = np.array([1.0, 1.0, 1.0, 1.0])
        self.WIN_DISTRIBUTION_TYPE = np.random.choice(["SR", "OR", "BT", "BR"], p=p / p.sum())

        p = np.array([1.0, 1.0, 1.0, 1.0])
        self.TAKE_TOKEN_FROM_X_FOR_MOVE = np.random.choice(["BT", "BR", "T", "R"], p=p / p.sum())
        p = np.array([1.0, 1.0])
        self.CAN_USE_LAST_TOKEN_FOR_MOVE = np.random.choice([True, False], p=p / p.sum())

        p = np.array([1.0, 1.0])
        self.MOVE_ON_OWN = np.random.choice([True, False], p=p / p.sum())
        p = np.array([1.0, 1.0])
        self.SHIFT_OTHER = np.random.choice([True, False], p=p / p.sum())
        p = np.array([1.0, 1.0])
        self.CAN_SWAP = np.random.choice([True, False], p=p / p.sum())
        p = np.array([1.0, 1.0])
        self.SWAP_SWAPS_TOKEN = np.random.choice([True, False], p=p / p.sum())
        p = np.array([1.0, 1.0])
        self.ADD_RANDOM_VALUES = np.random.choice([True, False], p=p / p.sum())
        p = np.array([1.0, 1.0])
        self.CAN_PLANT = np.random.choice([True, False], p=p / p.sum())
        p = np.array([1.0, 1.0])
        self.CAN_KEEP_TOKENS = np.random.choice([True, False], p=p / p.sum())
        p = np.array([1.0, 1.0])
        self.SINGLE_RPS = np.random.choice([True, False], p=p / p.sum())
        p = np.array([1.0, 3.0])
        self.USE_ALL_TOKEN_FOR_REPRODUCTION = np.random.choice([True, False], p=p / p.sum())
        p = np.array([1.0, 1.0])
        self.MUTATE_WHEN_TOKEN_NOT_ENOUGH = np.random.choice([True, False], p=p / p.sum())
        p = np.array([1.0, 1.0])
        self.REPRO_USES_TOKEN_FOR_LINK = np.random.choice([True, False], p=p / p.sum())
        p = np.array([3.0, 1.0])
        self.SEE_EVERYTHING = np.random.choice([True, False], p=p / p.sum())
        p = np.array([1.0, 1.0])
        self.CAN_DECLINE_MOVE = np.random.choice([True, False], p=p / p.sum())
        p = np.array([1.0, 1.0])
        self.MOVE_HOME_IF_DECLINED = np.random.choice([True, False], p=p / p.sum())
        p = np.array([1.0, 1.0])
        self.ACCEPT_MOVE_HOME_ON_OWN = np.random.choice([True, False], p=p / p.sum())

        p = 0.75
        self.SEE = np.vstack([int(np.random.random() < p) for _ in range(self.INPUT_AMOUNT)])

    def get_description(self):
        """
        :return: Get a string that contains all the settings.
        """
        see_text = ""
        for cur in self.SEE:
            see_text += f"{cur[0]} "

        description = f"M = {self.MUTATION_CHANCE}, T = {TOKEN_AMOUNT}, QA = {self.QUANTILE_AMOUNT}, " \
                      f"E={self.INPUT_EXPONENT}, L={[self.INPUT_AMOUNT] + self.LAYER_SIZES + [self.OUTPUT_AMOUNT]}\n" \
                      f"XGH={self.USE_X_FOR_GOING_HOME},XMP={self.USE_X_FOR_MOVING_PARTICLES}," \
                      f"XDR={self.USE_X_FOR_DEATH_RATING},XSW={self.USE_X_FOR_SWAPPING}," \
                      f"XSH={self.USE_X_FOR_SHIFTING},XRSP={self.USE_X_FOR_RPS}," \
                      f"XTA={self.USE_X_FOR_TOKEN_ALLOCATION},XKT={self.USE_X_FOR_KEEP_TOKENS}," \
                      f"XPL={self.USE_X_FOR_PLANTING},XXR={self.USE_X_FOR_X_REPRODUCTION}," \
                      f"XPP={self.USE_X_FOR_PLANTING_PARTICLE},XAM={self.USE_X_FOR_ACCEPTING_MOVE}," \
                      f"RX={self.REPRODUCE_X},PX={self.PLANT_ON_X}\n" \
                      f"GHD={self.GOING_HOME_DECISION},MPD={self.MOVING_PARTICLES_DECISION}," \
                      f"DRD={self.DEATH_RATING_DECISION},SWD={self.SWAPPING_DECISION}," \
                      f"SHD={self.SHIFTING_DECISION},RPSD={self.RPS_DECISION},PD={self.PLANTING_DECISION}," \
                      f"XRD={self.X_REPRODUCTION_DECISION}," \
                      f"PPD={self.PLANTING_PARTICLE_DECISION},AMD={self.ACCEPTING_MOVE_DECISION}," \
                      f"DR={self.DRAW_REPRODUCES},DF={self.DRAW_FORWARDS},AD={self.ALWAYS_DRAW}," \
                      f"WLR={self.WIN_LOSER_REPRODUCES},WWE={self.WIN_WINS_EVERYTHING}," \
                      f"WDT={self.WIN_DISTRIBUTION_TYPE}\n" \
                      f"TTM={self.TAKE_TOKEN_FROM_X_FOR_MOVE},CULT={self.CAN_USE_LAST_TOKEN_FOR_MOVE}," \
                      f"MOO={self.MOVE_ON_OWN},SO={self.SHIFT_OTHER},CS={self.CAN_SWAP},SST={self.SWAP_SWAPS_TOKEN}," \
                      f"RV={self.ADD_RANDOM_VALUES},CP={self.CAN_PLANT}," \
                      f"CKT={self.CAN_KEEP_TOKENS},SRPS={self.SINGLE_RPS},USTR={self.USE_ALL_TOKEN_FOR_REPRODUCTION}," \
                      f"MWTNE={self.MUTATE_WHEN_TOKEN_NOT_ENOUGH}," \
                      f"RUTL={self.REPRO_USES_TOKEN_FOR_LINK},SE={self.SEE_EVERYTHING},CDM={self.CAN_DECLINE_MOVE}," \
                      f"MHD={self.MOVE_HOME_IF_DECLINED},AMHOO={self.ACCEPT_MOVE_HOME_ON_OWN}\nS=[ {see_text}]" \


        return description


class Behavior:
    """
    Neural network with a rectified linear activation function that belongs to a particle.
    """
    weights: [np.array] = None
    biases: [np.array] = None

    def __init__(self, sim_options, parent_behavior=None):

        self.weights = []
        self.biases = []

        if parent_behavior is None:
            # Random weights and biases
            sizes = [sim_options.INPUT_AMOUNT] + sim_options.LAYER_SIZES + [sim_options.OUTPUT_AMOUNT]
            for i in range(len(sizes) - 1):
                columns = int(sizes[i])
                rows = int(sizes[i + 1])
                self.weights.append(np.random.normal(0.0, 1.0, (rows, columns)))
                self.biases.append(np.random.normal(0.0, 1.0, (rows, 1)))
        else:
            # Inheritance from parent behavior
            for i in range(len(parent_behavior.weights)):
                self.weights.append(parent_behavior.weights[i].copy())
                self.biases.append(parent_behavior.biases[i].copy())

    def get_output(self, input_array: np.array):
        """
        Feeds input vector through neural network and gets the output.
        Uses rectified linear activation function.

        :param input_array: must have shape input_amount x 1
        :return: output_array with shape output_amount x 1
        """
        output = input_array
        for i in range(len(self.weights)):
            output = np.maximum(np.matmul(self.weights[i], output) + self.biases[i], 0)
        return output

    def mutate(self, sim_options):
        """
        Mutate Values of weights and biases with probability MUTATION_CHANCE
        :param sim_options: Options of Simulation
        """
        for i in range(len(self.weights)):
            self.weights[i] += np.random.normal(0, 1, self.weights[i].shape) \
                * (np.random.uniform(0, 1, self.weights[i].shape) < sim_options.MUTATION_CHANCE)
            self.biases[i] += np.random.normal(0, 1, self.biases[i].shape) \
                * (np.random.uniform(0, 1, self.biases[i].shape) < sim_options.MUTATION_CHANCE)


class LinkNode:
    """
    The intermediary between link and particle.
    Holds some information about the state of the network and about decisions of the particle.
    """
    # Connection references
    link = None
    other_node = None
    particle = None

    is_shifting = None
    is_swapping = None
    played_move: int = None
    information = None
    allocated_token: int = None
    heritage_rating: float = None
    did_win: float = None

    def initialize(self, particle, link, other_node):
        """
        Manual initialization
        :param particle: Reference to particle of node
        :param link: Reference to link of node
        :param other_node: Reference of the other node of this link
        """
        self.link = link
        self.other_node = other_node
        self.connect(particle)
        self.reset_info()
        self.allocated_token = 0

    def disconnect(self):
        """
        Disconnects this node of its particle
        """
        self.particle.nodes.remove(self)
        self.particle = None

    def connect(self, particle):
        """
        Connects this node to a particle
        :param particle: Particle to connect to
        """
        self.particle = particle
        self.particle.nodes.append(self)

    def switch_to_particle(self, particle):
        """
        Disconnects of current particle and connect to new particle.
        :param particle: New particle to connect to.
        """
        self.disconnect()
        self.connect(particle=particle)
        self.reset_info()
        self.other_node.reset_info()
        self.link.reset_info()

    def prepare(self, sim_options):
        """
        Prepares information for next iteration.
        :param sim_options: Options of Simulation.
        """
        self.information = np.vstack([0, 0, 0])
        if not sim_options.SINGLE_RPS:
            if self.played_move == 0:
                self.information = np.vstack([1, 0, 0])
            elif self.played_move == 1:
                self.information = np.vstack([0, 1, 0])
            elif self.played_move == 2:
                self.information = np.vstack([0, 0, 1])
            else:
                self.information = np.vstack([0, 0, 0])

    def reset_info(self):
        """
        Resets the info if node or other_node has reconnected and information no longer represents the last interaction.
        """
        self.played_move = -1
        self.did_win = 0.0
        self.heritage_rating = 0.0


class Link:
    """
    The link between particles
    """

    node1: LinkNode = None
    node2: LinkNode = None
    token: int = None

    win_loss = None

    def __init__(self, particle1, particle2, token):
        self.node1 = LinkNode()
        self.node2 = LinkNode()
        self.node1.initialize(particle=particle1, link=self, other_node=self.node2)
        self.node2.initialize(particle=particle2, link=self, other_node=self.node1)
        self.token = token
        self.reset_info()

    def reset_info(self):
        """
        Resets the info if node1 or node2 has reconnected and information no longer represents the last interaction.
        """
        self.win_loss = 0.0

    def prepare(self, sim_options):
        """
        Prepares information for next iteration.
        :param sim_options: Options of Simulation
        """
        self.node1.prepare(sim_options=sim_options)
        self.node2.prepare(sim_options=sim_options)

    def eval_game(self, sim_options, draw_matrix, win_lose_matrix, data):
        """
        Evaluates the game of rock paper scissors and distributes allocated tokens according to the game result.
        :param sim_options: Options of Simulation
        :param draw_matrix: Matrix that governs the token distribution in case of a draw
        :param win_lose_matrix: Matrix that governs the token distribution in case of a win/lose
        :param data: Data for analysis
        """
        draw = False
        winner_node = None
        loser_node = None

        if sim_options.SINGLE_RPS:
            acting1 = self.node1.particle
            acting2 = self.node2.particle
        else:
            acting1 = self.node1
            acting2 = self.node2

        crit1 = acting1.played_move == 0 and acting2.played_move == 2
        crit2 = acting1.played_move == 1 and acting2.played_move == 0
        crit3 = acting1.played_move == 2 and acting2.played_move == 1
        if acting1.played_move == acting2.played_move:
            draw = True
        elif crit1 or crit2 or crit3:
            winner_node = self.node1
            loser_node = self.node2
        else:
            winner_node = self.node2
            loser_node = self.node1

        if draw or sim_options.ALWAYS_DRAW:
            data.draw_history[-1] += 1.0/data.link_amount_history[-1]
            input_token = np.array([self.node1.allocated_token, self.node2.allocated_token])
            result = np.matmul(draw_matrix, input_token)
            self.node1.particle.token += result[0]
            self.node1.particle.reproduction_tokens += result[1]
            self.node2.particle.token += result[2]
            self.node2.particle.reproduction_tokens += result[3]
            self.node1.did_win = (result[0] + result[1]) > self.node1.allocated_token
            self.node2.did_win = (result[2] + result[3]) > self.node2.allocated_token
            self.win_loss = abs(result[0] + result[1] - self.node1.allocated_token)
        else:
            pot_win = min(winner_node.allocated_token, loser_node.allocated_token)
            input_token = np.array([winner_node.allocated_token, loser_node.allocated_token, pot_win])
            result = np.matmul(win_lose_matrix, input_token)
            winner_node.particle.token += result[0]
            winner_node.particle.reproduction_tokens += result[1]
            loser_node.particle.token += result[2]
            loser_node.particle.reproduction_tokens += result[3]
            winner_node.did_win = (result[0] + result[1]) > winner_node.allocated_token
            loser_node.did_win = (result[2] + result[3]) > loser_node.allocated_token

    def try_swap(self, sim_options, data):
        """
        If there is mutual consent of both particles, a swap happens.
        :param sim_options: Options of Simulation
        :param data: Data for analysis
        """
        if sim_options.CAN_SWAP and self.node1.is_swapping and self.node2.is_swapping:

            data.swap_percentage_history[-1] += 2.0/data.particle_amount_history[-1]

            self.node1.particle.behavior, self.node2.particle.behavior = self.node2.particle.behavior, \
                                                                         self.node1.particle.behavior
            if sim_options.SWAP_SWAPS_TOKEN:
                self.node1.particle.token, self.node2.particle.token = self.node2.particle.token, \
                                                                       self.node1.particle.token

    def kill_link(self, surviving_particle, all_links, all_particles):
        """
        Kills the link and shifts all the links of the vanishing particle to the surviving particle.
        :param surviving_particle: The particle that survives when this link dies.
        :param all_links: all_links for reference
        :param all_particles: all_particles for reference
        """
        if surviving_particle is self.node1.particle:
            dying_particle = self.node2.particle
        else:
            dying_particle = self.node1.particle

        for cur_par in dying_particle.particles_at_this_position.copy():
            cur_par.move_to_particle(surviving_particle)

        surviving_particle.token += dying_particle.token
        surviving_particle.token += self.token
        dying_particle.token = 0
        dying_particle.position_particle.particles_at_this_position.remove(dying_particle)
        self.token = 0
        self.node1.disconnect()
        self.node2.disconnect()
        all_links.remove(self)
        all_particles.remove(dying_particle)
        already_connected_particles = [cur_node.other_node.particle for cur_node in surviving_particle.nodes]
        already_connected_links = [cur_node.link for cur_node in surviving_particle.nodes]
        for cur_node in dying_particle.nodes.copy():
            cur_particle = cur_node.other_node.particle
            if cur_particle in already_connected_particles:
                surviving_particle.token += cur_node.link.token
                cur_node.link.token = 0
                cur_node.other_node.disconnect()
                cur_node.disconnect()
                all_links.remove(cur_node.link)
            else:
                already_connected_particles.append(cur_particle)
                already_connected_links.append(cur_node.link)
                cur_node.switch_to_particle(surviving_particle)


class Particle:
    """
    The autonomous particle that is the core of the simulation.
    """

    nodes: [LinkNode] = None
    token = None
    position_particle = None
    particles_at_this_position: [] = None
    behavior: Behavior = None
    information = None
    go_home = None
    can_go_home = None
    go_particle = None
    reproduction_tokens = None
    reproduction_behavior = None
    age = None
    keep_tokens_per = None
    used_tokens_to_reproduce_per = None
    reproduce_at_home = None
    plant_particle = None
    played_move: int = None

    def __init__(self, token, behavior):
        self.nodes = []
        self.token = token
        self.position_particle = self
        self.particles_at_this_position = [self]
        self.behavior = behavior
        self.age = 0
        self.played_move = -1
        self.can_go_home = True

    def link_amount(self):
        """
        :return: Gets the amount of links this particle has.
        """
        return len(self.nodes)

    def prepare(self, sim_options):
        """
        Prepares information for next iteration. This includes assembling a information vector about the state of
        this particle with some quantile information about its neighbors.
        :param sim_options: Options of Simulation
        """
        self.go_particle = None
        self.reproduction_tokens = 0
        self.age += 1
        np.random.shuffle(self.nodes)

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

        token_quantiles = np.array(token_quantiles).reshape(sim_options.QUANTILE_AMOUNT, 1)**sim_options.INPUT_EXPONENT
        link_quantiles = np.array(link_quantiles).reshape(sim_options.QUANTILE_AMOUNT, 1)**sim_options.INPUT_EXPONENT
        particle_quantiles = np.array(particle_quantiles).reshape(sim_options.QUANTILE_AMOUNT, 1)**sim_options.INPUT_EXPONENT

        # Rock, Paper, Scissors Information
        rps_info = np.vstack([0, 0, 0])
        if sim_options.SINGLE_RPS:
            if self.played_move == 0:
                rps_info = np.vstack([1, 0, 0])
            elif self.played_move == 1:
                rps_info = np.vstack([0, 1, 0])
            elif self.played_move == 2:
                rps_info = np.vstack([0, 0, 1])
            else:
                rps_info = np.vstack([0, 0, 0])

        self.information = np.vstack([self.token**sim_options.INPUT_EXPONENT,
                                      self.link_amount()**sim_options.INPUT_EXPONENT,
                                      (len(self.particles_at_this_position)+1.0)**sim_options.INPUT_EXPONENT,
                                      rps_info, token_quantiles, link_quantiles, particle_quantiles])

    def get_link_to_particle(self, to_particle):
        """
        :param to_particle: The link to this particle is wanted.
        :return: Returns the link that connects this particle to another particle.
        """
        for cur_node in self.nodes:
            if cur_node.other_node.particle is to_particle:
                return cur_node.link
        return None

    def process_information_and_make_decisions(self, sim_options, data):
        """
        Assembles input vector for each node, feeds input through own, other and position behaviors and makes decisions
        according to the outputs of this process.
        :param sim_options: Options of Simulation
        :param data: Data for analysis
        """

        at_home = self.position_particle is self
        pos_link = self.get_link_to_particle(to_particle=self.position_particle)
        if pos_link:
            pos_link_token = pos_link.token**sim_options.INPUT_EXPONENT
        else:
            pos_link_token = 0

        output_arrays_self = []
        output_arrays_other = []
        output_arrays_position = []

        outputs_for_particles_at_this_position = [[] for _ in range(len(self.particles_at_this_position))]

        for i, cur_node in enumerate(self.nodes):
            if sim_options.ADD_RANDOM_VALUES:
                add_values = np.random.random((3, 1))
            else:
                add_values = np.zeros((3, 1))
            other_at_home = cur_node.other_node.particle.position_particle is cur_node.other_node.particle

            input_array = np.vstack([self.information,
                                     cur_node.other_node.particle.information,
                                     self.position_particle.information,
                                     cur_node.other_node.particle.position_particle.information,
                                     at_home,
                                     other_at_home,
                                     cur_node.link.token**sim_options.INPUT_EXPONENT,
                                     cur_node.did_win,
                                     (cur_node.link.win_loss+1)**sim_options.INPUT_EXPONENT,
                                     pos_link_token,
                                     cur_node.information,
                                     cur_node.other_node.information,
                                     add_values
                                     ])

            if not sim_options.SEE_EVERYTHING:
                input_array = input_array * sim_options.SEE

            output_arrays_self.append(self.behavior.get_output(input_array=input_array))
            output_arrays_other.append(cur_node.other_node.particle.behavior.get_output(input_array=input_array))
            output_arrays_position.append(self.position_particle.behavior.get_output(input_array=input_array))

            if sim_options.MOVE_ON_OWN or sim_options.ACCEPT_MOVE_HOME_ON_OWN:
                for j, cur_par in enumerate(self.particles_at_this_position):
                    outputs_for_particles_at_this_position[j].append(cur_par.behavior.get_output(input_array=input_array))

        output_arrays_self = np.hstack(output_arrays_self)
        output_arrays_other = np.hstack(output_arrays_other)
        output_arrays_position = np.hstack(output_arrays_position)

        if sim_options.MOVE_ON_OWN or sim_options.ACCEPT_MOVE_HOME_ON_OWN:
            for i, cur_par in enumerate(self.particles_at_this_position):
                outputs_for_particles_at_this_position[i] = np.hstack(outputs_for_particles_at_this_position[i])

        # 1 Go Home Decision
        output_arrays = self.get_correct_output(use_x=sim_options.USE_X_FOR_GOING_HOME,
                                                output_arrays_self=output_arrays_self,
                                                output_arrays_other=output_arrays_other,
                                                output_arrays_position=output_arrays_position)
        keep_going_sum = np.sum(output_arrays[0, :])
        come_home_sum = np.sum(output_arrays[1, :])
        to_choose = [True, False]
        choice = self.make_choice(to_choose=to_choose, p=[keep_going_sum, come_home_sum],
                                  choice_type=sim_options.GOING_HOME_DECISION)
        self.go_home = choice

        # 2 Accept Move
        if sim_options.CAN_DECLINE_MOVE:
            output_arrays = self.get_correct_output(use_x=sim_options.USE_X_FOR_ACCEPTING_MOVE,
                                                    output_arrays_self=output_arrays_self,
                                                    output_arrays_other=output_arrays_other,
                                                    output_arrays_position=output_arrays_position)
            if sim_options.ACCEPT_MOVE_HOME_ON_OWN:

                for i, cur_particle in enumerate(self.particles_at_this_position):
                    output_arrays = outputs_for_particles_at_this_position[i]
                    accept_sum = np.sum(output_arrays[2, :])
                    decline_sum = np.sum(output_arrays[3, :])
                    to_choose = [True, False]
                    choice = self.make_choice(to_choose=to_choose, p=[accept_sum, decline_sum],
                                              choice_type=sim_options.ACCEPTING_MOVE_DECISION)
                    cur_particle.can_go_home = choice

            else:
                accept_sum = np.sum(output_arrays[2, :])
                decline_sum = np.sum(output_arrays[3, :])
                to_choose = [True, False]
                choice = self.make_choice(to_choose=to_choose, p=[accept_sum, decline_sum],
                                          choice_type=sim_options.ACCEPTING_MOVE_DECISION)
                for cur_particle in self.particles_at_this_position:
                    cur_particle.can_go_home = choice

        # 3 Move particles
        output_arrays = self.get_correct_output(use_x=sim_options.USE_X_FOR_MOVING_PARTICLES,
                                                output_arrays_self=output_arrays_self,
                                                output_arrays_other=output_arrays_other,
                                                output_arrays_position=output_arrays_position)

        for i, cur_particle in enumerate(self.particles_at_this_position):
            if sim_options.MOVE_ON_OWN:
                choice = self.make_choice(to_choose=self.nodes, p=outputs_for_particles_at_this_position[i][4, :],
                                          choice_type=sim_options.MOVING_PARTICLES_DECISION)
                cur_particle.go_particle = choice.other_node.particle
            else:
                choice = self.make_choice(to_choose=self.nodes, p=output_arrays[4, :],
                                          choice_type=sim_options.MOVING_PARTICLES_DECISION)
                cur_particle.go_particle = choice.other_node.particle

        # 4 Dying direction
        output_arrays = self.get_correct_output(use_x=sim_options.USE_X_FOR_DEATH_RATING,
                                                output_arrays_self=output_arrays_self,
                                                output_arrays_other=output_arrays_other,
                                                output_arrays_position=output_arrays_position)
        heritage_outputs = output_arrays[5, :]
        for i, cur_node in enumerate(self.nodes):
            cur_node.heritage_rating = heritage_outputs[i]

        # 5 Swapping
        output_arrays = self.get_correct_output(use_x=sim_options.USE_X_FOR_SWAPPING,
                                                output_arrays_self=output_arrays_self,
                                                output_arrays_other=output_arrays_other,
                                                output_arrays_position=output_arrays_position)
        swapping_outputs = output_arrays[6, :]
        for cur_node in self.nodes:
            cur_node.is_swapping = False
        if np.sum(swapping_outputs) > 0.0:
            choice = self.make_choice(to_choose=self.nodes, p=swapping_outputs,
                                      choice_type=sim_options.SWAPPING_DECISION)
            choice.is_swapping = True

        # 6 Shifting Decision
        output_arrays = self.get_correct_output(use_x=sim_options.USE_X_FOR_SHIFTING,
                                                output_arrays_self=output_arrays_self,
                                                output_arrays_other=output_arrays_other,
                                                output_arrays_position=output_arrays_position)
        shifting1_outputs = output_arrays[7, :]
        shifting2_outputs = output_arrays[8, :]

        for i in range(len(self.nodes)):
            p = [shifting1_outputs[i], shifting2_outputs[i]]
            to_choose = [True, False]
            choice = self.make_choice(to_choose=to_choose, p=p, choice_type=sim_options.SHIFTING_DECISION)
            self.nodes[i].is_shifting = choice

        # 7 Rock paper scissors
        output_arrays = self.get_correct_output(use_x=sim_options.USE_X_FOR_RPS,
                                                output_arrays_self=output_arrays_self,
                                                output_arrays_other=output_arrays_other,
                                                output_arrays_position=output_arrays_position)
        rps_outputs = output_arrays[[9, 10, 11], :]
        to_choose = [0, 1, 2]
        rps_arrays = [data.r_history, data.p_history, data.s_history]
        if sim_options.SINGLE_RPS:
            p = np.sum(rps_outputs, axis=1)
            choice = self.make_choice(to_choose=to_choose, p=p, choice_type=sim_options.RPS_DECISION)
            if sim_options.ALWAYS_DRAW:
                choice = 0
            rps_arrays[choice][-1] += 1.0 / data.particle_amount_history[-1]

            data.RPS_data_array.append(choice)
            self.played_move = choice
        else:
            for i, cur_node in enumerate(self.nodes):
                p = rps_outputs[:, i]
                choice = self.make_choice(to_choose=to_choose, p=p, choice_type=sim_options.RPS_DECISION)
                if sim_options.ALWAYS_DRAW:
                    choice = 0
                rps_arrays[choice][-1] += 1.0 / (data.link_amount_history[-1]*2)
                data.RPS_data_array.append(choice)
                cur_node.played_move = choice

        # 8 Token allocation
        output_arrays = self.get_correct_output(use_x=sim_options.USE_X_FOR_TOKEN_ALLOCATION,
                                                output_arrays_self=output_arrays_self,
                                                output_arrays_other=output_arrays_other,
                                                output_arrays_position=output_arrays_position)
        token_outputs = output_arrays[12, :]
        sumx = np.sum(token_outputs)
        if sumx == 0.0:
            token_outputs = np.random.random(token_outputs.size)
        token_outputs_percent = token_outputs / token_outputs.sum()
        token_outputs_exact = token_outputs_percent * self.token
        token_outputs = np.round(token_outputs_exact)
        token_outputs_diff = token_outputs_exact - token_outputs
        # shape = token_outputs.shape
        for i in range(max(int(np.sum(token_outputs) - self.token), 0)):
            index = np.argmin(token_outputs_diff)
            token_outputs_diff[index] = 0.0
            token_outputs[index] -= 1
        for i in range(max(int(self.token - np.sum(token_outputs)), 0)):
            index = np.argmax(token_outputs_diff)
            token_outputs_diff[index] = 0.0
            token_outputs[index] += 1
        for i, cur_node in enumerate(self.nodes):
            cur_node.allocated_token = token_outputs[i]
        self.token = 0

        # 9 Keep tokens
        output_arrays = self.get_correct_output(use_x=sim_options.USE_X_FOR_KEEP_TOKENS,
                                                output_arrays_self=output_arrays_self,
                                                output_arrays_other=output_arrays_other,
                                                output_arrays_position=output_arrays_position)
        self.keep_tokens_per = np.sum(output_arrays[13, :])
        self.used_tokens_to_reproduce_per = np.sum(output_arrays[14, :])

        # 10 Plant or not
        output_arrays = self.get_correct_output(use_x=sim_options.USE_X_FOR_PLANTING,
                                                output_arrays_self=output_arrays_self,
                                                output_arrays_other=output_arrays_other,
                                                output_arrays_position=output_arrays_position)
        to_choose = [True, False]
        choice = self.make_choice(to_choose=to_choose, p=[np.sum(output_arrays[15, :]), np.sum(output_arrays[16, :])],
                                  choice_type=sim_options.PLANTING_DECISION)
        self.reproduce_at_home = choice
        if not sim_options.CAN_PLANT:
            self.reproduce_at_home = True

        # 11 Reproduce which
        output_arrays = self.get_correct_output(use_x=sim_options.USE_X_FOR_X_REPRODUCTION,
                                                output_arrays_self=output_arrays_self,
                                                output_arrays_other=output_arrays_other,
                                                output_arrays_position=output_arrays_position)
        if sim_options.REPRODUCE_X == "SELF":
            self.reproduction_behavior = self.behavior
        elif sim_options.REPRODUCE_X == "OTHER":
            choice = self.make_choice(to_choose=self.nodes, p=output_arrays[17, :],
                                      choice_type=sim_options.X_REPRODUCTION_DECISION)
            self.reproduction_behavior = choice.other_node.particle.behavior
        else:
            self.reproduction_behavior = self.position_particle.behavior

        # 12 Plant on which
        output_arrays = self.get_correct_output(use_x=sim_options.USE_X_FOR_PLANTING_PARTICLE,
                                                output_arrays_self=output_arrays_self,
                                                output_arrays_other=output_arrays_other,
                                                output_arrays_position=output_arrays_position)
        if sim_options.PLANT_ON_X == "SELF":
            self.plant_particle = self
        if sim_options.PLANT_ON_X == "OTHER":
            choice = self.make_choice(to_choose=self.nodes, p=output_arrays[18, :],
                                      choice_type=sim_options.PLANTING_PARTICLE_DECISION)
            self.plant_particle = choice.other_node.particle
        else:
            self.plant_particle = self.position_particle

    @staticmethod
    def get_correct_output(use_x, output_arrays_self, output_arrays_other, output_arrays_position):
        """
        Returns the output defined by use_x.
        :param use_x: Which output array should be used? (= SELF, OTHER, POSITION)
        :param output_arrays_self: output array of own behavior
        :param output_arrays_other: output array of neighbouring behavior
        :param output_arrays_position: output array of position behavior
        :return: The output array defined by use_x
        """
        if use_x == "SELF":
            return output_arrays_self
        elif use_x == "OTHER":
            return output_arrays_other
        else:   # POSITION
            return output_arrays_position

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
            return np.random.choice(to_choose)

    def reproduce(self, sim_options, pos, all_links, all_particles, data):
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
        if sim_options.USE_ALL_TOKEN_FOR_REPRODUCTION:
            self.reproduction_tokens += self.token
            self.token = 0

        if not sim_options.CAN_KEEP_TOKENS:
            self.keep_tokens_per = 0.0
            self.used_tokens_to_reproduce_per = 1.0

        if self.keep_tokens_per + self.used_tokens_to_reproduce_per == 0:
            self.keep_tokens_per = np.random.random()
            self.used_tokens_to_reproduce_per = 1.0 - self.keep_tokens_per

        keep_tokens_per = self.keep_tokens_per / (self.keep_tokens_per + self.used_tokens_to_reproduce_per)
        keep_tokens = int(round(self.reproduction_tokens * keep_tokens_per))
        data.kept_tokens_history[-1] += keep_tokens_per/data.particle_amount_history[-1]

        self.reproduction_tokens -= keep_tokens
        self.token += keep_tokens

        if self.reproduction_tokens == 0:
            return
        elif self.reproduction_tokens == 1:

            if self.token > 0 and sim_options.REPRO_USES_TOKEN_FOR_LINK:
                # Token for Link Creation
                self.token -= 1
                self.reproduction_tokens += 1
            else:
                self.token += self.reproduction_tokens
                self.reproduction_tokens = 0
                if sim_options.MUTATE_WHEN_TOKEN_NOT_ENOUGH:
                    self.behavior.mutate(sim_options=sim_options)
                return
        elif self.reproduction_tokens > 1:
            if self.token > 0 and sim_options.REPRO_USES_TOKEN_FOR_LINK:
                # Token for Link Creation
                self.token -= 1
                self.reproduction_tokens += 1

        if self.reproduce_at_home:
            data.reproduced_particles_history[-1] += 1.0/data.particle_amount_history[-1]
            new_behavior = Behavior(sim_options=sim_options, parent_behavior=self.reproduction_behavior)
            new_behavior.mutate(sim_options=sim_options)
            new_particle = Particle(token=self.reproduction_tokens-1,
                                    behavior=new_behavior)
            self.reproduction_tokens = 0
            if pos is not None:
                pos[new_particle] = pos[self].copy() * np.random.normal(1, 0.001)
            for cur_node in self.nodes:
                shift = cur_node.is_shifting
                if sim_options.SHIFT_OTHER:
                    shift = cur_node.other_node.is_shifting
                if shift:
                    cur_node.switch_to_particle(particle=new_particle)

            new_link = Link(particle1=self, particle2=new_particle, token=1)
            all_particles.append(new_particle)
            all_links.append(new_link)
        else:   # Plant
            data.planted_particles_history[-1] += 1.0/data.particle_amount_history[-1]
            new_behavior = Behavior(sim_options=sim_options, parent_behavior=self.reproduction_behavior)
            new_behavior.mutate(sim_options=sim_options)
            new_particle = Particle(token=self.reproduction_tokens - 1,
                                    behavior=new_behavior)
            self.reproduction_tokens = 0
            if pos is not None:
                pos[new_particle] = pos[self.plant_particle].copy() * np.random.normal(1, 0.001)
            new_link = Link(particle1=self.plant_particle, particle2=new_particle, token=1)
            all_particles.append(new_particle)
            all_links.append(new_link)

    def check_death(self, sim_options, all_links, all_particles, data):
        """
        If particle has no tokens anymore, it vanishes. The particle that inherits the links is chosen by the heritage
        ratings of the particle.
        :param sim_options: Options of Simulation
        :param all_links: Array of all links of simulation
        :param all_particles: Array of all particles of simulation
        :param data: Data for analysis
        """
        if self.token == 0:
            p = [cur_node.heritage_rating for cur_node in self.nodes]
            chosen_node = self.make_choice(to_choose=self.nodes, p=p, choice_type=sim_options.DEATH_RATING_DECISION)
            data.died_particles_history[-1] += 1.0 / data.particle_amount_history[-1]
            chosen_node.link.kill_link(surviving_particle=chosen_node.other_node.particle,
                                       all_links=all_links, all_particles=all_particles)

    def move_to_particle(self, particle):
        """
        Moves the reference/position Particle to another particle
        :param particle: Move reference/position to this particle
        """
        self.position_particle.particles_at_this_position.remove(self)
        self.position_particle = particle
        self.position_particle.particles_at_this_position.append(self)

    def move(self, sim_options, all_links, data):
        """
        Moves reference to new particle. If particle chose to go home it creates a new link between "self" and reference
        particle. This is only possible if the cost for the new link can be paid and if this move has been accepted.
        :param sim_options: Options of Simulation
        :param all_links: Array of all links of simulation
        :param data: Data for analysis
        """
        if self.go_home and self.position_particle is not self:
            for cur_node in self.position_particle.nodes:
                if cur_node.other_node.particle is self:
                    self.move_to_particle(self)
                    return
            # if reached -> not connected
            if not self.can_go_home:
                data.declined_new_links_history[-1] += 1.0/data.particle_amount_history[-1]
                if sim_options.MOVE_HOME_IF_DECLINED:
                    self.move_to_particle(self)
                return

            if sim_options.TAKE_TOKEN_FROM_X_FOR_MOVE in ["BR", "BT"]:
                criterion = self.token + self.reproduction_tokens
            elif sim_options.TAKE_TOKEN_FROM_X_FOR_MOVE == "T":
                criterion = self.token
            else:   # R
                criterion = self.reproduction_tokens

            comparison = int(not sim_options.CAN_USE_LAST_TOKEN_FOR_MOVE)

            if criterion > comparison:

                data.new_links_spawned_history[-1] += 1.0/data.particle_amount_history[-1]
                new_link = Link(particle1=self, particle2=self.position_particle, token=1)
                if sim_options.TAKE_TOKEN_FROM_X_FOR_MOVE == "BR":
                    if self.reproduction_tokens > 0:
                        self.reproduction_tokens -= 1
                    else:
                        self.token -= 1
                elif sim_options.TAKE_TOKEN_FROM_X_FOR_MOVE == "BT":
                    if self.reproduction_tokens > 0:
                        self.reproduction_tokens -= 1
                    else:
                        self.token -= 1
                elif sim_options.TAKE_TOKEN_FROM_X_FOR_MOVE == "T":
                    self.token -= 1
                else:   # R
                    self.reproduction_tokens -= 1

                all_links.append(new_link)
                self.move_to_particle(self)
                return

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
        reached = [len(particles_reached)]
        dims = [0]
        steps = [1]
        while len(particles_reached) < len(all_particles):

            for cur_par in particles_reached.copy():
                for cur_node in cur_par.nodes:
                    particles_reached.append(cur_node.other_node.particle)
            particles_reached = list(set(particles_reached))
            reached.append(len(particles_reached))
            steps.append(steps[-1] + 1)
            dims.append((np.log(reached[-1]) - np.log(reached[-2]))/(np.log(steps[-1]) - np.log(steps[-2])))
        return np.array(dims)


class Data:
    """
    Data container for analysis of the simulation.
    """

    particle_amount_history = None
    link_amount_history = None
    percent_home_history = None
    dim_history = None
    max_link_amount_history = None
    max_par_conc_history = None
    max_token_amount_history = None
    max_token_link_amount_history = None
    max_age_history = None
    RPS_data_array = None

    swap_percentage_history = None
    new_links_spawned_history = None
    declined_new_links_history = None
    planted_particles_history = None
    reproduced_particles_history = None
    links_per_particle_history = None
    died_particles_history = None
    kept_tokens_history = None

    draw_history = None
    r_history = None
    p_history = None
    s_history = None

    def __init__(self):
        """
        Initialize all all data variables
        """
        self.particle_amount_history = []
        self.link_amount_history = []
        self.percent_home_history = []
        self.dim_history = []
        self.max_link_amount_history = 1
        self.max_par_conc_history = 1
        self.max_token_amount_history = 1
        self.max_token_link_amount_history = 1
        self.max_age_history = 1
        self.RPS_data_array = []

        self.dim_stddev_history = []
        self.swap_percentage_history = []
        self.new_links_spawned_history = []
        self.declined_new_links_history = []
        self.planted_particles_history = []
        self.reproduced_particles_history = []
        self.died_particles_history = []
        self.links_per_particle_history = []
        self.kept_tokens_history = []

        self.draw_history = []
        self.r_history = []
        self.p_history = []
        self.s_history = []


class Simulation:
    """
    The main simulation class.
    """

    iterations_until_next_plot: int = 0
    current_plot: int = 0
    current_iteration = None

    sim_options: SimOptions = None
    particles: [Particle] = None
    links: [Link] = None

    name: str = None
    data = None

    # For Plotting
    angle = None
    pos = None

    # For RPS game
    draw_matrix = None
    win_lose_matrix = None

    def __init__(self, sim_options):
        """
        Initialization of the simulation. Create neccesary folders. Create initial particles and links.
        Create evaluation matrix for the game evaluation.
        :param sim_options: Options of Simulation
        """
        self.particles = []
        self.links = []
        self.sim_options = sim_options
        self.data = Data()
        self.current_iteration = 0
        self.angle = 0

        # Create Folder and Save Simulation Options
        self.name = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        parent_dir = os.path.dirname(os.path.realpath(__file__))
        parent_dir = os.path.join(parent_dir, "outputs")
        try:
            os.makedirs(parent_dir)
        except:
            print("outputs folder exists already")
        path = os.path.join(parent_dir, self.name)
        try:
            os.makedirs(path)
        except:
            print(f"{self.name} directory couldnt be made")
        with open(rf"outputs\{self.name}\{self.name}.obj", "wb") as handle_x:
            try:
                pickle.dump(self.sim_options, handle_x)
            except:
                pass

        # Particle and Link Initialization
        init_token_per_link = 1
        init_token_per_particle = int((TOKEN_AMOUNT - init_token_per_link*INIT_PARTICLE_AMOUNT)/INIT_PARTICLE_AMOUNT)
        for i in range(INIT_PARTICLE_AMOUNT):
            new_behavior = Behavior(sim_options=sim_options)
            self.particles.append(Particle(token=init_token_per_particle, behavior=new_behavior))
        if INIT_PARTICLE_AMOUNT == 2:
            self.links.append(Link(particle1=self.particles[0], particle2=self.particles[1], token=init_token_per_link))
        elif INIT_PARTICLE_AMOUNT > 2:
            for i in range(INIT_PARTICLE_AMOUNT):
                self.links.append(Link(particle1=self.particles[i-1], particle2=self.particles[i],
                                       token=init_token_per_link))

        self.init_eval_matrices()

    def init_eval_matrices(self):
        """
        Initializes the token distribution type for the evaluation of the
        rock, paper, scissors game according to simulation settings.
        """

        # Token 1, Winner
        # Reproduction Token 1, Winner
        # Token 2, Loser
        # Reproduction Token 2, Loser

        if self.sim_options.DRAW_REPRODUCES:
            if self.sim_options.DRAW_FORWARDS:
                self.draw_matrix = np.array([[0.0, 0.0],
                                             [0.0, 1.0],
                                             [0.0, 0.0],
                                             [1.0, 0.0]])
            else:
                self.draw_matrix = np.array([[0.0, 0.0],
                                             [1.0, 0.0],
                                             [0.0, 0.0],
                                             [0.0, 1.0]])
        else:
            if self.sim_options.DRAW_FORWARDS:
                self.draw_matrix = np.array([[0.0, 1.0],
                                             [0.0, 0.0],
                                             [1.0, 0.0],
                                             [0.0, 0.0]])
            else:
                self.draw_matrix = np.array([[1.0, 0.0],
                                             [0.0, 0.0],
                                             [0.0, 1.0],
                                             [0.0, 0.0]])

        if self.sim_options.WIN_WINS_EVERYTHING:
            if self.sim_options.WIN_DISTRIBUTION_TYPE == "SR":
                self.win_lose_matrix = np.array([[0.0, 1.0, 0.0],
                                                 [1.0, 0.0, 0.0],
                                                 [0.0, 0.0, 0.0],
                                                 [0.0, 0.0, 0.0]])
            elif self.sim_options.WIN_DISTRIBUTION_TYPE == "OR":
                self.win_lose_matrix = np.array([[1.0, 0.0, 0.0],
                                                 [0.0, 1.0, 0.0],
                                                 [0.0, 0.0, 0.0],
                                                 [0.0, 0.0, 0.0]])
            elif self.sim_options.WIN_DISTRIBUTION_TYPE == "BT":
                self.win_lose_matrix = np.array([[1.0, 1.0, 0.0],
                                                 [0.0, 0.0, 0.0],
                                                 [0.0, 0.0, 0.0],
                                                 [0.0, 0.0, 0.0]])
            else:   # BR
                self.win_lose_matrix = np.array([[0.0, 0.0, 0.0],
                                                 [1.0, 1.0, 0.0],
                                                 [0.0, 0.0, 0.0],
                                                 [0.0, 0.0, 0.0]])
        else:
            if self.sim_options.WIN_DISTRIBUTION_TYPE == "SR":
                win_lose_matrix_upper = np.array([[0.0, 0.0, 1.0],
                                                  [1.0, 0.0, 0.0]])
            elif self.sim_options.WIN_DISTRIBUTION_TYPE == "OR":
                win_lose_matrix_upper = np.array([[1.0, 0.0, 0.0],
                                                  [0.0, 0.0, 1.0]])
            elif self.sim_options.WIN_DISTRIBUTION_TYPE == "BT":
                win_lose_matrix_upper = np.array([[1.0, 0.0, 1.0],
                                                  [0.0, 0.0, 0.0]])
            else:  # BR
                win_lose_matrix_upper = np.array([[0.0, 0.0, 0.0],
                                                  [1.0, 0.0, 1.0]])

            if self.sim_options.WIN_LOSER_REPRODUCES:
                win_lose_matrix_lower = np.array([[0.0, 0.0, 0.0],
                                                  [0.0, 1.0, -1.0]])
            else:
                win_lose_matrix_lower = np.array([[0.0, 1.0, -1.0],
                                                  [0.0, 0.0, 0.0]])

            self.win_lose_matrix = np.vstack([win_lose_matrix_upper, win_lose_matrix_lower])

    def run_main_loop(self, iterations=1e15):
        """
        The main loop of the simulation.
        :param iterations: The amount of iterations that are executed
        """

        self.iterations_until_next_plot = PLOT_EVERY_X_ITERATIONS + 1
        # Main loop
        while iterations > 0:

            # 1 Iteration preparation
            self.prepare()
            for cur_link in self.links:
                cur_link.prepare(sim_options=self.sim_options)
            for cur_par in self.particles:
                cur_par.prepare(sim_options=self.sim_options)

            # 2 Decision making
            for cur_par in self.particles:
                cur_par.process_information_and_make_decisions(sim_options=self.sim_options, data=self.data)

            # 3 Game Evaluation
            for cur_link in self.links:
                cur_link.eval_game(sim_options=self.sim_options, draw_matrix=self.draw_matrix,
                                   win_lose_matrix=self.win_lose_matrix, data=self.data)

            # 4 Swap
            for cur_link in self.links:
                cur_link.try_swap(sim_options=self.sim_options, data=self.data)

            # 5 Move
            for cur_par in self.particles:
                cur_par.move(sim_options=self.sim_options, all_links=self.links, data=self.data)

            # 6 Reproduction
            for cur_par in self.particles.copy():
                cur_par.reproduce(sim_options=self.sim_options, pos=self.pos,
                                  all_particles=self.particles, all_links=self.links, data=self.data)

            # 7 Check Death
            for cur_par in self.particles.copy():
                cur_par.check_death(sim_options=self.sim_options, all_links=self.links,
                                    all_particles=self.particles, data=self.data)

            # Plotting
            self.plot_network(background_color="black")

            self.current_iteration += 1
            iterations -= 1

        # self.plot_network(background_color="white")

    def prepare(self):
        """
        Prepare iteration information for data analysis
        """

        # Randomize the order of execution so that no systemic advantages arise
        np.random.shuffle(self.particles)
        np.random.shuffle(self.links)

        # Prepare data arrays for analysis
        self.data.RPS_data_array = []
        self.data.particle_amount_history.append(len(self.particles))
        self.data.link_amount_history.append(len(self.links))
        self.data.links_per_particle_history.append(self.data.link_amount_history[-1]/self.data.particle_amount_history[-1])
        sum_x = sum([1 for cur_par in self.particles if cur_par is cur_par.position_particle])
        self.data.percent_home_history.append(sum_x / len(self.particles))
        self.data.swap_percentage_history.append(0.0)
        self.data.new_links_spawned_history.append(0.0)
        self.data.declined_new_links_history.append(0.0)
        self.data.planted_particles_history.append(0.0)
        self.data.reproduced_particles_history.append(0.0)
        self.data.died_particles_history.append(0.0)
        self.data.kept_tokens_history.append(0.0)
        self.data.draw_history.append(0.0)
        self.data.r_history.append(0.0)
        self.data.p_history.append(0.0)
        self.data.s_history.append(0.0)

    def plot_network(self, background_color = "black"):
        """
        Plots the network and data, saves images into the outputs folder.
        :param background_color: "black" or "white"
        """


        self.iterations_until_next_plot -= 1
        if self.iterations_until_next_plot <= 0 or self.current_iteration == 0:
            self.iterations_until_next_plot = PLOT_EVERY_X_ITERATIONS
        else:
            return

        token_amount_particles = sum([cur_par.token for cur_par in self.particles])
        token_amount_links = sum([cur_link.token for cur_link in self.links])
        print("Tokens", token_amount_particles + token_amount_links,
              ", Particles", len(self.particles),
              ", Links", len(self.links),
              ", Iteration", self.current_iteration)

        # Dimensionality calculation
        dims = []
        dims_max = []
        dims_amounts = []
        smooth_radius = 5
        smooth_array = [1.0 / smooth_radius] * smooth_radius
        for cur_par in np.random.choice(self.particles, CHECK_X_PARTICLES_FOR_DIM):
            cur_dims = cur_par.get_dimension_list(all_particles=self.particles)
            smoothed_dims = np.convolve(cur_dims, smooth_array)
            dims_max.append(np.max(smoothed_dims))
            dims_amounts.append(len(smoothed_dims))
            dims.append(smoothed_dims)
        self.data.dims = dims
        self.data.dim_history.append(np.average(dims_max))
        self.data.dim_stddev_history.append(np.std(dims_max))

        g = nx.Graph()
        for cur_par in self.particles:
            g.add_node(cur_par)
        for cur_link in self.links:
            g.add_edge(cur_link.node1.particle, cur_link.node2.particle, weight=1)

        if self.pos is None:
            self.pos = nx.fruchterman_reingold_layout(g, pos=self.pos, scale=1, iterations=100, dim=3)
        new_pos = nx.fruchterman_reingold_layout(g, pos=self.pos, scale=1, iterations=SPRING_MODEL_ITERATIONS, dim=3)

        differences = {}
        for part, coord in new_pos.items():
            differences[part] = coord - self.pos[part]

        for i in range(SMOOTH_PLOTS):
            projected_pos = {}
            depth = {}
            self.angle += ANGLE_SHIFT
            c, s = np.cos(self.angle), np.sin(self.angle)
            rotation_matrix = np.array([[c, 0, -s], [0, 1.0, 0], [s, 0, c]])
            for part, diff in differences.items():

                new_coord = self.pos[part] + diff * (1+i) / SMOOTH_PLOTS

                new_coord = np.matmul(rotation_matrix, new_coord)
                projected_pos[part] = new_coord[0:2]
                depth[part] = new_coord[-1]

            lim = 1.0
            edge_depth = []
            edge_sizes = []
            max_size = 0.9
            min_size = 0.5
            for edge in g.edges:
                cur_depth = (depth[edge[0]]+depth[edge[1]])/2
                cur_depth = min(max(cur_depth, -lim), lim)
                cur_size = min_size + (max_size-min_size)*(cur_depth+lim)/(2*lim)
                edge_depth.append(cur_depth)
                edge_sizes.append(cur_size)

            node_depth = []
            node_sizes = []
            max_size = 25
            min_size = 15
            for curPar in self.particles:
                cur_depth = depth[curPar]
                cur_depth = min(max(cur_depth, -lim), lim)
                cur_size = min_size + (max_size-min_size)*(cur_depth+lim)/(2*lim)
                node_depth.append(cur_depth)
                node_sizes.append(cur_size)

            res_x = 16*160
            ratio = 2.0
            plt.figure(figsize=(16*ratio, 9*ratio), dpi=res_x/(16*ratio))
            if background_color == "black":
                plt.style.use('dark_background')
                linecolor = "white"
                edgecolor = "black"
                cmap = plt.cm.gist_rainbow
            else:
                plt.style.use('default')
                linecolor = "black"
                edgecolor = "white"
                cmap = plt.cm.binary

            ax1 = plt.subplot2grid((8, 6), (0, 0), colspan=4, rowspan=8)

            ax2 = plt.subplot2grid((8, 6), (0, 4), colspan=1, rowspan=1)
            ax3 = plt.subplot2grid((8, 6), (1, 4), colspan=1, rowspan=1)
            ax4 = plt.subplot2grid((8, 6), (2, 4), colspan=1, rowspan=1)
            ax5 = plt.subplot2grid((8, 6), (3, 4), colspan=1, rowspan=1)
            ax6 = plt.subplot2grid((8, 6), (4, 4), colspan=1, rowspan=1)
            ax7 = plt.subplot2grid((8, 6), (5, 4), colspan=1, rowspan=1)
            ax8 = plt.subplot2grid((8, 6), (6, 4), colspan=1, rowspan=1)
            ax9 = plt.subplot2grid((8, 6), (7, 4), colspan=1, rowspan=1)

            ax10 = plt.subplot2grid((8, 6), (0, 5), colspan=1, rowspan=1)
            ax11 = plt.subplot2grid((8, 6), (1, 5), colspan=1, rowspan=1)
            ax12 = plt.subplot2grid((8, 6), (2, 5), colspan=1, rowspan=1)
            ax13 = plt.subplot2grid((8, 6), (3, 5), colspan=1, rowspan=1)
            ax14 = plt.subplot2grid((8, 6), (4, 5), colspan=1, rowspan=1)
            ax15 = plt.subplot2grid((8, 6), (5, 5), colspan=1, rowspan=1)
            ax16 = plt.subplot2grid((8, 6), (6, 5), colspan=1, rowspan=1)
            ax17 = plt.subplot2grid((8, 6), (7, 5), colspan=1, rowspan=1)

            plt.subplots_adjust(wspace=0.35, hspace=0.35)
            ax1.set_title(f"Game of Evolution, AT={token_amount_particles + token_amount_links}, "
                          f"{self.sim_options.get_description()}")

            backlash = 0.96

            tokens = [cur_par.token for cur_par in self.particles]
            self.data.max_token_amount_history = max(int(self.data.max_token_amount_history * backlash), max(tokens))
            bins = np.linspace(0.0, self.data.max_token_amount_history, 30)
            ax2.hist(tokens, rwidth=1, bins=bins, color=linecolor, edgecolor=edgecolor)
            ax2.set_title("Particle Tokens")
            ax2.set_yscale('log')

            particles_at_position = [len(cur_par.position_particle.particles_at_this_position) for cur_par in
                                     self.particles]
            self.data.max_par_conc_history = max(int(self.data.max_par_conc_history * backlash),
                                                 max(particles_at_position))
            bins = np.linspace(0.0, self.data.max_par_conc_history, 30)
            ax3.hist(particles_at_position, rwidth=1, bins=bins, color=linecolor, edgecolor=edgecolor)
            ax3.set_title("Particle Concentration")
            ax3.set_yscale('log')

            link_amounts = [cur_par.link_amount() for cur_par in self.particles]
            self.data.max_link_amount_history = max(int(self.data.max_link_amount_history * backlash),
                                                    max(link_amounts))
            bins = np.linspace(0.0, self.data.max_link_amount_history, 30)
            ax4.hist(link_amounts, rwidth=1, bins=bins, color=linecolor, edgecolor=edgecolor)
            ax4.set_title("Link Amount Distribution")
            ax4.set_yscale('log')

            ages = [cur_par.age for cur_par in self.particles]
            self.data.max_age_history = max(int(self.data.max_age_history * backlash), max(ages))
            bins = np.linspace(0.0, self.data.max_age_history, 30)
            ax5.hist(ages, rwidth=1, bins=bins, color=linecolor, edgecolor=edgecolor)
            ax5.set_title("Particle Age Distribution")
            ax5.set_yscale('log')

            ax6.set_title(f"Dimensionality Curves")
            max_dim_amount = int(np.max(dims_amounts))
            for i in range(len(self.data.dims)):
                ax6.plot(np.linspace(0.0, 1.0, dims_amounts[i]), dims[i], color=linecolor, alpha=0.15)
            ax6.plot(np.linspace(0.0, 1.0, max_dim_amount), [self.data.dim_history[-1]] * max_dim_amount,
                     color="cornflowerblue", alpha=1)
            ax6.set_xticks([])
            ax6.set_ylim(0.0, 5.5)

            ax7.set_title(f"Dimensionality {round(self.data.dim_history[-1], 4)}")
            x_data = list(range(len(self.data.dim_history)))
            y1 = np.maximum(np.array(self.data.dim_history) - np.array(self.data.dim_stddev_history), 0.0)
            y2 = np.array(self.data.dim_history) + np.array(self.data.dim_stddev_history)
            ax7.fill_between(x_data, y1, y2, color=linecolor, alpha=0.25)
            ax7.plot(x_data, self.data.dim_history, color=linecolor, alpha=1)
            ax7.plot(x_data, [3.0] * len(self.data.dim_history), color="cornflowerblue", alpha=1)
            ax7.set_ylim(0.0, 5.5)

            x_iter = np.linspace(0, self.current_iteration, len(self.data.percent_home_history))

            ax8.set_title(f"R=W P=B S=R Percentage")
            ax8.plot(x_iter, self.data.r_history, color=linecolor)
            ax8.plot(x_iter, self.data.p_history, color="cornflowerblue")
            ax8.plot(x_iter, self.data.s_history, color="coral")

            ax9.plot(x_iter, self.data.kept_tokens_history, color=linecolor)
            ax9.set_title(f"Kept Tokens Percentage {round(self.data.kept_tokens_history[-1], 4)}")

            ax10.plot(x_iter, self.data.percent_home_history, color=linecolor)
            ax10.set_title(f"Particles at Home {round(self.data.percent_home_history[-1], 4)}")

            ax11.set_title(
                f"Particle and Link Amount {self.data.particle_amount_history[-1]} {self.data.link_amount_history[-1]}")
            ax11.plot(x_iter, self.data.particle_amount_history, color=linecolor)
            ax11.plot(x_iter, self.data.link_amount_history, color=linecolor)

            ax12.set_title(f"Links per Particle  {round(self.data.links_per_particle_history[-1], 4)}")
            ax12.plot(x_iter, self.data.links_per_particle_history, color=linecolor)

            ax13.set_title(
                f"New Particles W=R  {round(self.data.reproduced_particles_history[-1], 4)}, "
                f"B=P  {round(self.data.planted_particles_history[-1], 4)}")
            ax13.plot(x_iter, self.data.reproduced_particles_history, color=linecolor)
            ax13.plot(x_iter, self.data.planted_particles_history, color="cornflowerblue")

            ax14.set_title(f"Died Particles {round(self.data.died_particles_history[-1], 4)}")
            ax14.plot(x_iter, self.data.died_particles_history, color=linecolor)

            ax15.set_title(f"Swapped Particles {round(self.data.swap_percentage_history[-1], 4)}")
            ax15.plot(x_iter, self.data.swap_percentage_history, color=linecolor)

            ax16.set_title(
                f"New Spawned Links {round(self.data.new_links_spawned_history[-1], 4)}, "
                f"{round(self.data.declined_new_links_history[-1], 4)}")
            ax16.plot(x_iter, self.data.new_links_spawned_history, color=linecolor)
            ax16.plot(x_iter, self.data.declined_new_links_history, color="cornflowerblue")

            ax17.set_title(f"Draw Percentage {round(self.data.draw_history[-1], 4)}")
            ax17.plot(x_iter, self.data.draw_history, color=linecolor)

            options = {
                'ax': ax1,
                'node_shape': ".",
                'node_size': node_sizes,
                'width': edge_sizes,
                'node_color': node_depth,
                'edge_color': edge_depth,

                'cmap': cmap,
                'vmin': -lim,
                'vmax': lim,

                'edge_cmap': cmap,
                'edge_vmin': -lim,
                'edge_vmax': lim,
                'linewidths': 0.0
            }
            nx.draw(G=g, pos=projected_pos, with_labels=False, **options)
            ax1.margins(x=0.1)

            ax1.set_xlim(-lim, lim)
            ax1.set_ylim(-lim, lim)

            plt.savefig(rf"outputs\{self.name}\Plot {int(self.current_plot)}, i = {self.current_iteration}.png")
            self.current_plot += 1
            plt.close()

        self.pos = new_pos


if __name__ == "__main__":
    """
    Runs the simulation according to the set parameters at the beginning of the script.
    """
    if LOADING_FOLDER:

        for f in os.listdir(LOADING_FOLDER):
            if f.endswith('.obj'):
                print(f)
                with open(rf"{LOADING_FOLDER}\{f}", 'rb') as handle:
                    simulation_options: SimOptions = pickle.load(handle)
                simulation = Simulation(sim_options=simulation_options)
                simulation.run_main_loop(iterations=ITERATION_AMOUNT)

    elif LOADING_FILE:

        with open(rf"outputs\{LOADING_FILE}\{LOADING_FILE}.obj", 'rb') as handle:
            simulation_options = pickle.load(handle)
        simulation = Simulation(sim_options=simulation_options)
        simulation.run_main_loop(iterations=ITERATION_AMOUNT)

    else:
        if EXPERIMENT_SERIES:
            while True:
                simulation_options = SimOptions(random_settings=RANDOM_SETTINGS)
                simulation = Simulation(sim_options=simulation_options)
                simulation.run_main_loop(iterations=ITERATION_AMOUNT)
        else:
            simulation_options = SimOptions(random_settings=RANDOM_SETTINGS)
            simulation = Simulation(sim_options=simulation_options)
            simulation.run_main_loop(iterations=ITERATION_AMOUNT)