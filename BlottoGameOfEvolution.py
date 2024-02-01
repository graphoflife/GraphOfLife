# IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
import datetime as datetime
import os
import pickle
matplotlib.use('Agg')
from scipy.sparse.linalg.eigen.arpack import eigsh
import time
import math
import threading
import csv
import matplotlib.colors as mcolors

import copy
from enum import Enum, auto

from numbers import Number

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

"""
Goal


I want a an algorithm where many different kind of strategies can emerge.

TODO
- Make option to disable path depenndency (no inheritage of behavior)

- Options like: can_plant_self, can_plant_neighbor, can_plant_walker

- Still implement a cost for new link?

- Derive prediction formula for how tokens move (schrödinger equation?)

- Now that multiple obsevations are possible, generalize chooseing rules
- Meta evoluion for link choices
- Make more evolutionary
- Make as generalized as possible
- Make restrictions later

- Make new option, which particle is relevant for meta evolution option selection in MOVE_ON_OWN

- Redo reconnection logic, set reconnection_particle from the get go

- Rename particle to agent

- Other Games

- Cake Cutting, like ultimatum suggests split of resources, other chooses which split he wants

- Blotto game, Done Done Done Done


"""
VERSION = "v012c_blotto"
# Setting options
LOADING_FOLDER = None # "favorite option 231209"   # "sample_pool"
SAMPLE_POOL_FOLDER = "favorite_option_blotto" # "best_options"
CHANGE_SETTINGS_PROBABILITY = 0.03
RANDOM_SETTINGS = True
EXPERIMENT_SERIES = True
FRAME_AMOUNT = int(1*30*60)

CAN_EXIT = True
SAVING_THRESHOLD = 9999999
# General simulation settings
INIT_PARTICLE_AMOUNT = 50
TOKEN_AMOUNT_RANGE = [300, 300]
TOKEN_AMOUNT = None

MAIN_USE_X_PROB = 1.0
OTHER_USE_X_PROB = 0.0

# Plotting options
GUI = False
PLOTTING = True

RANDOM_VIS_OPTIONS = True
PLOT_WITH_DIAGRAMS = True
ANGLE_SHIFT_RANGE = [-2 * np.pi / 180, 2 * np.pi / 180]
ANGLE_SHIFT = None
SMOOTH_PLOTS_RANGE = [6, 6]
SMOOTH_PLOTS = 1
SPRING_MODEL_ITERATIONS = 10
PLOT_EVERY_X_ITERATIONS_RANGE = [1, 1]
PLOT_EVERY_X_ITERATIONS = None

ITERATION_AMOUNT = None

# Analysis
CHECK_X_PARTICLES_FOR_DIM = 30
FINE_STRUCTURE_CONSTANT = 0.0072973525693
RESET_AGE_COUNTER_AT_RECONNECTION = False


def degree_distribution(G):
    vk = dict(G.degree())
    vk = list(vk.values())  # we get only the degree values
    maxk = np.max(vk)
    # mink = np.min(min)
    kvalues = np.arange(0, maxk + 1)  # possible values of k
    Pk = np.zeros(maxk + 1)  # P(k)
    for k in vk:
        Pk[k] = Pk[k] + 1
    Pk = Pk / sum(Pk)  # the sum of the elements of P(k) must to be equal to one

    return kvalues, Pk


def shannon_entropy(G):
    k,Pk = degree_distribution(G)
    H = 0
    for p in Pk:
        if(p > 0):
            H = H - p*math.log(p, 2)
    return H

black_color = [0, 0, 0, 1]
class VisOptions:

    backgroundcolor = (1, 1, 1, 1)
    cmap = mcolors.LinearSegmentedColormap.from_list("black_colormap", [black_color, black_color], N=256)
    cmap_edge = mcolors.LinearSegmentedColormap.from_list("black_colormap", [black_color, black_color], N=256)
    max_size_edge = 7.0
    min_size_edge = 3.5
    max_size_node = 1200
    min_size_node = 600

    def randomize(self):
        self.backgroundcolor = plt.cm.binary(np.random.uniform(0.0, 1.0))
        self.cmap = np.random.choice([plt.cm.binary, plt.cm.gist_rainbow, plt.cm.cool,
                                      plt.cm.summer, plt.cm.autumn, plt.cm.Wistia, plt.cm.winter, plt.cm.hot,
                                      plt.cm.Spectral, plt.cm.RdYlGn, plt.cm.rainbow, plt.cm.RdYlBu,
                                      plt.cm.viridis, plt.cm.plasma, plt.cm.coolwarm, plt.cm.twilight])
        self.cmap_edge = np.random.choice([plt.cm.binary, plt.cm.gist_rainbow, plt.cm.cool,
                                      plt.cm.summer, plt.cm.autumn, plt.cm.Wistia, plt.cm.winter, plt.cm.hot,
                                      plt.cm.Spectral, plt.cm.RdYlGn, plt.cm.rainbow, plt.cm.RdYlBu,
                                      plt.cm.viridis, plt.cm.plasma, plt.cm.coolwarm, plt.cm.twilight])
        self.max_size_edge = np.random.uniform(7.5, 7.5)
        self.min_size_edge = np.random.uniform(5.0, 5.0)
        self.max_size_node = np.random.uniform(1250, 1250)
        self.min_size_node = np.random.uniform(1000, 1000)


class SingleSimOption:

    def __init__(self, name, short_name, option, options: list, probability, can_meta_evolve, meta_type):
        self.name: SimOptionsEnum = name
        self.short_name: str = short_name
        self.option = option
        self.option_possibilities: dict = {str(cur_opt): cur_opt for cur_opt in options}
        self.probability = np.array(probability)
        self.probability = self.probability / np.sum(self.probability)
        self.can_meta_evolve = can_meta_evolve
        self.meta_type = meta_type
        self.sim_options = None

    def change_to_this(self, text):
        self.option = self.option_possibilities[text]

    def get(self):
        return self.option

    def set_reference_to_sim_options(self, sim_options):
        self.sim_options = sim_options

    def randomize(self):
        self.option = self.get_random_option()

    def get_random_option(self):
        return np.random.choice(list(self.option_possibilities.values()), p=self.probability)

class MetaEnum(Enum):

    # GLOBAL = auto()
    PARTICLE = auto()
    # LINK = auto()
    NONE = auto()


class SimOptionsEnum(Enum):

    USE_X_FOR_GOING_HOME = auto()
    USE_X_FOR_NEW_LINK = auto()
    USE_X_FOR_MOVING_PARTICLES = auto()
    USE_X_FOR_DEATH_RATING = auto()
    USE_X_FOR_SWAPPING = auto()
    USE_X_FOR_SHIFTING = auto()
    USE_X_FOR_TOKEN_ALLOCATION = auto()
    USE_X_FOR_PLANTING = auto()
    USE_X_FOR_X_REPRODUCTION = auto()
    USE_X_FOR_PLANTING_PARTICLE = auto()
    USE_X_FOR_ACCEPTING_LINK = auto()
    USE_X_FOR_INHERITAGING_POSITION = auto()
    USE_X_FOR_ACCEPTING_ALLOCATION = auto()
    USE_X_FOR_PUNISHING = auto()
    USE_X_FOR_KEEPING_REPRO_TOKEN = auto()
    USE_X_FOR_RECONNECTION = auto()
    USE_X_FOR_RECONNECTION_DIRECTION = auto()

    GOING_HOME_DECISION = auto()
    NEW_LINK_DECISION = auto()
    MOVING_PARTICLES_DECISION = auto()
    DEATH_RATING_DECISION = auto()
    SWAPPING_DECISION = auto()
    SHIFTING_DECISION = auto()
    PLANTING_DECISION = auto()
    X_REPRODUCTION_DECISION = auto()
    PLANTING_PARTICLE_DECISION = auto()
    ACCEPT_NEW_LINK_DECISION = auto()
    INHERITAGING_POSITION_DECISION = auto()
    ACCEPTING_ALLOCATION_DECISION = auto()
    PUNISHING_DECISION = auto()
    RECONNECTION_DECISION = auto()
    RECONNECTION_DIRECTION_DECISION = auto()

    MOVE_ON_OWN = auto()
    CAN_DECLINE_NEW_LINK = auto()

    ADD_RANDOM_VALUES = auto()

    MUTATION_CHANCE = auto()

    MUTATE_ALL_EACH_ITERATION = auto()

    KILL_INACTIVE_LINKS = auto()
    INACTIVITY_KILLS_WHICH_PARTICLE = auto()

    META_EVOLUTION = auto()
    MUTATION_CHANCE_META = auto()

    ACTIVATION_FUNCTION = auto()

    CAN_MOVE_TO_WALKERS_WALKER = auto()
    CAN_MOVE_TO_RANDOM = auto()
    CAN_NOT_MOVE = auto()

    CAN_RECONNECT = auto()

    CAN_PLANT = auto()
    CAN_SWAP = auto()

    BLOTTO_WINNER_TYPE = auto()

    REPRODUCTION_IS_A_DECISION = auto()

    CAN_TOKEN_AT_SELF = auto()
    CAN_REPRODUCE_AT_SELF = auto()
    CAN_TOKEN_AT_OTHER = auto()
    CAN_REPRODUCE_AT_OTHER = auto()

    WINNER_TOKEN_GO_TO = auto()
    WINNER_REPRO_TOKEN_GO_TO = auto()
    WHICH_BEHAVIOR_WINS = auto()
    WHICH_REPRO_BEHAVIOR_WINS = auto()
    MUTATE_WIN_BEHAVIOR = auto()

    USE_REPRO_BEH_FOR_REPRO_DEC = auto()

    USE_NEW_BEHAVIOR_FOR_PHASE_1 = auto()

class NewSimOptions:

    # Neural Network
    INPUT_SCALING = 0.1
    QUANTILE_AMOUNT = 3
    INPUT_AMOUNT = 29 + QUANTILE_AMOUNT * 3 * 4
    OUTPUT_AMOUNT = 36
    LAYER_SIZES = [int(INPUT_AMOUNT * 0.75)] * 2

    def set_random_settings(self, change_settings_probability):
        for key in self.all_options:
            if np.random.rand() < change_settings_probability:
                self.all_options[key].randomize()

    def get(self, enum_key, particle=None):
        if particle is None or not self.get(SimOptionsEnum.META_EVOLUTION):
            return self.all_options[enum_key].get()
        return particle.behavior.meta_options[enum_key]


    def save(self, filename: str, foldername:str):
        # Create a subfolder 'options' if it does not exist
        if not filename.endswith('.csv'):
            filename += '.csv'
        os.makedirs(foldername, exist_ok=True)
        filepath = os.path.join(foldername, filename)

        with open(filepath, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Option', 'Value'])
            for key, single_option in self.all_options.items():
                writer.writerow([key.name, single_option.option])

    def load(self, filename: str, foldername:str):
        # Ensure filename ends with .csv
        filepath = os.path.join(foldername, filename)

        with open(filepath, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                option_key_name = row['Option']
                option_value = row['Value']
                # Check if the option key exists in all_options
                if option_key_name in SimOptionsEnum.__members__:
                    option_key = SimOptionsEnum[option_key_name]
                    if option_key in self.all_options:
                        single_option = self.all_options[option_key]
                        # Check if the option value is valid in option_possibilities
                        if option_value in single_option.option_possibilities:
                            single_option.option = single_option.option_possibilities[option_value]

    def get_description(self):
        result = " "
        counter = 0
        for i, key in enumerate(self.all_options):
            single_option: SingleSimOption = self.all_options[key]
            if not single_option.can_meta_evolve or not self.get(SimOptionsEnum.META_EVOLUTION):
                if counter%10 == 0:
                    result += "\n"
                result += f"{single_option.short_name}={str(single_option.option)},"
                counter += 1
        return result

    def __init__(self):
        self.all_options = {}
        self.set_all_options()
        for key in self.all_options:
            self.all_options[key].set_reference_to_sim_options(sim_options=self)

    def create_meta_dictionary_for_particle(self):
        final_dict = {}
        for key in self.all_options:
            single_option: SingleSimOption = self.all_options[key]
            if single_option.can_meta_evolve:
                final_dict[key] = single_option.get_random_option()
        return final_dict

    def set_all_options(self):
        """_______________________________________________________________"""


        temp_enum = SimOptionsEnum.USE_X_FOR_GOING_HOME
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                 short_name="XGH",
                                                 option="SELF",
                                                 options=["SELF", "NEIGHBOR", "WALKER"],
                                                 probability=[MAIN_USE_X_PROB, OTHER_USE_X_PROB, OTHER_USE_X_PROB],
                                                 can_meta_evolve = True,
                                                 meta_type=MetaEnum.PARTICLE)

        temp_enum = SimOptionsEnum.USE_X_FOR_NEW_LINK
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                 short_name="XNL",
                                                 option="SELF",
                                                 options=["SELF", "NEIGHBOR", "WALKER"],
                                                 probability=[MAIN_USE_X_PROB, OTHER_USE_X_PROB, OTHER_USE_X_PROB],
                                                 can_meta_evolve = True,
                                                 meta_type=MetaEnum.PARTICLE)

        temp_enum = SimOptionsEnum.USE_X_FOR_MOVING_PARTICLES
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                 short_name="XMP",
                                                 option="SELF",
                                                 options=["SELF", "NEIGHBOR", "WALKER"],
                                                 probability=[MAIN_USE_X_PROB, OTHER_USE_X_PROB, OTHER_USE_X_PROB],
                                                 can_meta_evolve = True,
                                                 meta_type=MetaEnum.PARTICLE)

        temp_enum = SimOptionsEnum.USE_X_FOR_DEATH_RATING
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                 short_name="XDR",
                                                 option="SELF",
                                                 options=["SELF", "NEIGHBOR", "WALKER"],
                                                 probability=[MAIN_USE_X_PROB, OTHER_USE_X_PROB, OTHER_USE_X_PROB],
                                                 can_meta_evolve = True,
                                                 meta_type=MetaEnum.PARTICLE)

        temp_enum = SimOptionsEnum.USE_X_FOR_SWAPPING
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                 short_name="XS",
                                                 option="SELF",
                                                 options=["SELF", "NEIGHBOR", "WALKER"],
                                                 probability=[MAIN_USE_X_PROB, OTHER_USE_X_PROB, OTHER_USE_X_PROB],
                                                 can_meta_evolve = True,
                                                 meta_type=MetaEnum.PARTICLE)

        temp_enum = SimOptionsEnum.USE_X_FOR_SHIFTING
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                 short_name="XSH",
                                                 option="SELF",
                                                 options=["SELF", "NEIGHBOR", "WALKER"],
                                                 probability=[MAIN_USE_X_PROB, OTHER_USE_X_PROB, OTHER_USE_X_PROB],
                                                 can_meta_evolve = True,
                                                 meta_type=MetaEnum.PARTICLE)

        temp_enum = SimOptionsEnum.USE_X_FOR_TOKEN_ALLOCATION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                 short_name="XTA",
                                                 option="SELF",
                                                 options=["SELF", "NEIGHBOR", "WALKER"],
                                                 probability=[MAIN_USE_X_PROB, OTHER_USE_X_PROB, OTHER_USE_X_PROB],
                                                 can_meta_evolve = True,
                                                 meta_type=MetaEnum.PARTICLE)


        temp_enum = SimOptionsEnum.USE_X_FOR_PLANTING
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                 short_name="XP",
                                                 option="SELF",
                                                 options=["SELF", "NEIGHBOR", "WALKER"],
                                                 probability=[MAIN_USE_X_PROB, OTHER_USE_X_PROB, OTHER_USE_X_PROB],
                                                 can_meta_evolve = True,
                                                 meta_type=MetaEnum.PARTICLE)

        temp_enum = SimOptionsEnum.USE_X_FOR_X_REPRODUCTION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                 short_name="XXR",
                                                 option="SELF",
                                                 options=["SELF", "NEIGHBOR", "WALKER"],
                                                 probability=[MAIN_USE_X_PROB, OTHER_USE_X_PROB, OTHER_USE_X_PROB],
                                                 can_meta_evolve = True,
                                                 meta_type=MetaEnum.PARTICLE)

        temp_enum = SimOptionsEnum.USE_X_FOR_PLANTING_PARTICLE
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                 short_name="XPP",
                                                 option="SELF",
                                                 options=["SELF", "NEIGHBOR", "WALKER"],
                                                 probability=[MAIN_USE_X_PROB, OTHER_USE_X_PROB, OTHER_USE_X_PROB],
                                                 can_meta_evolve = True,
                                                 meta_type=MetaEnum.PARTICLE)

        temp_enum = SimOptionsEnum.USE_X_FOR_ACCEPTING_LINK
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                 short_name="XAL",
                                                 option="WALKER",
                                                 options=["SELF", "NEIGHBOR", "WALKER"],
                                                 probability=[OTHER_USE_X_PROB, OTHER_USE_X_PROB, MAIN_USE_X_PROB],
                                                 can_meta_evolve = True,
                                                 meta_type=MetaEnum.PARTICLE)

        temp_enum = SimOptionsEnum.USE_X_FOR_INHERITAGING_POSITION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                 short_name="XIP",
                                                 option="SELF",
                                                 options=["SELF", "NEIGHBOR", "WALKER"],
                                                 probability=[MAIN_USE_X_PROB, OTHER_USE_X_PROB, OTHER_USE_X_PROB],
                                                 can_meta_evolve = True,
                                                 meta_type=MetaEnum.PARTICLE)

        temp_enum = SimOptionsEnum.USE_X_FOR_ACCEPTING_ALLOCATION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                 short_name="XIP",
                                                 option="SELF",
                                                 options=["SELF", "NEIGHBOR", "WALKER"],
                                                 probability=[MAIN_USE_X_PROB, OTHER_USE_X_PROB, OTHER_USE_X_PROB],
                                                 can_meta_evolve = True,
                                                 meta_type=MetaEnum.PARTICLE)

        temp_enum = SimOptionsEnum.USE_X_FOR_PUNISHING
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="XPU",
                                                      option="SELF",
                                                      options=["SELF", "NEIGHBOR", "WALKER"],
                                                      probability=[MAIN_USE_X_PROB, OTHER_USE_X_PROB, OTHER_USE_X_PROB],
                                                      can_meta_evolve = True,
                                                      meta_type=MetaEnum.PARTICLE)

        temp_enum = SimOptionsEnum.USE_X_FOR_KEEPING_REPRO_TOKEN
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                 short_name="XKRT",
                                                 option="SELF",
                                                 options=["SELF", "NEIGHBOR", "WALKER"],
                                                 probability=[MAIN_USE_X_PROB, OTHER_USE_X_PROB, OTHER_USE_X_PROB],
                                                 can_meta_evolve = True,
                                                 meta_type=MetaEnum.PARTICLE)

        temp_enum = SimOptionsEnum.USE_X_FOR_RECONNECTION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                 short_name="XR",
                                                 option="SELF",
                                                 options=["SELF", "NEIGHBOR", "WALKER"],
                                                 probability=[MAIN_USE_X_PROB, OTHER_USE_X_PROB, OTHER_USE_X_PROB],
                                                 can_meta_evolve = True,
                                                 meta_type=MetaEnum.PARTICLE)

        temp_enum = SimOptionsEnum.USE_X_FOR_RECONNECTION_DIRECTION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                 short_name="XRD",
                                                 option="SELF",
                                                 options=["SELF", "NEIGHBOR", "WALKER"],
                                                 probability=[MAIN_USE_X_PROB, OTHER_USE_X_PROB, OTHER_USE_X_PROB],
                                                 can_meta_evolve = True,
                                                 meta_type=MetaEnum.PARTICLE)

        """_______________________________________________________________"""
        temp_enum = SimOptionsEnum.GOING_HOME_DECISION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="GHD",
                                                      option="PROB",
                                                      options=["MAX", "PROB", "RANDOM"],
                                                      probability=[3.0, 3.0, 1.0],
                                                      can_meta_evolve = True,
                                                      meta_type=MetaEnum.PARTICLE)

        temp_enum = SimOptionsEnum.NEW_LINK_DECISION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="NLD",
                                                      option="PROB",
                                                      options=["MAX", "PROB", "RANDOM"],
                                                      probability=[3.0, 3.0, 1.0],
                                                      can_meta_evolve = True,
                                                      meta_type=MetaEnum.PARTICLE)

        temp_enum = SimOptionsEnum.MOVING_PARTICLES_DECISION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="MPD",
                                                      option="PROB",
                                                      options=["MAX", "PROB", "RANDOM"],
                                                      probability=[3.0, 3.0, 1.0],
                                                      can_meta_evolve = True,
                                                      meta_type=MetaEnum.PARTICLE)

        temp_enum = SimOptionsEnum.DEATH_RATING_DECISION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="DRD",
                                                      option="PROB",
                                                      options=["MAX", "PROB", "RANDOM"],
                                                      probability=[3.0, 3.0, 1.0],
                                                      can_meta_evolve = True,
                                                      meta_type=MetaEnum.PARTICLE)


        temp_enum = SimOptionsEnum.SWAPPING_DECISION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="SD",
                                                      option="PROB",
                                                      options=["MAX", "PROB", "RANDOM"],
                                                      probability=[3.0, 3.0, 1.0],
                                                      can_meta_evolve = True,
                                                      meta_type=MetaEnum.PARTICLE)

        temp_enum = SimOptionsEnum.SHIFTING_DECISION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="SHD",
                                                      option="PROB",
                                                      options=["MAX", "PROB", "RANDOM"],
                                                      probability=[3.0, 3.0, 1.0],
                                                      can_meta_evolve = True,
                                                      meta_type=MetaEnum.PARTICLE)


        temp_enum = SimOptionsEnum.PLANTING_DECISION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="PD",
                                                      option="PROB",
                                                      options=["MAX", "PROB", "RANDOM"],
                                                      probability=[3.0, 3.0, 1.0],
                                                      can_meta_evolve = True,
                                                      meta_type=MetaEnum.PARTICLE)

        temp_enum = SimOptionsEnum.X_REPRODUCTION_DECISION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="XRD",
                                                      option="PROB",
                                                      options=["MAX", "PROB", "RANDOM"],
                                                      probability=[3.0, 3.0, 1.0],
                                                      can_meta_evolve = True,
                                                      meta_type=MetaEnum.PARTICLE)

        temp_enum = SimOptionsEnum.PLANTING_PARTICLE_DECISION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="PPD",
                                                      option="PROB",
                                                      options=["MAX", "PROB", "RANDOM"],
                                                      probability=[3.0, 3.0, 1.0],
                                                      can_meta_evolve = True,
                                                      meta_type=MetaEnum.PARTICLE)


        temp_enum = SimOptionsEnum.ACCEPT_NEW_LINK_DECISION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="ANLD",
                                                      option="PROB",
                                                      options=["MAX", "PROB", "RANDOM"],
                                                      probability=[3.0, 3.0, 1.0],
                                                      can_meta_evolve = True,
                                                      meta_type=MetaEnum.PARTICLE)


        temp_enum = SimOptionsEnum.INHERITAGING_POSITION_DECISION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="IPD",
                                                      option="PROB",
                                                      options=["MAX", "PROB", "RANDOM"],
                                                      probability=[3.0, 3.0, 1.0],
                                                      can_meta_evolve = True,
                                                      meta_type=MetaEnum.PARTICLE)

        temp_enum = SimOptionsEnum.ACCEPTING_ALLOCATION_DECISION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="IPD",
                                                      option="PROB",
                                                      options=["MAX", "PROB", "RANDOM"],
                                                      probability=[3.0, 3.0, 1.0],
                                                      can_meta_evolve = True,
                                                      meta_type=MetaEnum.PARTICLE)

        temp_enum = SimOptionsEnum.PUNISHING_DECISION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="PUD",
                                                      option="PROB",
                                                      options=["MAX", "PROB", "RANDOM"],
                                                      probability=[3.0, 3.0, 1.0],
                                                      can_meta_evolve = True,
                                                      meta_type=MetaEnum.PARTICLE)

        temp_enum = SimOptionsEnum.RECONNECTION_DECISION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="RD",
                                                      option="PROB",
                                                      options=["MAX", "PROB", "RANDOM"],
                                                      probability=[3.0, 3.0, 1.0],
                                                      can_meta_evolve=True,
                                                      meta_type=MetaEnum.PARTICLE)

        temp_enum = SimOptionsEnum.RECONNECTION_DIRECTION_DECISION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="RDD",
                                                      option="PROB",
                                                      options=["MAX", "PROB", "RANDOM"],
                                                      probability=[3.0, 3.0, 1.0],
                                                      can_meta_evolve=True,
                                                      meta_type=MetaEnum.PARTICLE)

        """_______________________________________________________________"""


        temp_enum = SimOptionsEnum.ADD_RANDOM_VALUES
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="ARV",
                                                      option=True,
                                                      options=[True, False],
                                                      probability=[1.0, 1.0],
                                                      can_meta_evolve = True,
                                                      meta_type=MetaEnum.PARTICLE)





        """_______________________________________________________________"""
        temp_enum = SimOptionsEnum.MOVE_ON_OWN
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="MOO",
                                                      option=True,
                                                      options=[True, False],
                                                      probability=[MAIN_USE_X_PROB, OTHER_USE_X_PROB],
                                                      can_meta_evolve = True,
                                                      meta_type=MetaEnum.PARTICLE)

        """_______________________________________________________________"""


        temp_enum = SimOptionsEnum.CAN_DECLINE_NEW_LINK
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="CDL",
                                                      option=True,
                                                      options=[True, False],
                                                      probability=[1.0, 1.0],
                                                      can_meta_evolve = False,
                                                      meta_type=MetaEnum.NONE)

        temp_enum = SimOptionsEnum.MUTATION_CHANCE
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="MC",
                                                      option=0.2,
                                                      options=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                                      probability=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
                                                      can_meta_evolve = False,
                                                      meta_type=MetaEnum.NONE)

        temp_enum = SimOptionsEnum.MUTATE_ALL_EACH_ITERATION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="MAEI",
                                                      option=False,
                                                      options=[True, False],
                                                      probability=[1.0, 1.0],
                                                      can_meta_evolve = False,
                                                      meta_type=MetaEnum.NONE)








        temp_enum = SimOptionsEnum.KILL_INACTIVE_LINKS
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="KIL",
                                                      option=True,
                                                      options=[True, False],
                                                      probability=[1.0, 1.0],
                                                      can_meta_evolve = False,
                                                      meta_type=MetaEnum.NONE)

        temp_enum = SimOptionsEnum.INACTIVITY_KILLS_WHICH_PARTICLE
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="IKWP",
                                                      option="MIN",
                                                      options=["MIN", "MAX", "RANDOM"],
                                                      probability=[1.0, 1.0, 1.0],
                                                      can_meta_evolve=False,
                                                      meta_type=MetaEnum.NONE)



        temp_enum = SimOptionsEnum.META_EVOLUTION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="ME",
                                                      option=True,
                                                      options=[True, False],
                                                      probability=[1.0, 0.0],
                                                      can_meta_evolve = False,
                                                      meta_type=MetaEnum.NONE)

        temp_enum = SimOptionsEnum.MUTATION_CHANCE_META
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="MCM",
                                                      option=0.2,
                                                      options=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                                      probability=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
                                                      can_meta_evolve=False,
                                                      meta_type=MetaEnum.NONE)

        temp_enum = SimOptionsEnum.ACTIVATION_FUNCTION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="ACF",
                                                      option="SIGMOID",
                                                      options=["RELU", "SIGMOID", "TANH"],
                                                      probability=[1.0, 1.0, 1.0],
                                                      can_meta_evolve=False,
                                                      meta_type=MetaEnum.NONE)

        temp_enum = SimOptionsEnum.CAN_MOVE_TO_WALKERS_WALKER
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="CMWW",
                                                      option=True,
                                                      options=[False, True],
                                                      probability=[1.0, 1.0],
                                                      can_meta_evolve=False,
                                                      meta_type=MetaEnum.NONE)

        temp_enum = SimOptionsEnum.CAN_MOVE_TO_RANDOM
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="CMR",
                                                      option=True,
                                                      options=[False, True],
                                                      probability=[1.0, 1.0],
                                                      can_meta_evolve=False,
                                                      meta_type=MetaEnum.NONE)

        temp_enum = SimOptionsEnum.CAN_NOT_MOVE
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="CNM",
                                                      option=True,
                                                      options=[False, True],
                                                      probability=[1.0, 1.0],
                                                      can_meta_evolve=False,
                                                      meta_type=MetaEnum.NONE)

        temp_enum = SimOptionsEnum.CAN_RECONNECT
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="CR",
                                                      option=False,
                                                      options=[False, True],
                                                      probability=[1.0, 1.0],
                                                      can_meta_evolve=False,
                                                      meta_type=MetaEnum.NONE)

        temp_enum = SimOptionsEnum.CAN_PLANT
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="CPL",
                                                      option=True,
                                                      options=[True, False],
                                                      probability=[1.0, 1.0],
                                                      can_meta_evolve=False,
                                                      meta_type=MetaEnum.NONE)
        temp_enum = SimOptionsEnum.CAN_SWAP
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="CSW",
                                                      option=True,
                                                      options=[True, False],
                                                      probability=[1.0, 1.0],
                                                      can_meta_evolve=False,
                                                      meta_type=MetaEnum.NONE)

        temp_enum = SimOptionsEnum.BLOTTO_WINNER_TYPE
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="BWT",
                                                      option="MAX",
                                                      options=["MAX", "PROB"],
                                                      probability=[1.0, 1.0],
                                                      can_meta_evolve=False,
                                                      meta_type=MetaEnum.NONE)

        temp_enum = SimOptionsEnum.REPRODUCTION_IS_A_DECISION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="RID",
                                                      option=True,
                                                      options=[True, False],
                                                      probability=[1.0, 1.0],
                                                      can_meta_evolve=False,
                                                      meta_type=MetaEnum.NONE)

        temp_enum = SimOptionsEnum.CAN_TOKEN_AT_SELF
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="CTAS",
                                                      option=True,
                                                      options=[True, False],
                                                      probability=[1.0, 1.0],
                                                      can_meta_evolve=False,
                                                      meta_type=MetaEnum.NONE)

        temp_enum = SimOptionsEnum.CAN_REPRODUCE_AT_SELF
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="CRAS",
                                                      option=True,
                                                      options=[True, False],
                                                      probability=[1.0, 1.0],
                                                      can_meta_evolve=False,
                                                      meta_type=MetaEnum.NONE)

        temp_enum = SimOptionsEnum.CAN_TOKEN_AT_OTHER
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="CTAO",
                                                      option=True,
                                                      options=[True, False],
                                                      probability=[1.0, 1.0],
                                                      can_meta_evolve=False,
                                                      meta_type=MetaEnum.NONE)

        temp_enum = SimOptionsEnum.CAN_REPRODUCE_AT_OTHER
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="CRAO",
                                                      option=True,
                                                      options=[True, False],
                                                      probability=[1.0, 1.0],
                                                      can_meta_evolve=False,
                                                      meta_type=MetaEnum.NONE)

        temp_enum = SimOptionsEnum.WINNER_TOKEN_GO_TO
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="WTGT",
                                                      option="COMPETITION_PLACE",
                                                      options=["COMPETITION_PLACE", "WINNER_PLACE"],
                                                      probability=[1.0, 1.0],
                                                      can_meta_evolve=False,
                                                      meta_type=MetaEnum.NONE)

        temp_enum = SimOptionsEnum.WINNER_REPRO_TOKEN_GO_TO
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="WRTGT",
                                                      option="COMPETITION_PLACE",
                                                      options=["COMPETITION_PLACE", "WINNER_PLACE"],
                                                      probability=[1.0, 1.0],
                                                      can_meta_evolve=False,
                                                      meta_type=MetaEnum.NONE)

        temp_enum = SimOptionsEnum.WHICH_BEHAVIOR_WINS
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="WBW",
                                                      option="WINNER_REPRO",
                                                      options=["WINNER_REPRO", "WINNER_HERE", "SELF_PARTICLE"],
                                                      probability=[0.0, 1.0, 1.0],
                                                      can_meta_evolve=False,
                                                      meta_type=MetaEnum.NONE)

        temp_enum = SimOptionsEnum.WHICH_REPRO_BEHAVIOR_WINS
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="WRBH",
                                                      option="WINNER_REPRO",
                                                      options=["WINNER_REPRO", "WINNER_HERE", "SELF_PARTICLE"],
                                                      probability=[1.0, 0.0, 1.0],
                                                      can_meta_evolve=False,
                                                      meta_type=MetaEnum.NONE)

        temp_enum = SimOptionsEnum.MUTATE_WIN_BEHAVIOR
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="MWB",
                                                      option=True,
                                                      options=[True, False],
                                                      probability=[1.0, 1.0],
                                                      can_meta_evolve=False,
                                                      meta_type=MetaEnum.NONE)

        temp_enum = SimOptionsEnum.USE_REPRO_BEH_FOR_REPRO_DEC
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="URBFRD",
                                                      option=True,
                                                      options=[True, False],
                                                      probability=[1.0, 1.0],
                                                      can_meta_evolve=False,
                                                      meta_type=MetaEnum.NONE)

        temp_enum = SimOptionsEnum.USE_NEW_BEHAVIOR_FOR_PHASE_1
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="URBFRD",
                                                      option=True,
                                                      options=[True, False],
                                                      probability=[1.0, 1.0],
                                                      can_meta_evolve=False,
                                                      meta_type=MetaEnum.NONE)



class Behavior:
    """
    Neural network with a rectified linear activation function that belongs to a particle.
    """
    weights: [np.array] = None
    biases: [np.array] = None

    meta_options: dict = None

    def __init__(self, sim_options: NewSimOptions, parent_behavior=None):

        self.weights = []
        self.biases = []

        if parent_behavior is None:
            # Random weights and biases
            sizes = [sim_options.INPUT_AMOUNT] + sim_options.LAYER_SIZES + [sim_options.OUTPUT_AMOUNT]
            for i in range(len(sizes) - 1):
                columns = int(sizes[i])
                rows = int(sizes[i + 1])
                self.weights.append(np.random.normal(0.0, 0.01, (rows, columns)))
                self.biases.append(np.random.normal(0.0, 0.01, (rows, 1)))

            self.meta_options = sim_options.create_meta_dictionary_for_particle()
        else:
            # Inheritance from parent behavior
            for i in range(len(parent_behavior.weights)):
                self.weights.append(parent_behavior.weights[i].copy())
                self.biases.append(parent_behavior.biases[i].copy())
            self.meta_options = copy.deepcopy(parent_behavior.meta_options)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def get_output(self, input_array: np.array, sim_options: NewSimOptions):
        """
        Feeds input vector through neural network and gets the output.
        Uses rectified linear activation function.

        :param input_array: must have shape input_amount x 1
        :return: output_array with shape output_amount x 1
        """
        output = input_array

        activation_function = sim_options.get(SimOptionsEnum.ACTIVATION_FUNCTION)
        if activation_function == "RELU":
            for i in range(len(self.weights)):
                output = np.maximum(np.matmul(self.weights[i], output) + self.biases[i], 0)
        elif activation_function == "SIGMOID":
            for i in range(len(self.weights)):
                output = self.sigmoid(np.matmul(self.weights[i], output) + self.biases[i])
            output = np.maximum(output, 0)
        else:   # TANH
            for i in range(len(self.weights)):
                output = np.tanh(np.matmul(self.weights[i], output) + self.biases[i])
            output = np.maximum(output, 0)
        return output

    def mutate(self, sim_options):
        """
        Mutate Values of weights and biases with probability MUTATION_CHANCE
        :param sim_options: Options of Simulation
        """
        THRESHOLD = 100
        for i in range(len(self.weights)):
            self.weights[i] += np.random.normal(0, 0.01, self.weights[i].shape) \
                * (np.random.uniform(0, 1, self.weights[i].shape) < sim_options.get(SimOptionsEnum.MUTATION_CHANCE))
            self.biases[i] += np.random.normal(0, 0.01, self.biases[i].shape) \
                * (np.random.uniform(0, 1, self.biases[i].shape) < sim_options.get(SimOptionsEnum.MUTATION_CHANCE))
            self.weights[i] = np.maximum(np.minimum(self.weights[i], THRESHOLD), -THRESHOLD)
            self.biases[i] = np.maximum(np.minimum(self.biases[i], THRESHOLD), -THRESHOLD)

        for key in self.meta_options:
            if np.random.uniform(0.0, 1.0) < sim_options.get(SimOptionsEnum.MUTATION_CHANCE_META):
                self.meta_options[key] = sim_options.all_options[key].get_random_option()


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
    is_reconnecting = None
    is_punishing = None
    is_accepting_other_suggestion = None
    information = None
    heritage_rating: float = None
    reconnection_direction_rating = None


    vis_pos = None
    vis_vel = None


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
        self.information = np.vstack([
            int(self.is_shifting),
            int(self.is_swapping)
            ])
        self.is_shifting = False
        self.is_swapping = False
        self.is_reconnecting = False
        self.is_punishing = False
        self.is_accepting_other_suggestion = False

    def reset_info(self):
        """
        Resets the info if node or other_node has reconnected and information no longer represents the last interaction.
        """

        self.heritage_rating = 0.0
        self.reconnection_rating = 0.0
        self.is_shifting = False
        self.is_swapping = False
        self.is_reconnecting = False
        self.is_punishing = False
        self.is_accepting_other_suggestion = False


class Link:
    """
    The link between particles
    """

    node1: LinkNode = None
    node2: LinkNode = None
    token: int = None

    age = None
    is_active = None
    active_value = None
    new_link = None

    vis_max_color_value = None


    def __init__(self, particle1, particle2, token):
        self.node1 = LinkNode()
        self.node2 = LinkNode()
        self.node1.initialize(particle=particle1, link=self, other_node=self.node2)
        self.node2.initialize(particle=particle2, link=self, other_node=self.node1)
        self.token = token
        self.age = 0
        self.is_active = True
        self.active_value = 0
        self.reset_info()
        self.is_new_link = True
        self.vis_max_color_value = 0


    def check_death(self, sim_options, all_links, all_particles, data, dead_particles, dead_links):
        if not self.is_active:
            data.inactive_links_history[-1] += 1.0
            decison = sim_options.get(SimOptionsEnum.INACTIVITY_KILLS_WHICH_PARTICLE)
            if decison == "MIN":
                surviving_particle = max([self.node1.particle, self.node2.particle], key=lambda x: x.token)
            elif decison == "MAX":
                surviving_particle = min([self.node1.particle, self.node2.particle], key=lambda x: x.token)
            else: # RANDOM
                surviving_particle = np.random.choice([self.node1.particle, self.node2.particle])
            data.died_particles_history[-1] += 1.0
            self.kill_link(sim_options=sim_options, surviving_particle=surviving_particle, all_links=all_links,
                           all_particles=all_particles, dead_particles=dead_particles, dead_links=dead_links)

    def reset_info(self):
        """
        Resets the info if node1 or node2 has reconnected and information no longer represents the last interaction.
        """
        if RESET_AGE_COUNTER_AT_RECONNECTION:
            self.age = 0

    def prepare(self, sim_options):
        """
        Prepares information for next iteration.
        :param sim_options: Options of Simulation
        """
        self.vis_max_color_value = self.active_value

        self.node1.prepare(sim_options=sim_options)
        self.node2.prepare(sim_options=sim_options)
        self.is_active = True
        self.active_value = 0
        self.age += 1
        self.is_new_link = False


    def try_swap(self, sim_options, data):
        """
        If there is mutual consent of both particles, a swap happens.
        :param sim_options: Options of Simulation
        :param data: Data for analysis
        """
        if self.node1.is_swapping and self.node2.is_swapping and sim_options.get(SimOptionsEnum.CAN_SWAP):
            self.is_active = True
            data.swap_percentage_history[-1] += 2.0

            self.node1.particle.behavior, self.node2.particle.behavior = self.node2.particle.behavior, \
                                                                         self.node1.particle.behavior


    def kill_link(self, sim_options, surviving_particle, all_links, all_particles, dead_particles, dead_links):
        """
        Kills the link and shifts all the links of the vanishing particle to the surviving particle.
        :param sim_options: Options of Simulation
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
        dying_particle.walker_position.particles_at_this_position.remove(dying_particle)
        self.token = 0
        self.node1.disconnect()
        self.node2.disconnect()
        all_links.remove(self)
        all_particles.remove(dying_particle)
        dead_links.append(self)
        dead_particles.append(dying_particle)
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
    walker_position = None
    particles_at_this_position: [] = None
    behavior: Behavior = None
    information = None
    make_new_link = None
    can_new_link = None
    go_particle = None
    reproduction_tokens = None
    reproduction_behavior = None
    new_behavior = None
    age = None
    reproduce_at_home = None
    plant_particle = None
    to_mutate = None
    going_home = None
    inherit_walker_position = None
    token_ref = None
    repro_token_ref = None
    reconnection_rating = None
    reconnection_particle = None
    is_new_link = None

    vis_pos = None
    vis_vel = None
    vis_max_color_value = None

    blotto_competition_here = None
    blotto_competition_repro = None

    blotto_winner_here = None
    blotto_winner_repro = None


    def __init__(self, token, behavior):
        self.nodes = []
        self.token = token
        self.walker_position = self
        self.particles_at_this_position = [self]
        self.behavior = behavior
        self.new_behavior = None
        self.age = 0
        self.can_new_link = True
        self.to_mutate = False
        self.going_home = False
        self.inherit_walker_position = False
        self.reproduction_tokens = 0
        self.token_ref = 0.0
        self.repro_token_ref = 0.0
        self.reconnection_rating = 0.0
        self.is_new_link = True
        self.vis_max_color_value = self.token

    def prepare_reconnect(self, sim_options):

        cannot_reconnect_to_particle = None

        for cur_node in self.nodes:
            if cur_node.is_reconnecting:
                cur_node.reconnection_rating = 0.0
                cannot_reconnect_to_particle = cur_node.other_node.particle
                break

        if cannot_reconnect_to_particle is None:
            self.reconnection_particle = self
            return

        choices = [cur_node.other_node.particle for cur_node in self.nodes]
        choices.append(self)
        p = [cur_node.reconnection_rating for cur_node in self.nodes]
        p.append(self.reconnection_rating)

        chosen_particle = Particle.make_choice(to_choose=choices, p=p,
                                  choice_type=sim_options.get(SimOptionsEnum.RECONNECTION_DIRECTION_DECISION,
                                                              particle=self))
        if chosen_particle is cannot_reconnect_to_particle:
            chosen_particle = self
        self.reconnection_particle = chosen_particle


    def do_reconnect(self, data):
        for cur_node in self.nodes:
            if cur_node.is_reconnecting:
                if self.reconnection_particle is not self:
                    cur_node.link.is_active = True
                    cur_node.switch_to_particle(self.reconnection_particle)
                    data.reconnection_history[-1] += 1.0
                break

    def get_own_node_to_other_particle(self, other_particle):
        for cur_node in self.nodes:
            if cur_node.other_node.particle is other_particle:
                return cur_node
        return None

    def get_winner_and_amount_of_competition(self, sim_options, competition):
        items = list(competition.items())
        if sim_options.get(SimOptionsEnum.BLOTTO_WINNER_TYPE) == "MAX":
            np.random.shuffle(items)
            winner_particle = max(items, key=lambda x: x[1])[0]
        else:
            keys = [item[0] for item in items]
            weights = np.array([item[1] for item in items])
            if np.sum(weights) == 0.0:
                weights = np.ones_like(weights)
            normalized_weights = weights / np.sum(weights)
            winner_particle = np.random.choice(keys, p=normalized_weights)
        return winner_particle, sum(competition.values())

    def eval_game(self, sim_options, data):
        """
        :param sim_options: Options of Simulation
        :param data: Data for analysis
        """
        winner_particle_here, sum_token_here = self.get_winner_and_amount_of_competition(sim_options=sim_options, competition=self.blotto_competition_here)
        winner_particle_repro, sum_token_repro = self.get_winner_and_amount_of_competition(sim_options=sim_options, competition=self.blotto_competition_repro)

        if sim_options.get(SimOptionsEnum.WINNER_TOKEN_GO_TO) == "COMPETITION_PLACE":
            self.token += sum_token_here
        else:   # "WINNER_PLACE"
            winner_particle_here.token += sum_token_here

        if sim_options.get(SimOptionsEnum.WINNER_REPRO_TOKEN_GO_TO) == "COMPETITION_PLACE":
            self.reproduction_tokens += sum_token_repro
        else:   # "WINNER_PLACE"
            winner_particle_here.reproduction_tokens += sum_token_repro

        if sim_options.get(SimOptionsEnum.WHICH_BEHAVIOR_WINS) == "WINNER_HERE":
            self.new_behavior = winner_particle_here.behavior
        elif sim_options.get(SimOptionsEnum.WHICH_BEHAVIOR_WINS) == "WINNER_REPRO":
            self.new_behavior = winner_particle_repro.behavior
        else:   # "SELF_PARTICLE"
            self.new_behavior = self.behavior

        if sim_options.get(SimOptionsEnum.WHICH_REPRO_BEHAVIOR_WINS) == "WINNER_HERE":
            self.reproduction_behavior = winner_particle_here.behavior
        elif sim_options.get(SimOptionsEnum.WHICH_REPRO_BEHAVIOR_WINS) == "WINNER_REPRO":
            self.reproduction_behavior = winner_particle_repro.behavior
        else:   # "SELF_PARTICLE"
            self.reproduction_behavior = self.behavior

        if sim_options.get(SimOptionsEnum.MUTATE_WIN_BEHAVIOR):
            self.to_mutate = True




    def set_repro_token(self, sim_options, data):

        if sim_options.get(SimOptionsEnum.REPRODUCTION_IS_A_DECISION):
            denominator = self.repro_token_ref + self.token_ref
            if denominator > 0.0:
                repro_percentage = self.repro_token_ref / denominator
                repro_tokens = int(np.round(repro_percentage * self.token))
                data.kept_repro_tokens_history[-1] += repro_tokens

                self.reproduction_tokens += repro_tokens
                self.token -= repro_tokens


    def set_new_behavior(self, sim_options):
        self.behavior = Behavior(sim_options=sim_options, parent_behavior=self.new_behavior)


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
        self.vis_max_color_value = self.token
        self.go_particle = None
        self.reproduction_tokens = 0
        self.age += 1
        self.to_mutate = False
        self.going_home = False
        self.inherit_walker_position = False
        self.can_new_link = True
        self.token_ref = 0.0
        self.repro_token_ref = 0.0
        self.reconnection_rating = 0.0
        self.is_new_link = False

        self.blotto_competition_here = {}
        self.blotto_competition_repro = {}
        self.blotto_winner_here = None
        self.blotto_winner_repro = None
        self.new_behavior = None


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

        token_quantiles = np.array(token_quantiles).reshape(sim_options.QUANTILE_AMOUNT, 1)*sim_options.INPUT_SCALING
        link_quantiles = np.array(link_quantiles).reshape(sim_options.QUANTILE_AMOUNT, 1)*sim_options.INPUT_SCALING
        particle_quantiles = np.array(particle_quantiles).reshape(sim_options.QUANTILE_AMOUNT, 1)*sim_options.INPUT_SCALING

        self.information = np.vstack([self.token*sim_options.INPUT_SCALING,
                                      self.reproduction_tokens*sim_options.INPUT_SCALING,
                                      self.link_amount()*sim_options.INPUT_SCALING,
                                      (len(self.particles_at_this_position))*sim_options.INPUT_SCALING,
                                      token_quantiles, link_quantiles, particle_quantiles])


    def get_link_to_particle(self, to_particle):
        """
        :param to_particle: The link to this particle is wanted.
        :return: Returns the link that connects this particle to another particle.
        """
        for cur_node in self.nodes:
            if cur_node.other_node.particle is to_particle:
                return cur_node.link
        return None


    def process_information_and_make_decisions_phase0(self, sim_options, data, all_particles):
        """
        Assembles input vector for each node, feeds input through own, other and position behaviors and makes decisions
        according to the outputs of this process.
        :param sim_options: Options of Simulation
        :param data: Data for analysis
        """

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
            if sim_options.get(SimOptionsEnum.ADD_RANDOM_VALUES, particle=self):
                add_values = np.random.random((3, 1))
            else:
                add_values = np.zeros((3, 1))
            other_walker_at_home = cur_node.other_node.particle.walker_position is cur_node.other_node.particle

            input_array = np.vstack([0.0, 0.0, 0.0, self.information,
                                     cur_node.other_node.particle.information,
                                     self.walker_position.information,
                                     cur_node.other_node.particle.walker_position.information,
                                     walker_at_home,
                                     other_walker_at_home,
                                     pos_link_connection,
                                     cur_node.information,
                                     cur_node.other_node.information,
                                     add_values
                                     ])
            output_arrays_self.append(self.behavior.get_output(input_array=input_array, sim_options=sim_options))
            input_array[0] = 1.0
            output_arrays_other.append(cur_node.other_node.particle.behavior.get_output(input_array=input_array, sim_options=sim_options))
            input_array[1] = 1.0
            output_arrays_walker.append(self.walker_position.behavior.get_output(input_array=input_array, sim_options=sim_options))
            input_array[0] = 0.0
            for j, cur_par in enumerate(self.particles_at_this_position):
                if sim_options.get(SimOptionsEnum.MOVE_ON_OWN, particle=self.particles_at_this_position[j]):
                    outputs_for_walkers_at_this_position[j].append(cur_par.behavior.get_output(input_array=input_array, sim_options=sim_options))

        output_arrays_self = np.hstack(output_arrays_self)
        output_arrays_other = np.hstack(output_arrays_other)
        output_arrays_walker = np.hstack(output_arrays_walker)

        for j, cur_par in enumerate(self.particles_at_this_position):
            if sim_options.get(SimOptionsEnum.MOVE_ON_OWN, particle=self.particles_at_this_position[j]):
                outputs_for_walkers_at_this_position[j] = np.hstack(outputs_for_walkers_at_this_position[j])

        index = 0

        # Token allocation
        output_arrays = self.get_correct_output(
            use_x=sim_options.get(SimOptionsEnum.USE_X_FOR_TOKEN_ALLOCATION, particle=self),
            output_arrays_self=output_arrays_self,
            output_arrays_other=output_arrays_other,
            output_arrays_walker=output_arrays_walker)
        token_outputs = output_arrays[index:index + 4, :]

        if sim_options.get(SimOptionsEnum.CAN_TOKEN_AT_SELF):
            token_outputs[0, :] *= 0.0
        if sim_options.get(SimOptionsEnum.CAN_REPRODUCE_AT_SELF):
            token_outputs[1, :] *= 0.0
        if sim_options.get(SimOptionsEnum.CAN_TOKEN_AT_OTHER):
            token_outputs[2, :] *= 0.0
        if not sim_options.get(SimOptionsEnum.CAN_REPRODUCE_AT_OTHER):
            token_outputs[3, :] *= 0.0

        sumx = np.sum(token_outputs)
        if sumx == 0.0:
            token_outputs = np.ones(token_outputs.shape)
            if sim_options.get(SimOptionsEnum.CAN_TOKEN_AT_SELF):
                token_outputs[0, :] *= 0.0
            if sim_options.get(SimOptionsEnum.CAN_REPRODUCE_AT_SELF):
                token_outputs[1, :] *= 0.0
            if sim_options.get(SimOptionsEnum.CAN_TOKEN_AT_OTHER):
                token_outputs[2, :] *= 0.0
            if not sim_options.get(SimOptionsEnum.CAN_REPRODUCE_AT_OTHER):
                token_outputs[3, :] *= 0.0
            sumx = np.sum(token_outputs)
            if sumx == 0.0:
                token_outputs = np.ones(token_outputs.shape)

        sum_tokens_here = np.mean(token_outputs[0, :])
        sum_tokens_repro = np.mean(token_outputs[1, :])

        token_outputs[0, :] *= 0.0
        token_outputs[1, :] *= 0.0

        token_outputs[0, 0] = sum_tokens_here
        token_outputs[1, 0] = sum_tokens_repro

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
        data.repro_token_self_invested_history[-1] += token_outputs[1, 0]

        self.blotto_competition_here[self] = token_outputs[0, 0]
        self.blotto_competition_repro[self] = token_outputs[1, 0]
        for i, cur_node in enumerate(self.nodes):
            cur_node.link.active_value += token_outputs[2, i] + token_outputs[3, i]
            data.token_other_invested_history[-1] += token_outputs[2, i]
            data.repro_token_other_invested_history[-1] += token_outputs[3, i]
            cur_node.other_node.particle.blotto_competition_here[self] = token_outputs[2, i]
            cur_node.other_node.particle.blotto_competition_repro[self] = token_outputs[3, i]
        self.token = 0
        index += 4


    def process_information_and_make_decisions_phase1(self, sim_options, data, all_particles):
        """
        Assembles input vector for each node, feeds input through own, other and position behaviors and makes decisions
        according to the outputs of this process.
        :param sim_options: Options of Simulation
        :param data: Data for analysis
        """

        walker_at_home = self.walker_position is self
        pos_link = self.get_link_to_particle(to_particle=self.walker_position)
        if pos_link:
            pos_link_connection = 1
        else:
            pos_link_connection = 0

        output_arrays_self = []
        output_arrays_self_repro = []
        output_arrays_other = []
        output_arrays_walker = []

        outputs_for_walkers_at_this_position = [[] for _ in range(len(self.particles_at_this_position))]

        for i, cur_node in enumerate(self.nodes):
            if sim_options.get(SimOptionsEnum.ADD_RANDOM_VALUES, particle=self):
                add_values = np.random.random((3, 1))
            else:
                add_values = np.zeros((3, 1))
            other_walker_at_home = cur_node.other_node.particle.walker_position is cur_node.other_node.particle

            input_array = np.vstack([0.0, 0.0, 1.0, self.information,
                                     cur_node.other_node.particle.information,
                                     self.walker_position.information,
                                     cur_node.other_node.particle.walker_position.information,
                                     walker_at_home,
                                     other_walker_at_home,
                                     pos_link_connection,
                                     cur_node.information,
                                     cur_node.other_node.information,
                                     add_values
                                     ])
            output_arrays_self.append(self.behavior.get_output(input_array=input_array, sim_options=sim_options))
            if sim_options.get(SimOptionsEnum.USE_REPRO_BEH_FOR_REPRO_DEC):
                output_arrays_self_repro.append(self.reproduction_behavior.get_output(input_array=input_array, sim_options=sim_options))
            input_array[0] = 1.0
            output_arrays_other.append(cur_node.other_node.particle.behavior.get_output(input_array=input_array, sim_options=sim_options))
            input_array[1] = 1.0
            output_arrays_walker.append(self.walker_position.behavior.get_output(input_array=input_array, sim_options=sim_options))
            input_array[0] = 0.0
            for j, cur_par in enumerate(self.particles_at_this_position):
                if sim_options.get(SimOptionsEnum.MOVE_ON_OWN, particle=self.particles_at_this_position[j]):
                    outputs_for_walkers_at_this_position[j].append(cur_par.behavior.get_output(input_array=input_array, sim_options=sim_options))

        output_arrays_self = np.hstack(output_arrays_self)
        output_arrays_other = np.hstack(output_arrays_other)
        output_arrays_walker = np.hstack(output_arrays_walker)

        if sim_options.get(SimOptionsEnum.USE_REPRO_BEH_FOR_REPRO_DEC):
            output_reproduction_arrays = np.hstack(output_arrays_self_repro)
        else:
            output_reproduction_arrays = output_arrays_self

        for j, cur_par in enumerate(self.particles_at_this_position):
            if sim_options.get(SimOptionsEnum.MOVE_ON_OWN, particle=self.particles_at_this_position[j]):
                outputs_for_walkers_at_this_position[j] = np.hstack(outputs_for_walkers_at_this_position[j])

        index = 0

        # Token allocation on phase0
        index += 4

        # Decide How much percent to reproduce at self
        output_arrays = self.get_correct_output(
            use_x=sim_options.get(SimOptionsEnum.USE_X_FOR_KEEPING_REPRO_TOKEN, particle=self),
            output_arrays_self=output_arrays_self,
            output_arrays_other=output_arrays_other,
            output_arrays_walker=output_arrays_walker)
        self.token_ref = np.sum(output_arrays[index, :])
        self.repro_token_ref = np.sum(output_arrays[index + 1, :])
        index += 2

        # 1 Make new link Decision
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

        # 2 Accept new link
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

        # 3 Move particles
        for i, cur_particle in enumerate(self.particles_at_this_position):

            if sim_options.get(SimOptionsEnum.MOVE_ON_OWN, particle=self.particles_at_this_position[i]):

                choices = [0, 1, 2] + self.nodes
                self_option = np.mean(outputs_for_walkers_at_this_position[i][index, :])
                walker_option = np.mean(outputs_for_walkers_at_this_position[i][index + 1, :])
                random_option = np.mean(outputs_for_walkers_at_this_position[i][index + 2, :])
                neighbor_option = outputs_for_walkers_at_this_position[i][index + 3, :]

                if not sim_options.get(SimOptionsEnum.CAN_NOT_MOVE):
                    self_option = 0.0
                if not sim_options.get(SimOptionsEnum.CAN_MOVE_TO_WALKERS_WALKER):
                    walker_option = 0.0
                if not sim_options.get(SimOptionsEnum.CAN_MOVE_TO_RANDOM):
                    random_option = 0.0
                probability = np.hstack([self_option, walker_option, random_option, neighbor_option])
                if probability.sum() == 0.0:
                    self_option = 1.0
                    walker_option = 1.0
                    random_option = 1.0
                    neighbor_option = np.ones_like(neighbor_option)
                    if not sim_options.get(SimOptionsEnum.CAN_NOT_MOVE):
                        self_option = 0.0
                    if not sim_options.get(SimOptionsEnum.CAN_MOVE_TO_WALKERS_WALKER):
                        walker_option = 0.0
                    if not sim_options.get(SimOptionsEnum.CAN_MOVE_TO_RANDOM):
                        random_option = 0.0
                    probability = np.hstack([self_option, walker_option, random_option, neighbor_option])

                choice = self.make_choice(to_choose=choices, p=probability,
                                          choice_type=sim_options.get(SimOptionsEnum.MOVING_PARTICLES_DECISION, particle=self.particles_at_this_position[i]))
                if choice == 0:
                    cur_particle.go_particle = self
                elif choice == 1:
                    cur_particle.go_particle = self.walker_position
                elif choice == 2:
                    cur_particle.go_particle = np.random.choice(all_particles)
                else:
                    cur_particle.go_particle = choice.other_node.particle
            else:
                output_arrays = self.get_correct_output(
                    use_x=sim_options.get(SimOptionsEnum.USE_X_FOR_MOVING_PARTICLES,
                                            particle=self.particles_at_this_position[i]),
                                            output_arrays_self=output_arrays_self,
                                            output_arrays_other=output_arrays_other,
                                            output_arrays_walker=output_arrays_walker)

                choices = [0, 1, 2] + self.nodes
                self_option = np.mean(output_arrays[index, :])
                walker_option = np.mean(output_arrays[index + 1, :])
                random_option = np.mean(output_arrays[index + 2, :])
                neighbor_option = output_arrays[index + 3, :]

                if not sim_options.get(SimOptionsEnum.CAN_NOT_MOVE):
                    self_option = 0.0
                if not sim_options.get(SimOptionsEnum.CAN_MOVE_TO_WALKERS_WALKER):
                    walker_option = 0.0
                if not sim_options.get(SimOptionsEnum.CAN_MOVE_TO_RANDOM):
                    random_option = 0.0
                probability = np.hstack([self_option, walker_option, random_option, neighbor_option])
                if probability.sum() == 0.0:
                    self_option = 1.0
                    walker_option = 1.0
                    random_option = 1.0
                    neighbor_option = np.ones_like(neighbor_option)
                    if not sim_options.get(SimOptionsEnum.CAN_NOT_MOVE):
                        self_option = 0.0
                    if not sim_options.get(SimOptionsEnum.CAN_MOVE_TO_WALKERS_WALKER):
                        walker_option = 0.0
                    if not sim_options.get(SimOptionsEnum.CAN_MOVE_TO_RANDOM):
                        random_option = 0.0
                    probability = np.hstack([self_option, walker_option, random_option, neighbor_option])

                choice = self.make_choice(to_choose=choices, p=probability,
                                          choice_type=sim_options.get(SimOptionsEnum.MOVING_PARTICLES_DECISION,
                                                                      particle=self.particles_at_this_position[i]))
                if choice == 0:
                    cur_particle.go_particle = self
                elif choice == 1:
                    cur_particle.go_particle = self.walker_position
                elif choice == 2:
                    cur_particle.go_particle = np.random.choice(all_particles)
                else:
                    cur_particle.go_particle = choice.other_node.particle
        index += 4

        # 4 Dying direction
        output_arrays = self.get_correct_output(use_x=sim_options.get(SimOptionsEnum.USE_X_FOR_DEATH_RATING, particle=self),
                                                output_arrays_self=output_arrays_self,
                                                output_arrays_other=output_arrays_other,
                                                output_arrays_walker=output_arrays_walker)
        heritage_outputs = output_arrays[index, :]
        for i, cur_node in enumerate(self.nodes):
            cur_node.heritage_rating = heritage_outputs[i]
        index += 1

        # 5 Swapping
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

        # Decide to go home
        output_arrays = self.get_correct_output(use_x=sim_options.get(SimOptionsEnum.USE_X_FOR_GOING_HOME, particle=self),
                                                output_arrays_self=output_arrays_self,
                                                output_arrays_other=output_arrays_other,
                                                output_arrays_walker=output_arrays_walker)
        go_home_sum = np.sum(output_arrays[index, :])
        dont_go_home_sum = np.sum(output_arrays[index + 1, :])
        to_choose = [True, False]
        choice = self.make_choice(to_choose=to_choose, p=[go_home_sum, dont_go_home_sum],
                                  choice_type=sim_options.get(SimOptionsEnum.GOING_HOME_DECISION, particle=self))
        self.going_home = choice
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
            choice = self.make_choice(to_choose=self.nodes, p=swapping_outputs,
                                      choice_type=sim_options.get(SimOptionsEnum.RECONNECTION_DECISION, particle=self))
            choice.is_reconnecting = True
        index += 2

        # Reconnection Direction
        output_arrays = self.get_correct_output(
                use_x=sim_options.get(SimOptionsEnum.USE_X_FOR_RECONNECTION_DIRECTION, particle=self),
                output_arrays_self=output_arrays_self,
                output_arrays_other=output_arrays_other,
                output_arrays_walker=output_arrays_walker)
        reconnection_direction_outputs_self = output_arrays[index, :]
        reconnection_direction_outputs = output_arrays[index+1, :]
        self.reconnection_rating = np.mean(reconnection_direction_outputs_self)
        for i, cur_node in enumerate(self.nodes):
            cur_node.reconnection_rating = reconnection_direction_outputs[i]



        # 6 Shifting Decision
        output_arrays = self.get_correct_output(use_x=sim_options.get(SimOptionsEnum.USE_X_FOR_SHIFTING, particle=self),
                                                output_arrays_self=output_reproduction_arrays,
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


        # 11 Plant or not
        output_arrays = self.get_correct_output(use_x=sim_options.get(SimOptionsEnum.USE_X_FOR_PLANTING, particle=self),
                                                output_arrays_self=output_reproduction_arrays,
                                                output_arrays_other=output_arrays_other,
                                                output_arrays_walker=output_arrays_walker)
        to_choose = [True, False]
        choice = self.make_choice(to_choose=to_choose, p=[np.sum(output_arrays[index, :]), np.sum(output_arrays[index + 1, :])],
                                  choice_type=sim_options.get(SimOptionsEnum.PLANTING_DECISION, particle=self))
        self.reproduce_at_home = choice
        index += 2



        # 13 Plant on which
        output_arrays = self.get_correct_output(use_x=sim_options.get(SimOptionsEnum.USE_X_FOR_PLANTING_PARTICLE, particle=self),
                                                output_arrays_self=output_reproduction_arrays,
                                                output_arrays_other=output_arrays_other,
                                                output_arrays_walker=output_arrays_walker)
        choices = [0, 1] + self.nodes     # self, walker, neighbors
        self_option = np.mean(output_arrays[index, :])
        walker_option = np.mean(output_arrays[index + 1, :])
        neighbor_option = output_arrays[index + 2, :]
        choice = self.make_choice(to_choose=choices, p=np.hstack([self_option,walker_option,neighbor_option]),
                         choice_type=sim_options.get(SimOptionsEnum.PLANTING_PARTICLE_DECISION, particle=self))
        if choice == 0:
            self.plant_particle = self
        elif choice == 1:
            self.plant_particle = self.walker_position
        else:
            self.plant_particle = choice.other_node.particle
        index += 3


        # inheritage decision
        output_arrays = self.get_correct_output(use_x=sim_options.get(SimOptionsEnum.USE_X_FOR_INHERITAGING_POSITION, particle=self),
                                                output_arrays_self=output_reproduction_arrays,
                                                output_arrays_other=output_arrays_other,
                                                output_arrays_walker=output_arrays_walker)
        inherit_sum = np.sum(output_arrays[index, :])
        dont_inherit_sum = np.sum(output_arrays[index + 1, :])
        to_choose = [True, False]
        choice = self.make_choice(to_choose=to_choose, p=[inherit_sum, dont_inherit_sum],
                                  choice_type=sim_options.get(SimOptionsEnum.INHERITAGING_POSITION_DECISION, particle=self))
        self.inherit_walker_position = choice
        index += 2

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
            return np.random.choice(to_choose)

    def reproduce(self, sim_options, pos, all_links, all_particles, data, vel, ph_dict_li):
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

        if self.reproduce_at_home or not sim_options.get(SimOptionsEnum.CAN_PLANT):
            data.reproduced_particles_history[-1] += 1.0
            new_behavior = Behavior(sim_options=sim_options, parent_behavior=self.reproduction_behavior)
            new_particle = Particle(token=self.reproduction_tokens, behavior=new_behavior)
            new_particle.to_mutate = True
            if self.inherit_walker_position:
                new_particle.move_to_particle(self.walker_position)
            self.reproduction_tokens = 0
            if pos is not None:
                pos[new_particle] = pos[self].copy() * np.random.normal(1, 0.001)
                vel[new_particle] = vel[self].copy()
            for cur_node in self.nodes:
                if cur_node.is_shifting:
                    cur_node.switch_to_particle(particle=new_particle)

            new_link = Link(particle1=self, particle2=new_particle, token=0)
            if pos is not None:
                new_link.node1.vis_pos = pos[self].copy()
                new_link.node2.vis_pos = pos[new_particle].copy()
            if GUI:
                ph_dict_li[self] = new_particle
            all_particles.append(new_particle)
            all_links.append(new_link)
        else:   # Plant
            data.planted_particles_history[-1] += 1.0
            new_behavior = Behavior(sim_options=sim_options, parent_behavior=self.reproduction_behavior)
            new_particle = Particle(token=self.reproduction_tokens, behavior=new_behavior)
            new_particle.to_mutate = True
            if self.inherit_walker_position:
                new_particle.move_to_particle(self.walker_position)
            self.reproduction_tokens = 0
            if pos is not None:
                pos[new_particle] = pos[self.plant_particle].copy() * np.random.normal(1, 0.001)
                vel[new_particle] = np.zeros(3)
            new_link = Link(particle1=self.plant_particle, particle2=new_particle, token=0)
            if pos is not None:
                new_link.node1.vis_pos = pos[self].copy()
                new_link.node2.vis_pos = pos[new_particle].copy()
            if GUI:
                ph_dict_li[self] = new_particle
            all_particles.append(new_particle)
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
            p = [cur_node.heritage_rating for cur_node in self.nodes]
            chosen_node = self.make_choice(to_choose=self.nodes, p=p, choice_type=sim_options.get(SimOptionsEnum.DEATH_RATING_DECISION, particle=self))
            data.died_particles_history[-1] += 1.0
            chosen_node.link.kill_link(sim_options=sim_options, surviving_particle=chosen_node.other_node.particle,
                                       all_links=all_links, all_particles=all_particles, dead_particles=dead_particles, dead_links=dead_links)

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
                    return
            # if reached -> not connected
            if not self.can_new_link:
                data.declined_new_links_history[-1] += 1.0
                return

            data.new_links_spawned_history[-1] += 1.0
            new_link = Link(particle1=self, particle2=self.walker_position, token=0)
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
        if self.going_home:
            self.move_to_particle(self)
        else:
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
        dims = []
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
    dim_max_history = None
    max_link_amount_history = None
    max_par_conc_history = None
    max_token_amount_history = None
    max_token_link_amount_history = None
    max_age_history = None
    max_link_age_history = None

    swap_percentage_history = None
    new_links_spawned_history = None
    declined_new_links_history = None
    planted_particles_history = None
    reproduced_particles_history = None
    links_per_particle_history = None
    died_particles_history = None
    shannon_entropy = None
    mean_steps_history = None


    reconnection_history = None
    inactive_links_history = None


    repro_token_self_invested_history = None
    token_self_invested_history = None
    repro_token_other_invested_history = None
    token_other_invested_history = None
    win_token_percentage_history = None
    win_repro_token_percentage_history = None

    def __init__(self):
        """
        Initialize all all data variables
        """
        self.particle_amount_history = []
        self.link_amount_history = []
        self.percent_home_history = []
        self.dim_max_history = []
        self.dim_mean_history = []
        self.max_link_amount_history = 1
        self.max_par_conc_history = 1
        self.max_token_amount_history = 1
        self.max_token_link_amount_history = 1
        self.max_age_history = 1
        self.max_link_age_history = 1

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

        self.allocation_accepted_history = []
        self.punished_history = []


        self.reconnection_history = []

        self.inactive_links_history = []

        self.repro_token_self_invested_history = []
        self.token_self_invested_history = []
        self.repro_token_other_invested_history = []
        self.token_other_invested_history = []
        self.win_token_percentage_history = []
        self.win_repro_token_percentage_history = []


    def renormalize(self, token_amount):

        self.swap_percentage_history[-1] = self.swap_percentage_history[-1] / self.particle_amount_history[-1]
        self.new_links_spawned_history[-1] = self.new_links_spawned_history[-1] / self.particle_amount_history[-1]
        self.declined_new_links_history[-1] = self.declined_new_links_history[-1] / self.particle_amount_history[-1]
        self.planted_particles_history[-1] = self.planted_particles_history[-1] / self.particle_amount_history[-1]
        self.reproduced_particles_history[-1] = self.reproduced_particles_history[-1] / self.particle_amount_history[-1]
        self.died_particles_history[-1] = self.died_particles_history[-1] / self.particle_amount_history[-1]
        self.kept_repro_tokens_history[-1] = self.kept_repro_tokens_history[-1] / token_amount
        self.allocation_accepted_history[-1] = self.allocation_accepted_history[-1] / (self.link_amount_history[-1] * 2)
        self.punished_history[-1] = self.punished_history[-1] /  (self.link_amount_history[-1] * 2)

        self.reconnection_history[-1] /= self.particle_amount_history[-1]

        self.inactive_links_history[-1] /= self.link_amount_history[-1]

        self.repro_token_self_invested_history[-1] /= token_amount
        self.token_self_invested_history[-1] /= token_amount
        self.repro_token_other_invested_history[-1] /= token_amount
        self.token_other_invested_history[-1] /= token_amount
        self.win_token_percentage_history[-1] /= token_amount
        self.win_repro_token_percentage_history[-1] /= token_amount




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

    def __init__(self, sim_options, event=None):
        """
        Initialization of the simulation. Create neccesary folders. Create initial particles and links.
        Create evaluation matrix for the game evaluation.
        :param sim_options: Options of Simulation
        """
        self.particles = []
        self.links = []
        self.sim_options_original = sim_options
        self.sim_options = copy.deepcopy(self.sim_options_original)
        self.data = Data()
        self.current_iteration = 0
        self.angle = 0

        self.dead_links = []
        self.dead_particles = []

        # GUI things
        self.event: threading.Event = event

        self.particle_heritage_dictionary_last_iteration = {}
        self.particle_heritage_dictionary_gui_queue = {}

        self.is_running = True
        self.gui_g_array = None
        self.gui_pos_array = None
        self.gui_dt = None
        self.gui_g = None
        self.gui_par_tokens = None
        self.gui_link_tokens = None
        self.node_sizes_array = None
        self.edge_sizes_array = None
        self.node_colors_array = None
        self.edge_colors_array = None

        self.vis_max_node_color_value = 1.0
        self.vis_max_edge_color_value = 1.0

        # Create Folder and Save Simulation Options
        self.name = f"GOE_{VERSION}_"+ datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        parent_dir = os.path.dirname(os.path.realpath(__file__))
        output_dir = os.path.join(parent_dir, "outputs")
        every_option_dir = os.path.join(parent_dir, "every_option")
        path = os.path.join(output_dir, self.name)

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(every_option_dir, exist_ok=True)
        if PLOTTING:
            os.makedirs(path, exist_ok=True)
            self.sim_options.save(self.name, path)
        self.sim_options.save(self.name, every_option_dir)



        self.vis_options = VisOptions()
        if RANDOM_VIS_OPTIONS:
            self.vis_options.randomize()

        global TOKEN_AMOUNT
        global ANGLE_SHIFT
        global SMOOTH_PLOTS
        global PLOT_EVERY_X_ITERATIONS
        global ITERATION_AMOUNT


        print("Iterations Amount:", ITERATION_AMOUNT)

        # Particle and Link Initialization
        init_token_per_particle = int(TOKEN_AMOUNT/INIT_PARTICLE_AMOUNT)
        for i in range(INIT_PARTICLE_AMOUNT):
            new_behavior = Behavior(sim_options=sim_options)
            self.particles.append(Particle(token=init_token_per_particle, behavior=new_behavior))
        if INIT_PARTICLE_AMOUNT == 2:
            self.links.append(Link(particle1=self.particles[0], particle2=self.particles[1], token=0))
        elif INIT_PARTICLE_AMOUNT > 2:
            for i in range(INIT_PARTICLE_AMOUNT):
                self.links.append(Link(particle1=self.particles[i-1], particle2=self.particles[i], token=0))

    def kill_duplicate_links(self):
        for cur_par in self.particles:
            already_connected = []
            for cur_node in cur_par.nodes.copy():
                if cur_node.other_node.particle in already_connected or cur_node.other_node.particle is cur_par:
                    cur_node.disconnect()
                    cur_node.other_node.disconnect()
                    self.links.remove(cur_node.link)
                else:
                    already_connected.append(cur_node.other_node.particle)


    def run_main_loop(self, iterations=1e15, plotting = True):
        """
        The main loop of the simulation.
        :param iterations: The amount of iterations that are executed
        """

        self.iterations_until_next_plot = PLOT_EVERY_X_ITERATIONS + 1
        # Main loop
        self.gui_dt = 1.0/30.0
        print("total iterations:", iterations)
        while iterations > 0 and self.is_running:
            start_time = time.time()
            self.sim_options = copy.deepcopy(self.sim_options_original)


            # 1 Iteration preparation
            self.prepare()
            for cur_link in self.links:
                cur_link.prepare(sim_options=self.sim_options)
            for cur_par in self.particles:
                cur_par.prepare(sim_options=self.sim_options)

            # 2.5 Decision making
            for cur_par in self.particles:
                cur_par.process_information_and_make_decisions_phase0(sim_options=self.sim_options, data=self.data, all_particles=self.particles)

            # 3 Game Evaluation
            for cur_par in self.particles:
                cur_par.eval_game(sim_options=self.sim_options, data=self.data)


            if self.sim_options.get(SimOptionsEnum.USE_NEW_BEHAVIOR_FOR_PHASE_1):
                for cur_par in self.particles:
                    cur_par.set_new_behavior(sim_options=self.sim_options)

            # 2.5 Decision making
            for cur_par in self.particles:
                cur_par.process_information_and_make_decisions_phase1(sim_options=self.sim_options, data=self.data, all_particles=self.particles)

            if not self.sim_options.get(SimOptionsEnum.USE_NEW_BEHAVIOR_FOR_PHASE_1):
                for cur_par in self.particles:
                    cur_par.set_new_behavior(sim_options=self.sim_options)

            for cur_par in self.particles:
                cur_par.set_repro_token(sim_options=self.sim_options, data=self.data)

            for cur_link in self.links:
                if cur_link.active_value > 0:
                    cur_link.is_active = True
                else:
                    cur_link.is_active = False

            # 4 Swap
            for cur_link in self.links:
                cur_link.try_swap(sim_options=self.sim_options, data=self.data)

            # 5 Move
            for cur_par in self.particles:
                cur_par.move()

            # Reconnection
            if self.sim_options.get(SimOptionsEnum.CAN_RECONNECT):
                for cur_par in self.particles:
                    cur_par.prepare_reconnect(sim_options=self.sim_options)

                for cur_par in self.particles:
                    cur_par.do_reconnect(data=self.data)

                self.kill_duplicate_links()

            # 5.25 New_link
            for cur_par in self.particles:
                cur_par.try_making_new_link(sim_options=self.sim_options, all_links=self.links, data=self.data, pos=self.pos)

            # 6 Reproduction
            for cur_par in self.particles.copy():
                cur_par.reproduce(sim_options=self.sim_options, pos=self.pos,
                                  all_particles=self.particles, all_links=self.links, data=self.data,
                                  vel=self.vel, ph_dict_li=self.particle_heritage_dictionary_last_iteration)



            # 6.5 check death of links
            if self.sim_options.get(SimOptionsEnum.KILL_INACTIVE_LINKS):
                for cur_link in self.links:
                    cur_link.check_death(sim_options=self.sim_options, all_links=self.links,
                                         all_particles=self.particles, data=self.data, dead_particles=self.dead_particles, dead_links=self.dead_links)

            # 7 Check Death
            for cur_par in self.particles.copy():
                cur_par.check_death(sim_options=self.sim_options, all_links=self.links,
                                    all_particles=self.particles, data=self.data, dead_particles=self.dead_particles, dead_links=self.dead_links)

            # 8 Mutate
            mutate_all_each_iteration = self.sim_options.get(SimOptionsEnum.MUTATE_ALL_EACH_ITERATION)
            for cur_par in self.particles:
                if cur_par.to_mutate or mutate_all_each_iteration:
                    cur_par.behavior.mutate(sim_options=self.sim_options)

            # Plotting
            token_amount = sum([cur_par.token for cur_par in self.particles])
            self.data.renormalize(token_amount=token_amount)

            if self.sim_options.get(SimOptionsEnum.META_EVOLUTION):
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

            if plotting:
                self.plot_network()
            else:
                self.calc_some_stats()
                self.graph_plotting_operations_gui()

            global TOKEN_AMOUNT

            crit1 = len(self.particles) < int(INIT_PARTICLE_AMOUNT * 0.3)
            if crit1:
                break

            if CAN_EXIT:
                hastocrit = self.current_iteration > 30
                crit2 = all(np.array(self.data.shannon_entropy[-10:]) < 0.25) and hastocrit
                crit4 = all(np.array(self.data.died_particles_history[-10:]) < 0.01) and hastocrit
                crit5 = all(np.array(self.data.reproduced_particles_history[-10:]) + np.array(self.data.planted_particles_history[-10:]) < 0.01) and hastocrit
                crit6 = all(np.array(self.data.links_per_particle_history[-10:]) < 1.005) and hastocrit
                crit9 = all(np.array(self.data.links_per_particle_history[-10:]) > 6) and hastocrit

                if crit1 or crit2 or crit4 or crit5 or crit6 or crit9:
                    break

            if self.current_iteration > SAVING_THRESHOLD:
                self.sim_options.save(self.name, "best_options")
                break

            self.current_iteration += 1
            iterations -= 1
            end_time = time.time()

            self.gui_dt = end_time - start_time



    def calc_some_stats(self):

        token_amount_particles = sum([cur_par.token for cur_par in self.particles])
        token_amount_links = sum([cur_link.token for cur_link in self.links])

        print("Tokens", token_amount_particles + token_amount_links,
              ", Particles", len(self.particles),
              ", Links", len(self.links),
              ", Iteration", self.current_iteration)


        self.g = nx.Graph()
        for cur_par in self.particles:
            self.g.add_node(cur_par)
        for cur_link in self.links:
            self.g.add_edge(cur_link.node1.particle, cur_link.node2.particle, weight=1)

        tokens = [cur_par.token for cur_par in self.particles]
        tokens_var = float(np.var(tokens))
        particles_at_position = [len(cur_par.walker_position.particles_at_this_position) for cur_par in
                                 self.particles]
        particles_at_position_var = float(np.var(particles_at_position))
        link_amounts = [cur_par.link_amount() for cur_par in self.particles]
        link_amounts_var = float(np.var(link_amounts))
        ages = [cur_par.age for cur_par in self.particles]
        ages_var = float(np.var(ages))
        link_ages = [cur_link.age for cur_link in self.links]
        link_ages_var = float(np.var(link_ages))

        self.data.shannon_entropy.append(shannon_entropy(self.g))

        # Dimensionality calculation
        dims = []
        dims_max = []
        dims_mean = []
        dims_amounts = []
        for cur_par in np.random.choice(self.particles, CHECK_X_PARTICLES_FOR_DIM):
            cur_dims = cur_par.get_dimension_list(all_particles=self.particles)
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

        if self.pos is not None:
            temp_pos = {}
            for cur_par in self.particles:
                temp_pos[cur_par] = self.pos[cur_par]
            self.pos = temp_pos

            temp_vel = {}
            for cur_par in self.particles:
                temp_vel[cur_par] = self.vel[cur_par]
            self.vel = temp_vel

    def graph_plotting_operations_gui(self):
        if GUI == True:
            self.event.set()
            self.g = nx.Graph()
            for cur_par in self.particles:
                self.g.add_node(cur_par)
            for cur_link in self.links:
                self.g.add_edge(cur_link.node1.particle, cur_link.node2.particle, weight=1)

            keys = list(self.particle_heritage_dictionary_last_iteration.keys())
            for key in keys:
                # Pop and print each item
                value = self.particle_heritage_dictionary_last_iteration.pop(key)
                self.particle_heritage_dictionary_gui_queue[key] = value

            self.gui_g = self.g
            self.gui_par_tokens = {particle: particle.token for particle in self.particles}
            self.gui_link_tokens = {(link.node1.particle, link.node2.particle): link.active_value for link in self.links}

            self.event.clear()

    def get_g_and_other(self):
        temp = self.particle_heritage_dictionary_gui_queue.copy()
        self.particle_heritage_dictionary_gui_queue = {}
        return self.gui_g, temp, self.gui_dt, self.gui_par_tokens, self.gui_link_tokens

    def prepare(self):
        """
        Prepare iteration information for data analysis
        """

        # Randomize the order of execution so that no systemic advantages arise
        np.random.shuffle(self.particles)
        np.random.shuffle(self.links)

        self.dead_links = []
        self.dead_particles = []

        # Prepare data arrays for analysis
        self.data.particle_amount_history.append(len(self.particles))
        self.data.link_amount_history.append(len(self.links))
        self.data.links_per_particle_history.append(self.data.link_amount_history[-1]/self.data.particle_amount_history[-1])
        sum_x = sum([1 for cur_par in self.particles if cur_par is cur_par.walker_position])
        self.data.percent_home_history.append(sum_x / len(self.particles))
        self.data.swap_percentage_history.append(0.0)
        self.data.new_links_spawned_history.append(0.0)
        self.data.declined_new_links_history.append(0.0)
        self.data.planted_particles_history.append(0.0)
        self.data.reproduced_particles_history.append(0.0)
        self.data.died_particles_history.append(0.0)
        self.data.kept_repro_tokens_history.append(0.0)

        self.data.allocation_accepted_history.append(0.0)
        self.data.punished_history.append(0.0)

        self.data.reconnection_history.append(0.0)

        self.data.inactive_links_history.append(0.0)

        self.data.repro_token_self_invested_history.append(0.0)
        self.data.token_self_invested_history.append(0.0)
        self.data.repro_token_other_invested_history.append(0.0)
        self.data.token_other_invested_history.append(0.0)
        self.data.win_token_percentage_history.append(0.0)
        self.data.win_repro_token_percentage_history.append(0.0)


    def plot_network(self):
        """
        Plots the network and data, saves images into the outputs folder.
        :param background_color: "black" or "white"
        """
        token_amount_particles = sum([cur_par.token for cur_par in self.particles])
        token_amount_links = sum([cur_link.token for cur_link in self.links])
        print("Tokens", token_amount_particles + token_amount_links,
              ", Particles", len(self.particles),
              ", Links", len(self.links),
              ", Iteration", self.current_iteration)

        self.g = nx.Graph()
        for cur_par in self.particles:
            self.g.add_node(cur_par)
        for cur_link in self.links:
            self.g.add_edge(cur_link.node1.particle, cur_link.node2.particle, weight=1)

        tokens = [cur_par.token for cur_par in self.particles]
        tokens_var = float(np.var(tokens))
        particles_at_position = [len(cur_par.walker_position.particles_at_this_position) for cur_par in
                                 self.particles]
        particles_at_position_var = float(np.var(particles_at_position))
        link_amounts = [cur_par.link_amount() for cur_par in self.particles]
        link_amounts_var = float(np.var(link_amounts))
        ages = [cur_par.age for cur_par in self.particles]
        ages_var = float(np.var(ages))
        link_ages = [cur_link.age for cur_link in self.links]
        link_ages_var = float(np.var(link_ages))

        self.data.shannon_entropy.append(shannon_entropy(self.g))

        # Dimensionality calculation
        dims = []
        dims_max = []
        dims_mean = []
        dims_amounts = []
        for cur_par in np.random.choice(self.particles, CHECK_X_PARTICLES_FOR_DIM):
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
        if self.iterations_until_next_plot <= 0 or self.current_iteration == 0:
            self.iterations_until_next_plot = PLOT_EVERY_X_ITERATIONS
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

        new_pos = nx.fruchterman_reingold_layout(self.g, pos=self.pos, scale=1, iterations=SPRING_MODEL_ITERATIONS, dim=3)


        for i in range(SMOOTH_PLOTS):
            projected_pos = {}
            depth = {}
            self.angle += ANGLE_SHIFT
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
            max_value = max(link_tokens)
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

            if PLOT_WITH_DIAGRAMS:
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

                ax1 = plt.subplot2grid((row_amount_tot, col_amount_tot), (0, 0), colspan=col_amount_main, rowspan=row_amount_tot)
                axs = []

                for jndex in range(col_amount_main, col_amount_tot):
                    for index in range(row_amount_tot):
                        axs.append(plt.subplot2grid((row_amount_tot, col_amount_tot), (index, jndex), colspan=1, rowspan=1))
                ax_index = 0

                plt.subplots_adjust(wspace=0.35, hspace=0.35)
                ax1.set_title(f"Game of Evolution, {self.name}, AT={token_amount_particles + token_amount_links}, "
                              f"{self.sim_options.get_description()}")

                backlash = 0.96


                self.data.max_token_amount_history = max(int(self.data.max_token_amount_history * backlash), max(tokens))
                bins = np.linspace(0.0, self.data.max_token_amount_history, 30)
                axs[ax_index].hist(tokens, rwidth=1, bins=bins, color=linecolor, edgecolor=edgecolor)
                axs[ax_index].set_title(f"Particle Tokens, var={round(tokens_var, 3)}")
                axs[ax_index].set_yscale('log')
                ax_index += 1

                self.data.max_par_conc_history = max(int(self.data.max_par_conc_history * backlash),
                                                     max(particles_at_position))
                bins = np.linspace(0.0, self.data.max_par_conc_history, 30)
                axs[ax_index].hist(particles_at_position, rwidth=1, bins=bins, color=linecolor, edgecolor=edgecolor)
                axs[ax_index].set_title(f"Particle Concentration, var={round(particles_at_position_var, 3)}")
                axs[ax_index].set_yscale('log')
                self.data.max_link_amount_history = max(int(self.data.max_link_amount_history * backlash), max(link_amounts))
                ax_index += 1

                bins = np.linspace(0.0, self.data.max_link_amount_history, 30)
                axs[ax_index].hist(link_amounts, rwidth=1, bins=bins, color=linecolor, edgecolor=edgecolor)
                axs[ax_index].set_title(f"Link Amount Distribution, var={round(link_amounts_var, 3)}")
                axs[ax_index].set_yscale('log')
                ax_index += 1

                self.data.max_age_history = max(int(self.data.max_age_history * backlash), max(ages))
                bins = np.linspace(0.0, self.data.max_age_history, 30)
                axs[ax_index].hist(ages, rwidth=1, bins=bins, color=linecolor, edgecolor=edgecolor)
                axs[ax_index].set_title(f"Particle Age Distribution, var={round(ages_var, 3)}")
                axs[ax_index].set_yscale('log')
                ax_index += 1

                self.data.max_link_age_history = max(int(self.data.max_link_age_history * backlash), max(link_ages))
                bins = np.linspace(0.0, self.data.max_link_age_history, 30)
                axs[ax_index].hist(link_ages, rwidth=1, bins=bins, color=linecolor, edgecolor=edgecolor)
                axs[ax_index].set_title(f"Link Age Distribution, var={round(link_ages_var, 3)}")
                axs[ax_index].set_yscale('log')
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
                    f"Particle and Link Amount {self.data.particle_amount_history[-1]} {self.data.link_amount_history[-1]}")
                axs[ax_index].plot(x_iter, self.data.particle_amount_history, color=linecolor)
                axs[ax_index].plot(x_iter, self.data.link_amount_history, color=linecolor)
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
                axs[ax_index].plot(x_iter, self.data.kept_repro_tokens_history, color="cornflowerblue")
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

                axs[ax_index].set_title(f"Swapped Particles {round(self.data.swap_percentage_history[-1], 4)}")
                axs[ax_index].plot(x_iter, self.data.swap_percentage_history, color=linecolor)
                ax_index += 1

                axs[ax_index].set_title(f"Reconnections {round(self.data.reconnection_history[-1], 4)}")
                axs[ax_index].plot(x_iter, self.data.reconnection_history, color=linecolor)
                ax_index += 1

                axs[ax_index].set_title(f"Token Invested Self B=REPRO")
                axs[ax_index].plot(x_iter, self.data.token_self_invested_history, color=linecolor)
                axs[ax_index].plot(x_iter, self.data.repro_token_self_invested_history, color="cornflowerblue")
                ax_index += 1

                axs[ax_index].set_title(f"Token Invested Other B=REPRO")
                axs[ax_index].plot(x_iter, self.data.token_other_invested_history, color=linecolor)
                axs[ax_index].plot(x_iter, self.data.repro_token_other_invested_history, color="cornflowerblue")
                ax_index += 1

                axs[ax_index].set_title(f"Win Token Percentage B=REPRO")
                axs[ax_index].plot(x_iter, self.data.win_token_percentage_history, color=linecolor)
                axs[ax_index].plot(x_iter, self.data.win_repro_token_percentage_history, color="cornflowerblue")
                ax_index += 1




                for ax in axs:
                    ax.grid(axis="y")

                if len(dead_par_pos) > 0:
                    self.my_draw_networkx_dead_nodes(node_pos=dead_par_pos, rotation_matrix=rotation_matrix, ax=ax1, **options_dead_node)
                if len(self.dead_links) > 0:
                    self.my_draw_networkx_dead_edges(dead_links=self.dead_links, ax=ax1, rotation_matrix=rotation_matrix, **options_dead_edge)
                nx.draw_networkx_nodes(G=g_plot, pos=projected_pos,ax=ax1, **options_node)
                self.my_draw_networkx_edges(G=g_plot, ax=ax1, rotation_matrix=rotation_matrix, **options_edge)
                ax1.margins(x=0.1)
                ax1.set_axis_off()
                ax1.set_xlim(-lim, lim)
                ax1.set_ylim(-lim, lim)

                plt.savefig(rf"outputs\{self.name}\Plot {int(self.current_plot)}, i = {self.current_iteration}.png",
                            facecolor=self.vis_options.backgroundcolor, edgecolor='none')
                self.current_plot += 1
                plt.close()
            else:
                res_x = 1440
                ratio = 2.0
                plt.figure()
                plt.figure(figsize=(9 * ratio, 9 * ratio), dpi=res_x / (9 * ratio))
                """
                if background_color == "black":
                    plt.style.use('dark_background')
                else:
                    plt.style.use('default')
                """
                ax1 = plt.subplot2grid((1, 1), (0, 0), colspan=1, rowspan=1)

                if len(dead_par_pos) > 0:
                    self.my_draw_networkx_dead_nodes(node_pos=dead_par_pos, rotation_matrix=rotation_matrix, ax=ax1,
                                                     **options_dead_node)
                if len(self.dead_links) > 0:
                    self.my_draw_networkx_dead_edges(dead_links=self.dead_links, ax=ax1,
                                                     rotation_matrix=rotation_matrix, **options_dead_edge)
                nx.draw_networkx_nodes(G=g_plot, pos=projected_pos, ax=ax1, **options_node)
                self.my_draw_networkx_edges(G=g_plot, ax=ax1, rotation_matrix=rotation_matrix, **options_edge)
                ax1.margins(x=0.1)
                ax1.set_axis_off()

                ax1.set_xlim(-lim, lim)
                ax1.set_ylim(-lim, lim)
                plt.savefig(rf"outputs\{self.name}\Plot {int(self.current_plot)}, i = {self.current_iteration}.png",
                            facecolor=self.vis_options.backgroundcolor, edgecolor='none')
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




TOKEN_AMOUNT = int(np.random.uniform(TOKEN_AMOUNT_RANGE[0], TOKEN_AMOUNT_RANGE[1]))
ANGLE_SHIFT = np.random.uniform(ANGLE_SHIFT_RANGE[0], ANGLE_SHIFT_RANGE[1])
SMOOTH_PLOTS = int(np.random.uniform(SMOOTH_PLOTS_RANGE[0], SMOOTH_PLOTS_RANGE[1]))
PLOT_EVERY_X_ITERATIONS = int(np.random.uniform(PLOT_EVERY_X_ITERATIONS_RANGE[0], PLOT_EVERY_X_ITERATIONS_RANGE[1]))
ITERATION_AMOUNT = int(FRAME_AMOUNT * PLOT_EVERY_X_ITERATIONS / SMOOTH_PLOTS)  # TEMP

if __name__ == "__main__":
    """
    Runs the simulation according to the set parameters at the beginning of the script.
    """

    print("Iterations:", ITERATION_AMOUNT)
    if LOADING_FOLDER:

        for f in os.listdir(LOADING_FOLDER):
            if f.endswith('.csv'):
                print(f)
                simulation_options = NewSimOptions()
                simulation_options.load(f, LOADING_FOLDER)
                simulation = Simulation(sim_options=simulation_options)
                simulation.run_main_loop(iterations=ITERATION_AMOUNT, plotting = PLOTTING)

    elif SAMPLE_POOL_FOLDER:

        while True:
            sample_files = [f for f in os.listdir(SAMPLE_POOL_FOLDER) if f.endswith('.csv')]
            file = np.random.choice(sample_files)
            simulation_options = NewSimOptions()
            simulation_options.load(file, SAMPLE_POOL_FOLDER)
            simulation_options.set_random_settings(change_settings_probability=CHANGE_SETTINGS_PROBABILITY)
            simulation = Simulation(sim_options=simulation_options)
            simulation.run_main_loop(iterations=ITERATION_AMOUNT, plotting = PLOTTING)


    else:
        if RANDOM_SETTINGS:
            change_settings_probability = 1.0
        else:
            change_settings_probability = 0.0
        if EXPERIMENT_SERIES:
            while True:
                simulation_options = NewSimOptions()
                simulation_options.set_random_settings(change_settings_probability=change_settings_probability)
                simulation = Simulation(sim_options=simulation_options)
                simulation.run_main_loop(iterations=ITERATION_AMOUNT, plotting = PLOTTING)
        else:
            simulation_options = NewSimOptions()
            simulation_options.set_random_settings(change_settings_probability=change_settings_probability)
            simulation = Simulation(sim_options=simulation_options)
            simulation.run_main_loop(iterations=ITERATION_AMOUNT, plotting = PLOTTING)