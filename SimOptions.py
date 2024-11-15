import numpy as np
from enum import Enum, auto

import csv
import os


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
    USE_X_BEH = auto()
    DECISION = auto()

    # LINK = auto()
    NONE = auto()


class SimOptionsEnum(Enum):

    USE_X_FOR_NEW_LINK = auto()
    USE_X_FOR_MOVING_PARTICLES = auto()
    #USE_X_FOR_DEATH_RATING = auto()
    USE_X_FOR_SWAPPING = auto()
    USE_X_FOR_SHIFTING = auto()
    USE_X_FOR_TOKEN_ALLOCATION = auto()
    USE_X_FOR_PLANTING = auto()
    USE_X_FOR_PLANTING_PARTICLE = auto()
    USE_X_FOR_ACCEPTING_LINK = auto()
    USE_X_FOR_INHERITAGING_POSITION = auto()
    USE_X_FOR_KEEPING_REPRO_TOKEN = auto()
    USE_X_FOR_RECONNECTION = auto()
    USE_X_FOR_RECONNECTION_DIRECTION = auto()

    NEW_LINK_DECISION = auto()
    MOVING_PARTICLES_DECISION = auto()
    #DEATH_RATING_DECISION = auto()
    SWAPPING_DECISION = auto()
    SHIFTING_DECISION = auto()
    PLANTING_DECISION = auto()
    X_REPRODUCTION_DECISION = auto()
    PLANTING_PARTICLE_DECISION = auto()
    ACCEPT_NEW_LINK_DECISION = auto()
    INHERITAGING_POSITION_DECISION = auto()
    ACCEPTING_ALLOCATION_DECISION = auto()
    RECONNECTION_DECISION = auto()
    RECONNECTION_DIRECTION_DECISION = auto()

    MOVE_ON_OWN = auto()
    MOVE_ON_OWN_BY_SELF = auto()

    CAN_DECLINE_NEW_LINK = auto()

    MUTATION_CHANCE = auto()

    MUTATE_ALL_EACH_ITERATION = auto()

    KILL_INACTIVE_LINKS = auto()
    INACTIVITY_KILLS_WHICH_PARTICLE = auto()

    META_EVOLUTION_DEC = auto()
    META_EVOLUTION_USE_X_BEH = auto()
    MUTATION_CHANCE_META = auto()

    ACTIVATION_FUNCTION = auto()

    CAN_MOVE_TO_WALKERS_WALKER = auto()
    CAN_MOVE_TO_RANDOM = auto()
    CAN_MOVE_HOME = auto()
    CAN_NOT_MOVE = auto()
    CAN_MOVE_TO_NEIGBHBOR = auto()
    CAN_MOVE_TO_NEIGBHBORS_WALKER = auto()

    CAN_RECONNECT = auto()

    CAN_PLANT = auto()
    CAN_REPRODUCE_AT_HOME = auto()

    CAN_PLANT_ON_SELF = auto()
    CAN_PLANT_ON_WALKER = auto()
    CAN_PLANT_ON_RANDOM = auto()
    CAN_PLANT_ON_NEIGHBOR = auto()
    CAN_PLANT_ON_NEIGBHBORS_WALKER = auto()

    CAN_SWAP = auto()

    BLOTTO_PROB_WINNER = auto()

    WINNER_TOKEN_GO_TO_COMPETITION_PLACE = auto()
    WINNER_HERE_BEHAVIOR_WINS = auto()
    MUTATE_WIN_BEHAVIOR = auto()

    CAN_INHERIT_WALKER_POSITION = auto()
    CAN_MOVE_WHEN_DEATH = auto()
    MOVE_RANDOMLY_WHEN_DEATH = auto()

    KEEPS_ONE_TOKEN_WHEN_REPRODUCTION = auto()

    ONLY_USE_OWN_BEHAVIOR = auto()

    NEW_REPRODUCTION_LINK_PRODUCTION = auto()
    CAN_STILL_SHIFT_AT_NEW_REPRODUCTION = auto()
    REPRODUCTION_LINKS_DECISION = auto()

class NewSimOptions:

    # Neural Network
    INPUT_SCALING = 0.1
    QUANTILE_AMOUNT = 3
    INPUT_AMOUNT = 26 + QUANTILE_AMOUNT * 3 * 4
    OUTPUT_AMOUNT = 36
    LAYER_SIZES = [int(INPUT_AMOUNT * 0.75)] * 2

    name = ""

    def set_random_settings(self, change_settings_probability):
        for key in self.all_options:
            if np.random.rand() < change_settings_probability:
                self.all_options[key].randomize()

    def get(self, enum_key, particle=None):
        single_option = self.all_options[enum_key]
        if enum_key in (SimOptionsEnum.META_EVOLUTION_DEC, SimOptionsEnum.META_EVOLUTION_USE_X_BEH, SimOptionsEnum.ONLY_USE_OWN_BEHAVIOR):
            return single_option.get()

        if self.get(SimOptionsEnum.ONLY_USE_OWN_BEHAVIOR) and single_option.meta_type == MetaEnum.USE_X_BEH:
            first_key = list(single_option.option_possibilities)[0]
            first_value = single_option.option_possibilities[first_key]
            return first_value

        crit1 = self.get(SimOptionsEnum.META_EVOLUTION_DEC) and single_option.meta_type == MetaEnum.DECISION
        crit2 = self.get(SimOptionsEnum.META_EVOLUTION_USE_X_BEH) and single_option.meta_type == MetaEnum.USE_X_BEH
        if crit1 or crit2:
            return particle.behavior.meta_options[enum_key]
        else:
            return single_option.get()


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

        self.name=filename

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

            crit1 = not (self.get(SimOptionsEnum.META_EVOLUTION_DEC) and single_option.meta_type == MetaEnum.DECISION)  # False
            crit2 = not (self.get(SimOptionsEnum.META_EVOLUTION_USE_X_BEH) and single_option.meta_type == MetaEnum.USE_X_BEH) # True
            if (not single_option.can_meta_evolve) or (crit1 and crit2):
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
            crit1 = self.get(SimOptionsEnum.META_EVOLUTION_DEC) and single_option.meta_type == MetaEnum.DECISION
            crit2 = self.get(SimOptionsEnum.META_EVOLUTION_USE_X_BEH) and single_option.meta_type == MetaEnum.USE_X_BEH
            if single_option.can_meta_evolve and (crit1 or crit2):
                final_dict[key] = single_option.get_random_option()
        return final_dict

    def set_all_options(self):
        """_______________________________________________________________"""


        temp_enum = SimOptionsEnum.USE_X_FOR_NEW_LINK
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                 short_name="XNL",
                                                 option="SELF",
                                                 options=["SELF", "NEIGHBOR", "WALKER"],
                                                 probability=[1.0, 1.0, 0.0],
                                                 can_meta_evolve = True,
                                                 meta_type=MetaEnum.USE_X_BEH)

        temp_enum = SimOptionsEnum.USE_X_FOR_MOVING_PARTICLES
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                 short_name="XMP",
                                                 option="SELF",
                                                 options=["SELF", "NEIGHBOR", "WALKER"],
                                                 probability=[1.0, 1.0, 0.0],
                                                 can_meta_evolve = True,
                                                 meta_type=MetaEnum.USE_X_BEH)

        """
        temp_enum = SimOptionsEnum.USE_X_FOR_DEATH_RATING
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                 short_name="XDR",
                                                 option="SELF",
                                                 options=["SELF", "NEIGHBOR", "WALKER"],
                                                 probability=[1.0, 1.0, 0.0],
                                                 can_meta_evolve = True,
                                                 meta_type=MetaEnum.USE_X_BEH)
        """

        temp_enum = SimOptionsEnum.USE_X_FOR_SWAPPING
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                 short_name="XS",
                                                 option="SELF",
                                                 options=["SELF", "NEIGHBOR", "WALKER"],
                                                 probability=[1.0, 1.0, 0.0],
                                                 can_meta_evolve = True,
                                                 meta_type=MetaEnum.USE_X_BEH)

        temp_enum = SimOptionsEnum.USE_X_FOR_SHIFTING
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                 short_name="XSH",
                                                 option="SELF",
                                                 options=["SELF", "NEIGHBOR", "WALKER"],
                                                 probability=[1.0, 1.0, 0.0],
                                                 can_meta_evolve = True,
                                                 meta_type=MetaEnum.USE_X_BEH)

        temp_enum = SimOptionsEnum.USE_X_FOR_TOKEN_ALLOCATION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                 short_name="XTA",
                                                 option="SELF",
                                                 options=["SELF", "NEIGHBOR", "WALKER"],
                                                 probability=[1.0, 1.0, 0.0],
                                                 can_meta_evolve = True,
                                                 meta_type=MetaEnum.USE_X_BEH)

        temp_enum = SimOptionsEnum.USE_X_FOR_PLANTING
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                 short_name="XP",
                                                 option="SELF",
                                                 options=["SELF", "NEIGHBOR", "WALKER"],
                                                 probability=[1.0, 1.0, 0.0],
                                                 can_meta_evolve = True,
                                                 meta_type=MetaEnum.USE_X_BEH)

        temp_enum = SimOptionsEnum.USE_X_FOR_PLANTING_PARTICLE
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                 short_name="XPP",
                                                 option="SELF",
                                                 options=["SELF", "NEIGHBOR", "WALKER"],
                                                 probability=[1.0, 1.0, 0.0],
                                                 can_meta_evolve = True,
                                                 meta_type=MetaEnum.USE_X_BEH)

        temp_enum = SimOptionsEnum.USE_X_FOR_ACCEPTING_LINK
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                 short_name="XAL",
                                                 option="WALKER",
                                                 options=["WALKER", "SELF", "NEIGHBOR"],
                                                 probability=[1.0, 0.0, 1.0],
                                                 can_meta_evolve = True,
                                                 meta_type=MetaEnum.USE_X_BEH)

        temp_enum = SimOptionsEnum.USE_X_FOR_INHERITAGING_POSITION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                 short_name="XIP",
                                                 option="SELF",
                                                 options=["SELF", "NEIGHBOR", "WALKER"],
                                                 probability=[1.0, 1.0, 0.0],
                                                 can_meta_evolve = True,
                                                 meta_type=MetaEnum.USE_X_BEH)


        temp_enum = SimOptionsEnum.USE_X_FOR_KEEPING_REPRO_TOKEN
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                 short_name="XKRT",
                                                 option="SELF",
                                                 options=["SELF", "NEIGHBOR", "WALKER"],
                                                 probability=[1.0, 1.0, 0.0],
                                                 can_meta_evolve = True,
                                                 meta_type=MetaEnum.USE_X_BEH)

        temp_enum = SimOptionsEnum.USE_X_FOR_RECONNECTION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                 short_name="XR",
                                                 option="SELF",
                                                 options=["SELF", "NEIGHBOR", "WALKER"],
                                                 probability=[1.0, 1.0, 0.0],
                                                 can_meta_evolve = True,
                                                 meta_type=MetaEnum.USE_X_BEH)

        temp_enum = SimOptionsEnum.USE_X_FOR_RECONNECTION_DIRECTION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                 short_name="XRD",
                                                 option="SELF",
                                                 options=["SELF", "NEIGHBOR", "WALKER"],
                                                 probability=[1.0, 1.0, 0.0],
                                                 can_meta_evolve = True,
                                                 meta_type=MetaEnum.USE_X_BEH)

        """_______________________________________________________________"""

        temp_enum = SimOptionsEnum.NEW_LINK_DECISION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="NLD",
                                                      option="PROB",
                                                      options=["MAX", "PROB", "RANDOM"],
                                                      probability=[3.0, 3.0, 1.0],
                                                      can_meta_evolve = True,
                                                      meta_type=MetaEnum.DECISION)

        temp_enum = SimOptionsEnum.MOVING_PARTICLES_DECISION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="MPD",
                                                      option="PROB",
                                                      options=["MAX", "PROB", "RANDOM"],
                                                      probability=[3.0, 3.0, 1.0],
                                                      can_meta_evolve = True,
                                                      meta_type=MetaEnum.DECISION)

        """
        temp_enum = SimOptionsEnum.DEATH_RATING_DECISION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="DRD",
                                                      option="PROB",
                                                      options=["MAX", "PROB", "RANDOM"],
                                                      probability=[3.0, 3.0, 1.0],
                                                      can_meta_evolve = True,
                                                      meta_type=MetaEnum.DECISION)

        """

        temp_enum = SimOptionsEnum.SWAPPING_DECISION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="SD",
                                                      option="PROB",
                                                      options=["MAX", "PROB", "RANDOM"],
                                                      probability=[3.0, 3.0, 1.0],
                                                      can_meta_evolve = True,
                                                      meta_type=MetaEnum.DECISION)

        temp_enum = SimOptionsEnum.SHIFTING_DECISION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="SHD",
                                                      option="PROB",
                                                      options=["MAX", "PROB", "RANDOM"],
                                                      probability=[3.0, 3.0, 1.0],
                                                      can_meta_evolve = True,
                                                      meta_type=MetaEnum.DECISION)


        temp_enum = SimOptionsEnum.PLANTING_DECISION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="PD",
                                                      option="PROB",
                                                      options=["MAX", "PROB", "RANDOM"],
                                                      probability=[3.0, 3.0, 1.0],
                                                      can_meta_evolve = True,
                                                      meta_type=MetaEnum.DECISION)

        temp_enum = SimOptionsEnum.X_REPRODUCTION_DECISION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="XRD",
                                                      option="PROB",
                                                      options=["MAX", "PROB", "RANDOM"],
                                                      probability=[3.0, 3.0, 1.0],
                                                      can_meta_evolve = True,
                                                      meta_type=MetaEnum.DECISION)

        temp_enum = SimOptionsEnum.PLANTING_PARTICLE_DECISION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="PPD",
                                                      option="PROB",
                                                      options=["MAX", "PROB", "RANDOM"],
                                                      probability=[3.0, 3.0, 1.0],
                                                      can_meta_evolve = True,
                                                      meta_type=MetaEnum.DECISION)


        temp_enum = SimOptionsEnum.ACCEPT_NEW_LINK_DECISION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="ANLD",
                                                      option="PROB",
                                                      options=["MAX", "PROB", "RANDOM"],
                                                      probability=[3.0, 3.0, 1.0],
                                                      can_meta_evolve = True,
                                                      meta_type=MetaEnum.DECISION)


        temp_enum = SimOptionsEnum.INHERITAGING_POSITION_DECISION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="IPD",
                                                      option="PROB",
                                                      options=["MAX", "PROB", "RANDOM"],
                                                      probability=[3.0, 3.0, 1.0],
                                                      can_meta_evolve = True,
                                                      meta_type=MetaEnum.DECISION)

        temp_enum = SimOptionsEnum.ACCEPTING_ALLOCATION_DECISION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="IPD",
                                                      option="PROB",
                                                      options=["MAX", "PROB", "RANDOM"],
                                                      probability=[3.0, 3.0, 1.0],
                                                      can_meta_evolve = True,
                                                      meta_type=MetaEnum.DECISION)


        temp_enum = SimOptionsEnum.RECONNECTION_DECISION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="RD",
                                                      option="PROB",
                                                      options=["MAX", "PROB", "RANDOM"],
                                                      probability=[3.0, 3.0, 1.0],
                                                      can_meta_evolve=True,
                                                      meta_type=MetaEnum.DECISION)

        temp_enum = SimOptionsEnum.RECONNECTION_DIRECTION_DECISION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="RDD",
                                                      option="PROB",
                                                      options=["MAX", "PROB", "RANDOM"],
                                                      probability=[3.0, 3.0, 1.0],
                                                      can_meta_evolve=True,
                                                      meta_type=MetaEnum.DECISION)



        """_______________________________________________________________"""

        temp_enum = SimOptionsEnum.MOVE_ON_OWN
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="MOO",
                                                      option=True,
                                                      options=[True, False],
                                                      probability=[1.0, 1.0],
                                                      can_meta_evolve=True,
                                                      meta_type=MetaEnum.PARTICLE)

        temp_enum = SimOptionsEnum.MOVE_ON_OWN_BY_SELF
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="MOOBS",
                                                      option=True,
                                                      options=[True, False],
                                                      probability=[1.0, 1.0],
                                                      can_meta_evolve=False,
                                                      meta_type=MetaEnum.NONE)


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
                                                      option=0.3,
                                                      options= [0.3],#[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                                      probability= [1.0], #[1.0, 0.9, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                      can_meta_evolve = False,
                                                      meta_type=MetaEnum.NONE)

        temp_enum = SimOptionsEnum.MUTATE_ALL_EACH_ITERATION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="MAEI",
                                                      option=False,
                                                      options=[True, False],
                                                      probability=[1.0, 0.0],
                                                      can_meta_evolve = False,
                                                      meta_type=MetaEnum.NONE)

        temp_enum = SimOptionsEnum.KILL_INACTIVE_LINKS
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="KIL",
                                                      option=True,
                                                      options=[True, False],
                                                      probability=[1.0, 0.0],   #TODO relevant compettion
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

        temp_enum = SimOptionsEnum.META_EVOLUTION_DEC
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="ME",
                                                      option=True,
                                                      options=[True, False],
                                                      probability=[0.0, 1.0],   # TODO
                                                      can_meta_evolve = False,
                                                      meta_type=MetaEnum.NONE)

        temp_enum = SimOptionsEnum.META_EVOLUTION_USE_X_BEH
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="MEXB",
                                                      option=True,
                                                      options=[True, False],
                                                      probability=[0.0, 2.0],   #TODO
                                                      can_meta_evolve = False,
                                                      meta_type=MetaEnum.NONE)

        temp_enum = SimOptionsEnum.MUTATION_CHANCE_META
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="MCM",
                                                      option=0.3,
                                                      options= [0.3],# [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  #TODO
                                                      probability=[1], #[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
                                                      can_meta_evolve=False,
                                                      meta_type=MetaEnum.NONE)

        temp_enum = SimOptionsEnum.ACTIVATION_FUNCTION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="ACF",
                                                      option="SIGMOID",
                                                      options=["RELU", "SIGMOID", "TANH"],
                                                      probability=[0.0, 1.0, 0.0],
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
                                                      probability=[3.0, 0.0],
                                                      can_meta_evolve=False,
                                                      meta_type=MetaEnum.NONE)

        temp_enum = SimOptionsEnum.CAN_MOVE_TO_NEIGBHBOR
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="CMN",
                                                      option=True,
                                                      options=[False, True],
                                                      probability=[1.0, 4.0],
                                                      can_meta_evolve=False,
                                                      meta_type=MetaEnum.NONE)

        temp_enum = SimOptionsEnum.CAN_MOVE_TO_NEIGBHBORS_WALKER
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="CMNW",
                                                      option=True,
                                                      options=[False, True],
                                                      probability=[1.0, 1.0],
                                                      can_meta_evolve=False,
                                                      meta_type=MetaEnum.NONE)

        temp_enum = SimOptionsEnum.CAN_MOVE_HOME
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="CMH",
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
                                                      probability=[0.0, 1.0],   #TODO
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

        temp_enum = SimOptionsEnum.CAN_REPRODUCE_AT_HOME
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="CRAH",
                                                      option=True,
                                                      options=[True, False],
                                                      probability=[1.0, 0.0],   #TODO
                                                      can_meta_evolve=False,
                                                      meta_type=MetaEnum.NONE)

        temp_enum = SimOptionsEnum.CAN_PLANT_ON_SELF
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="CPS",
                                                      option=True,
                                                      options=[True, False],
                                                      probability=[1.0, 1.0],
                                                      can_meta_evolve=False,
                                                      meta_type=MetaEnum.NONE)

        temp_enum = SimOptionsEnum.CAN_PLANT_ON_WALKER
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="CPW",
                                                      option=True,
                                                      options=[True, False],
                                                      probability=[1.0, 1.0],
                                                      can_meta_evolve=False,
                                                      meta_type=MetaEnum.NONE)

        temp_enum = SimOptionsEnum.CAN_PLANT_ON_RANDOM
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="CPR",
                                                      option=True,
                                                      options=[True, False],
                                                      probability=[0.0, 3.0],
                                                      can_meta_evolve=False,
                                                      meta_type=MetaEnum.NONE)

        temp_enum = SimOptionsEnum.CAN_PLANT_ON_NEIGHBOR
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="CPN",
                                                      option=True,
                                                      options=[True, False],
                                                      probability=[1.0, 1.0],
                                                      can_meta_evolve=False,
                                                      meta_type=MetaEnum.NONE)

        temp_enum = SimOptionsEnum.CAN_PLANT_ON_NEIGBHBORS_WALKER
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="CPNW",
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

        temp_enum = SimOptionsEnum.BLOTTO_PROB_WINNER
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="BPW",
                                                      option=True,
                                                      options=[True, False],
                                                      probability=[1.0, 1.0],   #TODO what is the influence?
                                                      can_meta_evolve=False,
                                                      meta_type=MetaEnum.NONE)


        temp_enum = SimOptionsEnum.WINNER_TOKEN_GO_TO_COMPETITION_PLACE
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="WTGTCP",
                                                      option=False,
                                                      options=[False, True],
                                                      probability=[0.0, 1.0],   #TODO always 100% so set to 0% and see what happens?
                                                      can_meta_evolve=False,
                                                      meta_type=MetaEnum.NONE)


        temp_enum = SimOptionsEnum.WINNER_HERE_BEHAVIOR_WINS
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="WBW",
                                                      option=False,
                                                      options=[True, False],
                                                      probability=[1.0, 0.0],   #TODO
                                                      can_meta_evolve=False,
                                                      meta_type=MetaEnum.NONE)


        temp_enum = SimOptionsEnum.MUTATE_WIN_BEHAVIOR
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="MWB",
                                                      option=True,
                                                      options=[True, False],
                                                      probability=[1.0, 0.0],   # TODO
                                                      can_meta_evolve=False,
                                                      meta_type=MetaEnum.NONE)


        temp_enum = SimOptionsEnum.CAN_INHERIT_WALKER_POSITION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="CIWP",
                                                      option=True,
                                                      options=[True, False],
                                                      probability=[1.0, 1.0],
                                                      can_meta_evolve=False,
                                                      meta_type=MetaEnum.NONE)

        temp_enum = SimOptionsEnum.CAN_MOVE_WHEN_DEATH
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="CMWD",
                                                      option=True,
                                                      options=[True, False],
                                                      probability=[1.0, 1.0],
                                                      can_meta_evolve=False,
                                                      meta_type=MetaEnum.NONE)

        temp_enum = SimOptionsEnum.MOVE_RANDOMLY_WHEN_DEATH
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="MRWD",
                                                      option=False,
                                                      options=[False, True],
                                                      probability=[1.0, 1.0],
                                                      can_meta_evolve=False,
                                                      meta_type=MetaEnum.NONE)


        temp_enum = SimOptionsEnum.KEEPS_ONE_TOKEN_WHEN_REPRODUCTION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="K1TWR",
                                                      option=False,
                                                      options=[False, True],
                                                      probability=[1.0, 0.0],   # TODO competition relevant
                                                      can_meta_evolve=False,
                                                      meta_type=MetaEnum.NONE)

        temp_enum = SimOptionsEnum.ONLY_USE_OWN_BEHAVIOR
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="OUOB",
                                                      option=False,
                                                      options=[False, True],
                                                      probability=[0.0, 1.0],   #TODO
                                                      can_meta_evolve=False,
                                                      meta_type=MetaEnum.NONE)

        temp_enum = SimOptionsEnum.NEW_REPRODUCTION_LINK_PRODUCTION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="NRLP",
                                                      option=False,
                                                      options=[False, True],
                                                      probability=[1.0, 1.0],
                                                      can_meta_evolve=False,
                                                      meta_type=MetaEnum.NONE)

        temp_enum = SimOptionsEnum.CAN_STILL_SHIFT_AT_NEW_REPRODUCTION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="CSSANR",
                                                      option=False,
                                                      options=[False, True],
                                                      probability=[1.0, 1.0],
                                                      can_meta_evolve=False,
                                                      meta_type=MetaEnum.NONE)

        temp_enum = SimOptionsEnum.REPRODUCTION_LINKS_DECISION
        self.all_options[temp_enum] = SingleSimOption(name=temp_enum,
                                                      short_name="RLD",
                                                      option="PROB",
                                                      options=["MAX", "PROB", "RANDOM"],
                                                      probability=[3.0, 3.0, 1.0],
                                                      can_meta_evolve = True,
                                                      meta_type=MetaEnum.DECISION)


