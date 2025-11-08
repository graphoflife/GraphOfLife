import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
import datetime as datetime
import os
import pickle

LOADING_FOLDER = "sample_pool"




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

    FREE_LINKS = True
    MOVE_HOME_AFTER_NEW_LINK = True
    PAY_TAX = True
    TAX_REPRODUCES = True
    GO_HOME_IF_ALREADY_CONNECTED = True

    INHERIT_REFERENCE_WHEN_BIRTH = False
    INHERIT_REFERENCE_WHEN_DEATH = False
    REPRODUCTION_WITH_ONE_TOKEN = False
    ALLOCATION_OF_ONE_TOKEN = False

    ALLOCATION_GAME = False
    BIGGER_ALLOCATION_WINS = True
    TOKEN_ALLOCATION_DECISION = "DIST"  # MAX, DIST, PROB, RANDOM

    MUTATION_CHANCE = 0.75

    # Neural Network
    INPUT_EXPONENT = -1
    QUANTILE_AMOUNT = 3
    INPUT_AMOUNT = 39 + QUANTILE_AMOUNT * 3 * 4
    OUTPUT_AMOUNT = 19
    LAYER_SIZES = [int(INPUT_AMOUNT * 1.3)] * 2
    SEE = np.vstack([1] * INPUT_AMOUNT)

    def set_random_settings(self, change_settings_probability):
        """
        Sets the settings randomly according to preset probabilities.
        """
        print(f"changed settings with probability of {change_settings_probability}")
        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 1.0, 1.0])
            self.USE_X_FOR_GOING_HOME = np.random.choice(["SELF", "OTHER", "POSITION"], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 1.0, 1.0])
            self.USE_X_FOR_MOVING_PARTICLES = np.random.choice(["SELF", "OTHER", "POSITION"], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 1.0, 1.0])
            self.USE_X_FOR_DEATH_RATING = np.random.choice(["SELF", "OTHER", "POSITION"], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 1.0, 1.0])
            self.USE_X_FOR_SWAPPING = np.random.choice(["SELF", "OTHER", "POSITION"], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 1.0, 1.0])
            self.USE_X_FOR_SHIFTING = np.random.choice(["SELF", "OTHER", "POSITION"], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 1.0, 1.0])
            self.USE_X_FOR_RPS = np.random.choice(["SELF", "OTHER", "POSITION"], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 1.0, 1.0])
            self.USE_X_FOR_TOKEN_ALLOCATION = np.random.choice(["SELF", "OTHER", "POSITION"], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 1.0, 1.0])
            self.USE_X_FOR_KEEP_TOKENS = np.random.choice(["SELF", "OTHER", "POSITION"], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 1.0, 1.0])
            self.USE_X_FOR_PLANTING = np.random.choice(["SELF", "OTHER", "POSITION"], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 1.0, 1.0])
            self.USE_X_FOR_X_REPRODUCTION = np.random.choice(["SELF", "OTHER", "POSITION"], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 1.0, 1.0])
            self.USE_X_FOR_PLANTING_PARTICLE = np.random.choice(["SELF", "OTHER", "POSITION"], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 1.0, 1.0])
            self.USE_X_FOR_ACCEPTING_MOVE = np.random.choice(["SELF", "OTHER", "POSITION"], p=p / p.sum())

        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 1.0, 1.0])
            self.REPRODUCE_X = np.random.choice(["SELF", "OTHER", "POSITION"], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 1.0, 1.0])
            self.PLANT_ON_X = np.random.choice(["SELF", "OTHER", "POSITION"], p=p / p.sum())

        if np.random.rand() < change_settings_probability:
            p = np.array([3.0, 1.0, 1.0])
            self.GOING_HOME_DECISION = np.random.choice(["MAX", "PROB", "RANDOM"], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([3.0, 1.0, 1.0])
            self.MOVING_PARTICLES_DECISION = np.random.choice(["MAX", "PROB", "RANDOM"], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([3.0, 1.0, 1.0])
            self.DEATH_RATING_DECISION = np.random.choice(["MAX", "PROB", "RANDOM"], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([3.0, 1.0, 1.0])
            self.SWAPPING_DECISION = np.random.choice(["MAX", "PROB", "RANDOM"], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([3.0, 1.0, 1.0])
            self.SHIFTING_DECISION = np.random.choice(["MAX", "PROB", "RANDOM"], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([3.0, 1.0, 1.0])
            self.RPS_DECISION = np.random.choice(["MAX", "PROB", "RANDOM"], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([3.0, 1.0, 1.0])
            self.PLANTING_DECISION = np.random.choice(["MAX", "PROB", "RANDOM"], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([3.0, 1.0, 1.0])
            self.X_REPRODUCTION_DECISION = np.random.choice(["MAX", "PROB", "RANDOM"], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([3.0, 1.0, 1.0])
            self.PLANTING_PARTICLE_DECISION = np.random.choice(["MAX", "PROB", "RANDOM"], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([3.0, 1.0, 1.0])
            self.ACCEPTING_MOVE_DECISION = np.random.choice(["MAX", "PROB", "RANDOM"], p=p / p.sum())

        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 1.0])
            self.DRAW_REPRODUCES = np.random.choice([True, False], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 1.0])
            self.DRAW_FORWARDS = np.random.choice([True, False], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 1.0])
            self.ALWAYS_DRAW = np.random.choice([True, False], p=p / p.sum())

        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 1.0])
            self.WIN_LOSER_REPRODUCES = np.random.choice([True, False], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 1.0])
            self.WIN_WINS_EVERYTHING = np.random.choice([True, False], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 1.0, 1.0, 1.0])
            self.WIN_DISTRIBUTION_TYPE = np.random.choice(["SR", "OR", "BT", "BR"], p=p / p.sum())

        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 1.0, 1.0, 1.0])
            self.TAKE_TOKEN_FROM_X_FOR_MOVE = np.random.choice(["BT", "BR", "T", "R"], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 1.0])
            self.CAN_USE_LAST_TOKEN_FOR_MOVE = np.random.choice([True, False], p=p / p.sum())

        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 1.0])
            self.MOVE_ON_OWN = np.random.choice([True, False], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 1.0])
            self.SHIFT_OTHER = np.random.choice([True, False], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 1.0])
            self.CAN_SWAP = np.random.choice([True, False], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 1.0])
            self.SWAP_SWAPS_TOKEN = np.random.choice([True, False], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 1.0])
            self.ADD_RANDOM_VALUES = np.random.choice([True, False], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 1.0])
            self.CAN_PLANT = np.random.choice([True, False], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 1.0])
            self.CAN_KEEP_TOKENS = np.random.choice([True, False], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 1.0])
            self.SINGLE_RPS = np.random.choice([True, False], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 3.0])
            self.USE_ALL_TOKEN_FOR_REPRODUCTION = np.random.choice([True, False], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 1.0])
            self.MUTATE_WHEN_TOKEN_NOT_ENOUGH = np.random.choice([True, False], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 1.0])
            self.REPRO_USES_TOKEN_FOR_LINK = np.random.choice([True, False], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([3.0, 1.0])
            self.SEE_EVERYTHING = np.random.choice([True, False], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 1.0])
            self.CAN_DECLINE_MOVE = np.random.choice([True, False], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 1.0])
            self.MOVE_HOME_IF_DECLINED = np.random.choice([True, False], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 1.0])
            self.ACCEPT_MOVE_HOME_ON_OWN = np.random.choice([True, False], p=p / p.sum())

        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 1.0])
            self.FREE_LINKS = np.random.choice([True, False], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 1.0])
            self.MOVE_HOME_AFTER_NEW_LINK = np.random.choice([True, False], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 1.0])
            self.PAY_TAX = np.random.choice([True, False], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 1.0])
            self.TAX_REPRODUCES = np.random.choice([True, False], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 1.0])
            self.GO_HOME_IF_ALREADY_CONNECTED = np.random.choice([True, False], p=p / p.sum())

        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 1.0])
            self.INHERIT_REFERENCE_WHEN_BIRTH = np.random.choice([True, False], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 1.0])
            self.INHERIT_REFERENCE_WHEN_DEATH = np.random.choice([True, False], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 3.0])
            self.REPRODUCTION_WITH_ONE_TOKEN = np.random.choice([True, False], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 3.0])
            self.ALLOCATION_OF_ONE_TOKEN = np.random.choice([True, False], p=p / p.sum())

        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 2.0])
            self.ALLOCATION_GAME = np.random.choice([True, False], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 1.0])
            self.BIGGER_ALLOCATION_WINS = np.random.choice([True, False], p=p / p.sum())
        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 2.0, 2.0, 2.0])
            self.TOKEN_ALLOCATION_DECISION = np.random.choice(["DIST", "MAX", "PROB", "RANDOM"], p=p / p.sum())

        if np.random.rand() < change_settings_probability:
            p = 0.75
            self.SEE = np.vstack([int(np.random.random() < p) for _ in range(self.INPUT_AMOUNT)])
        if np.random.rand() < change_settings_probability:
            p = np.array([1.0, 1.0, 1.0, 1.0])
            self.MUTATION_CHANCE = np.random.choice([0.25, 0.5, 0.75, 1.0], p=p/p.sum())

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
                      f"MHD={self.MOVE_HOME_IF_DECLINED},AMHOO={self.ACCEPT_MOVE_HOME_ON_OWN}\n" \
                      f"FL={self.FREE_LINKS},MHANL={self.MOVE_HOME_AFTER_NEW_LINK},PT={self.PAY_TAX}," \
                      f"TR={self.TAX_REPRODUCES},GHIAC={self.GO_HOME_IF_ALREADY_CONNECTED}," \
                      f"IRB={self.INHERIT_REFERENCE_WHEN_BIRTH},IRD={self.INHERIT_REFERENCE_WHEN_DEATH}," \
                      f"R1T={self.REPRODUCTION_WITH_ONE_TOKEN},A1T={self.ALLOCATION_OF_ONE_TOKEN}," \
                      f"AG={self.ALLOCATION_GAME},BAW={self.BIGGER_ALLOCATION_WINS}," \
                      f"TAD={self.TOKEN_ALLOCATION_DECISION}\nS=[ {see_text}]" \

        return description


all_simulation_options = []
for f in os.listdir(LOADING_FOLDER):
    if f.endswith('.obj'):
        print(f)
        with open(rf"{LOADING_FOLDER}\{f}", 'rb') as handle:
            simulation_options: SimOptions = pickle.load(handle)
        all_simulation_options.append(simulation_options)

result_dir = {}
for cur_simulation_option in all_simulation_options:
    cur_vars = vars(cur_simulation_option)
    for cur_option in list(cur_vars.items()):
        cur_option_name = cur_option[0]
        cur_option_set = cur_option[1]
        if cur_option_name != "SEE":
            if cur_option_name not in list(result_dir.keys()):
                result_dir[cur_option_name] = {}
            if cur_option_set not in list(result_dir[cur_option_name].keys()):
                result_dir[cur_option_name][cur_option_set] = 0
            result_dir[cur_option_name][cur_option_set] += 1

print(result_dir)