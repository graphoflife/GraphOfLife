import pickle
from Data import *
from SimOptions import *

# TODO Recreate current diagrams

# TODO make analysis per node

# TODO find reason for death in p2... part of smaller network or no tokens there after game?

# TODO where did they move their walkers? where are they now?

# try to correlate any kind of data, for example link amounts of walker positions and so on
# extract as much information as possible

# think about minimal amount of information needed to store whats happening such that it can
# still be reconstructed what happened, withouth the neural networks. How does this scale as the simulation gets bigger.



# how many tokens are redistributed?

# which parts of the network died because of smaller networks

# get amount of tokens that the newly spawned

# understand with who they make links, do they just always try to do it?
# get number of possible nodes that could make a link vs number of actually made links

# TODO power law exponents

# TODO make analysis (correlation, variation, covariation, dependant on rules, eigenwerte of adjancency matrix)

# TODO make analysis of time change of certain values

# TODO many scatter plots (one node = one point with x and y with two information about node)
# Different groups of nodes with different colors (survivors vs deaths and so on)

# TODO make analysis of competition (maybe just of winner)

# TODO compare single runs with competitions (but all with same option)... does competition have influence of eovlution

# TODO NOPRIO recreate current graph video evolution
# TODO NOPRIO recreate current evolutioanry graph



def add_to_counter_dict(dict, value, how_much = 1):
    if value in dict:
        dict[value] += how_much
    else:
        dict[value] = how_much

def load_sim_options(self, filename: str, foldername:str):

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


i = 500
file_path = fr"E:\GraphOfLifeOuputs\SingleRuns\v016a_graph_of_life_2025_05_03_20_05_11_273525\Pickle_Simulation_Information\Iteration XXX.pkl"


with open(file_path.replace("XXX",str(i-1)), 'rb') as file:
    last_step = pickle.load(file)

with open(file_path.replace("XXX",str(i)), 'rb') as file:
    cur_step = pickle.load(file)


node_amount_before_p1 = len(last_step.nodes_p2)
node_amount_after_p1 = len(cur_step.nodes_p1)
node_amount_after_p2 = len(cur_step.nodes_p2)

node_deaths_after_p1 = len(cur_step.dead_nodes_p1)
node_deaths_after_p2 = len(cur_step.dead_nodes_p2)

edge_amount_before_p1 = len(last_step.dead_nodes_p2)
edge_amount_after_p1 = len(cur_step.dead_nodes_p1)
edge_amount_after_p2 = len(cur_step.dead_nodes_p2)

reproduced_nodes_amount_p1 = len(cur_step.reproduced_nodes_p1)

percentage_of_nodes_that_reproduced = reproduced_nodes_amount_p1/node_amount_before_p1

# They died because they allocated all their tokens to the child -> ripping apart network and killing more
original_nodes_that_died_after_p1 = list(set(cur_step.dead_nodes_p1) & set(last_step.nodes_p2))
original_node_amount_that_died_after_p1 = len(original_nodes_that_died_after_p1)
percentage_of_original_nodes_that_died_after_p1 = original_node_amount_that_died_after_p1/node_amount_before_p1

newly_reproduced_nodes = [cur_reproduction[1] for cur_reproduction in cur_step.reproduced_nodes_p1]
newly_reproduced_nodes_that_instantly_died = list(set(cur_step.dead_nodes_p1) & set(newly_reproduced_nodes))
# they died because they were part of the smaller network
newly_reproduced_node_amount_that_instantly_died = len(newly_reproduced_nodes_that_instantly_died)
percentage_of_newly_spawned_nodes_that_instantly_died_p1 = newly_reproduced_node_amount_that_instantly_died/reproduced_nodes_amount_p1

#sanity check 1
if (node_amount_before_p1 + reproduced_nodes_amount_p1 - node_deaths_after_p1 == node_amount_after_p1) == False:
    print("Sanity Check  1 failed, this should not be true, find mistake")
#sanity check 2
if (node_amount_after_p1 - node_deaths_after_p2 == node_amount_after_p2) == False:
    print("Sanity Check  2 failed, this should not be true, find mistake")
#sanity check 3
if (newly_reproduced_node_amount_that_instantly_died + original_node_amount_that_died_after_p1 == node_deaths_after_p1) == False:
    print("Sanity Check  3 failed, this should not be true, find mistake")

def get_token_distribution(node_tokens):
    counter_dict_to_fill = {}
    for cur_node_token in node_tokens:
        token = cur_node_token[1]
        add_to_counter_dict(counter_dict_to_fill, token)
    counter_dict_to_fill = dict(sorted(counter_dict_to_fill.items()))

    return counter_dict_to_fill

token_distribution_before_p1_counter_dict = get_token_distribution(last_step.node_tokens_p2)
token_distribution_after_p1_counter_dict =  get_token_distribution(cur_step.node_tokens_p1)
token_distribution_after_p2_counter_dict =  get_token_distribution(cur_step.node_tokens_p2)

def get_walker_concentration_distribution(node_walkers):
    counter_dict_to_fill = {}
    node_walker_pointers_dict = {}
    for cur_node_walker in node_walkers:
        walker = cur_node_walker[1]
        add_to_counter_dict(node_walker_pointers_dict, walker)

    for cur_node_walker_pointer in node_walker_pointers_dict.items():
        walker_amount = cur_node_walker_pointer[1]
        add_to_counter_dict(counter_dict_to_fill, walker_amount)
    counter_dict_to_fill = dict(sorted(counter_dict_to_fill.items()))
    return counter_dict_to_fill

walker_concentration_distribution_before_p1_counter_dict = get_walker_concentration_distribution(last_step.node_walkers_p2)
walker_concentration_distribution_after_p1_counter_dict = get_walker_concentration_distribution(cur_step.node_walkers_p1)
walker_concentration_distribution_after_p2_counter_dict = get_walker_concentration_distribution(cur_step.node_walkers_p2)


def get_link_amount_distribution(edges):
    nodes_link_count_dict = {}
    for cur_edge in edges:
        add_to_counter_dict(nodes_link_count_dict, cur_edge[1])
        add_to_counter_dict(nodes_link_count_dict, cur_edge[2])

    counter_dict_to_fill = {}
    for cur_node_link_amount in nodes_link_count_dict.items():
        add_to_counter_dict(counter_dict_to_fill, cur_node_link_amount[1])
    counter_dict_to_fill = dict(sorted(counter_dict_to_fill.items()))
    return counter_dict_to_fill

link_amount_distribution_before_p1_counter_dict = get_link_amount_distribution(last_step.edges_p2)
link_amount_distribution_after_p1_counter_dict = get_link_amount_distribution(cur_step.edges_p1)
link_amount_distribution_after_p2_counter_dict = get_link_amount_distribution(cur_step.edges_p2)


def get_node_age_distribution(node_ages):
    counter_dict_to_fill = {}
    for cur_node_age in node_ages:
        add_to_counter_dict(counter_dict_to_fill, cur_node_age[1])
    counter_dict_to_fill = dict(sorted(counter_dict_to_fill.items()))
    return counter_dict_to_fill

node_age_distribution_before_p1_counter_dict = get_node_age_distribution(last_step.node_ages_p2)
node_age_distribution_after_p2_counter_dict = get_node_age_distribution(cur_step.node_ages_p2)

def get_behavior_age_distribution(behavior_ages):
    counter_dict_to_fill = {}
    for cur_node_age in behavior_ages:
        add_to_counter_dict(counter_dict_to_fill, cur_node_age[1])
    counter_dict_to_fill = dict(sorted(counter_dict_to_fill.items()))
    return counter_dict_to_fill

behavior_age_distribution_before_p1_counter_dict = get_node_age_distribution(last_step.behavior_ages_p2)
behavior_age_distribution_after_p2_counter_dict = get_node_age_distribution(cur_step.behavior_ages_p2)

def get_token_attack_along_edge_distribution(token_attacks):
    attack_counter_dict = {}
    for cur_token_attack in token_attacks:
        sorted_id_list = sorted([cur_token_attack[0], cur_token_attack[1]])
        id_string = f"{sorted_id_list[0]}, {sorted_id_list[1]}"
        add_to_counter_dict(attack_counter_dict, id_string, cur_token_attack[2])

    counter_dict_to_fill = {}
    for cur_attack_amount in attack_counter_dict.items():
        add_to_counter_dict(counter_dict_to_fill, cur_attack_amount[1])
    counter_dict_to_fill = dict(sorted(counter_dict_to_fill.items()))

    return counter_dict_to_fill

token_attack_along_edge_distribution_before_p1_counter_dict = get_token_attack_along_edge_distribution(last_step.token_attacks_p2)
token_attack_along_edge_distribution_after_p2_counter_dict = get_token_attack_along_edge_distribution(cur_step.token_attacks_p2)

