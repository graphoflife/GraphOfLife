import pickle
from Data import *
from SimOptions import *
import networkx as nx


# TODO Recreate current diagrams

# TODO make analysis per node

# TODO find reason for death in p2... part of smaller network or no tokens there after game?

# TODO where did they move their walkers? where are they now?

# TODO better understand shifts and reconnection

# try to correlate any kind of data, for example link amounts of walker positions and so on
# extract as much information as possible

# think about minimal amount of information needed to store whats happening such that it can
# still be reconstructed what happened, withouth the neural networks. How does this scale as the simulation gets bigger.

# dimensionality, centrality and all those fucking values for each node

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


i = 942
file_path = fr"C:\Users\stefa\Desktop\Iteration XXX.pkl"


with open(file_path.replace("XXX",str(i-1)), 'rb') as file:
    last_step = pickle.load(file)

with open(file_path.replace("XXX",str(i)), 'rb') as file:
    cur_step = pickle.load(file)

# Nodes
ids_nodes_before_p1 = last_step.nodes_p2
ids_nodes_after_p1 = cur_step.nodes_p1
ids_nodes_after_p2 = cur_step.nodes_p2


amount_node_before_p1 = len(last_step.nodes_p2)
amount_node_after_p1 = len(cur_step.nodes_p1)
amount_node_after_p2 = len(cur_step.nodes_p2)

# Deaths
ids_nodes_death_during_p1 = cur_step.dead_nodes_p1
ids_nodes_death_during_p2 = cur_step.dead_nodes_p2

amount_node_deaths_after_p1 = len(cur_step.dead_nodes_p1)
amount_node_deaths_after_p2 = len(cur_step.dead_nodes_p2)

# Edges
ids_edges_before_p1 = last_step.edges_p2
ids_edges_after_p1 = cur_step.edges_p1
ids_edges_after_p2 = cur_step.edges_p2

amount_edge_before_p1 = len(last_step.edges_p2)
amount_edge_after_p1 =  len(cur_step.edges_p1)
amount_edge_after_p2 =  len(cur_step.edges_p2)

# Reproduction
ids_parent_nodes_that_reproduced = [cur_reproduction[0] for cur_reproduction in cur_step.reproduced_nodes_p1]
ids_newly_reproduced_nodes = [cur_reproduction[1] for cur_reproduction in cur_step.reproduced_nodes_p1]

amount_reproduced_nodes_p1 = len(cur_step.reproduced_nodes_p1)
percentage_of_nodes_that_reproduced = amount_reproduced_nodes_p1/amount_node_before_p1

# They died because they allocated all their tokens to the child -> ripping apart network and killing more
ids_original_nodes_that_died_after_p1 = list(set(cur_step.dead_nodes_p1) & set(last_step.nodes_p2))
amount_original_node_that_died_after_p1 = len(ids_original_nodes_that_died_after_p1)
percentage_of_original_nodes_that_died_after_p1 = amount_original_node_that_died_after_p1/amount_node_before_p1

ids_newly_reproduced_nodes_that_instantly_died = list(set(cur_step.dead_nodes_p1) & set(ids_newly_reproduced_nodes))
# they died because they were part of the smaller network
amount_newly_reproduced_node_that_instantly_died = len(ids_newly_reproduced_nodes_that_instantly_died)
percentage_of_newly_spawned_nodes_that_instantly_died_p1 = amount_newly_reproduced_node_that_instantly_died/amount_reproduced_nodes_p1

ids_unique_nodes = list(set(last_step.nodes_p2 + cur_step.nodes_p1 + cur_step.nodes_p2 + ids_newly_reproduced_nodes_that_instantly_died))


amount_total_token_before_p1 = sum([cur_node_token[1] for cur_node_token in last_step.node_tokens_p2])
amount_total_token_after_p1 = sum([cur_node_token[1] for cur_node_token in cur_step.node_tokens_p1])
amount_total_token_after_p2 = sum([cur_node_token[1] for cur_node_token in cur_step.node_tokens_p2])

amount_token_used_for_reproduction = sum([cur_reproduced[2] for cur_reproduced in cur_step.reproduced_nodes_p1])
percentage_token_used_for_reproduction = amount_token_used_for_reproduction/amount_total_token_before_p1


ids_nodes_that_did_not_reproduce = [node_id for node_id in last_step.nodes_p2 if node_id not in ids_parent_nodes_that_reproduced]

amount_token_that_get_redistributed_because_of_instant_child_death = sum([cur_repro[2] for cur_repro in cur_step.reproduced_nodes_p1 if cur_repro[1] in ids_newly_reproduced_nodes_that_instantly_died])

# Walkers
ids_nodes_with_walkers_at_self_before_p1 = [cur_node_walker[0] for cur_node_walker in last_step.node_walkers_p2 if cur_node_walker[0] == cur_node_walker[1]]
ids_nodes_with_walkers_at_self_after_p1 = [cur_node_walker[0] for cur_node_walker in cur_step.node_walkers_p1 if cur_node_walker[0] == cur_node_walker[1]]
ids_nodes_with_walkers_at_self_after_p2 = [cur_node_walker[0] for cur_node_walker in cur_step.node_walkers_p2 if cur_node_walker[0] == cur_node_walker[1]]

amount_nodes_with_walkers_at_self_before_p1 = len(ids_nodes_with_walkers_at_self_before_p1)
amount_nodes_with_walkers_at_self_after_p1 = len(ids_nodes_with_walkers_at_self_after_p1 )
amount_nodes_with_walkers_at_self_after_p2 = len(ids_nodes_with_walkers_at_self_after_p2 )

percentage_nodes_with_walkers_at_self_before_p1 = amount_nodes_with_walkers_at_self_before_p1 / amount_node_before_p1
percentage_nodes_with_walkers_at_self_after_p1 = amount_nodes_with_walkers_at_self_after_p1 / amount_node_after_p1
percentage_nodes_with_walkers_at_self_after_p2 = amount_nodes_with_walkers_at_self_after_p2 / amount_node_after_p2

amount_total_token_received_through_redistribution_during_p1 = amount_total_token_after_p1 - amount_total_token_before_p1
amount_total_token_received_through_redistribution_during_p2 = amount_total_token_after_p2 - amount_total_token_after_p1


def create_graph(nodes_px, edges_px):
    G = nx.Graph()
    G.add_nodes_from(nodes_px)
    for cur_edge in edges_px:
        G.add_edge(cur_edge[1], cur_edge[2])
    return G


G_before_p1 = create_graph(last_step.nodes_p2, last_step.edges_p2)
G_after_p1 = create_graph(cur_step.nodes_p1, cur_step.edges_p1)
G_after_p2 = create_graph(cur_step.nodes_p2, cur_step.edges_p2)


class Node:

    id_node: int = None

    bool_could_potentially_make_link_during_p1: bool = None # TODO
    bool_did_actually_make_link_during_p1: bool = None # TODO
    id_did_actually_make_link_with_node_during_p1: int = None # TODO

    ids_neighboring_nodes_before_p1 = None
    ids_neighboring_nodes_after_p1 = None
    ids_neighboring_nodes_after_p2 = None

    amount_neighbors_before_p1: int = None
    amount_neighbors_after_p1: int = None
    amount_neighbors_after_p2: int = None

    id_walker_before_p1: int = None
    id_walker_after_p1: int = None
    id_walker_after_p2: int = None

    bool_walker_at_home_before_p1: bool = None
    bool_walker_at_home_after_p1: bool = None
    bool_walker_at_home_after_p2: bool = None

    id_behavior_before_p1: int = None
    id_behavior_after_p1: int = None
    id_behavior_after_p2: int = None

    percentage_of_all_nodes_with_same_ancient_parent: float = None # TODO

    int_age_before_p1: int = None
    int_age_after_p2: int = None

    int_behavior_age_before_p1: int = None # TODO
    int_behavior_age_after_p2: int = None  # TODO

    amount_token_before_p1: int = None # TODO
    amount_token_used_for_reproduction: int = None # TODO
    id_of_child: int = None
    amount_token_received_during_p1_through_redistribution: int = None # TODO
    bool_did_plant: bool = None # TODO
    amount_token_after_p1: int = None # TODO
    amount_token_used_for_defense: int = None # TODO
    dict_token_used_for_node_attacks: dict = None # TODO
    amount_token_used_for_attack: int = None # TODO
    dict_blotto_competition_here: dict = None # TODO
    amount_total_competition_token: int = None # TODO
    amount_token_that_winner_has_allocated: int = None # TODO
    bool_did_win_at_home: bool = None # TODO
    bool_did_win_elsewhere: bool = None # TODO
    ids_did_win_at_these_nodes = None # TODO
    amount_wins_somewhere = None # TODO
    amount_wins_elsewhere = None # TODO

    amount_token_after_p2: int = None # TODO

    bool_spawned_this_iteration_p1: bool = None # TODO
    bool_did_reproduce_this_iteration_p1: bool = None # TODO
    bool_died_because_all_tokens_used_for_reproduction_during_p1: bool = None # TODO
    bool_died_instantly_after_spawning_during_p1: bool = None # TODO
    bool_died_because_it_was_part_of_smaller_fragmented_network_p1: bool = None # TODO
    bool_died_during_p1: bool = None # TODO

    bool_died_because_of_competition_p2: bool = None # TODO
    bool_died_because_it_was_part_of_smaller_fragmented_network_p2: bool = None # TODO
    bool_died_during_p2: bool = None # TODO


    float_dimensionality_after_p1: float = None # TODO
    float_dimensionality_after_p2: float = None # TODO

    float_closeness_centrality_after_p1: float = None # TODO
    float_closeness_centrality_after_p2: float = None # TODO

    float_betweenness_centrality_after_p1: float = None # TODO
    float_betweenness_centrality_after_p2: float = None # TODO

    float_eigenvector_centrality_after_p1: float = None # TODO
    float_eigenvector_centrality_after_p2: float = None # TODO

    float_clustering_coefficient_after_p1: float = None # TODO
    float_clustering_coefficient_after_p2: float = None # TODO

    float_k_core_number_after_p1: float = None # TODO
    float_k_core_number_after_p2: float = None # TODO

    float_local_efficiency_after_p1: float = None # TODO
    float_local_efficiency_after_p2: float = None # TODO

    float_eccentricity_after_p1: float = None # TODO
    float_eccentricity_after_p2: float = None # TODO

    float_average_shortest_path_length_after_p1: float = None # TODO
    float_average_shortest_path_length_after_p2: float = None # TODO

    float_assortativity_after_p1: float = None # TODO
    float_assortativity_after_p2: float = None # TODO

    float_community_membership_after_p1: float = None # TODO
    float_community_membership_after_p2: float = None # TODO

    float_hubs_and_authorities_after_p1: float = None # TODO
    float_hubs_and_authorities_after_p2: float = None # TODO

    float_ollivier_ricci_curvature_after_p1: float = None # TODO
    float_ollivier_ricci_curvature_after_p2: float = None # TODO

    float_forman_ricci_curvature_after_p1: float = None # TODO
    float_forman_ricci_curvature_after_p2: float = None # TODO

    floats_fractal_dimensionality_curve_after_p1 = None # TODO
    floats_fractal_dimensionality_curve_after_p2 = None # TODO

    float_random_walk_return_probability_after_p1: float = None # TODO
    float_random_walk_return_probability_after_p2: float = None # TODO

    float_shannon_entropy_after_p1: float = None # TODO
    float_shannon_entropy_after_p2: float = None # TODO

    float_node_communicability_after_p1: float = None # TODO
    float_node_communicability_after_p2: float = None # TODO


all_nodes = {}

for cur_node_id in ids_unique_nodes:

    cur_node = Node()
    all_nodes[cur_node_id] = cur_node

    cur_node.id_node = cur_node_id

    if cur_node_id in ids_nodes_before_p1:
        cur_node.ids_neighboring_nodes_before_p1 = list(G_before_p1.neighbors(cur_node_id))
        cur_node.amount_neighbors_before_p1 = G_before_p1.degree(cur_node_id)

    if cur_node_id in ids_nodes_after_p1:
        cur_node.ids_neighboring_nodes_after_p1 = list(G_after_p1.neighbors(cur_node_id))
        cur_node.amount_neighbors_after_p1 = G_after_p1.degree(cur_node_id)

    if cur_node_id in ids_nodes_after_p2:
        cur_node.ids_neighboring_nodes_after_p2 = list(G_after_p2.neighbors(cur_node_id))
        cur_node.amount_neighbors_after_p2 = G_after_p2.degree(cur_node_id)



for cur_node_walker in last_step.node_walkers_p2:
    all_nodes[cur_node_walker[0]].id_walker_before_p1 = cur_node_walker[1]
    all_nodes[cur_node_walker[0]].bool_walker_at_home_before_p1 = cur_node_walker[0] == cur_node_walker[1]
for cur_node_walker in cur_step.node_walkers_p1:
    all_nodes[cur_node_walker[0]].id_walker_after_p1 = cur_node_walker[1]
    all_nodes[cur_node_walker[0]].bool_walker_at_home_after_p1 = cur_node_walker[0] == cur_node_walker[1]
for cur_node_walker in cur_step.node_walkers_p2:
    all_nodes[cur_node_walker[0]].id_walker_after_p2 = cur_node_walker[1]
    all_nodes[cur_node_walker[0]].bool_walker_at_home_after_p2 = cur_node_walker[0] == cur_node_walker[1]

for cur_behavior in last_step.behaviors_p2:
    all_nodes[cur_behavior[0]].id_behavior_before_p1 = cur_behavior[1]
for cur_behavior in cur_step.behaviors_p1:
    all_nodes[cur_behavior[0]].id_behavior_after_p1 = cur_behavior[1]
for cur_behavior in cur_step.behaviors_p2:
    all_nodes[cur_behavior[0]].id_behavior_after_p2 = cur_behavior[1]

for cur_age in last_step.node_ages_p2:
    all_nodes[cur_age[0]].int_age_before_p1 = cur_age[1]
for cur_age in cur_step.node_ages_p2:
    all_nodes[cur_age[0]].int_age_after_p2 = cur_age[1]

for cur_node_token in last_step.node_tokens_p2:
    all_nodes[cur_node_token[0]].amount_token_before_p1 = cur_node_token[1]
for cur_node_token in cur_step.node_tokens_p1:
    all_nodes[cur_node_token[0]].amount_token_after_p1 = cur_node_token[1]
for cur_node_token in cur_step.node_tokens_p2:
    all_nodes[cur_node_token[0]].amount_token_after_p2 = cur_node_token[1]

for cur_repro in cur_step.reproduced_nodes_p1:
    parent_node = all_nodes[cur_repro[0]]
    parent_node.amount_token_used_for_reproduction = cur_repro[2]
    parent_node.id_of_child = cur_repro[1]
    parent_node.bool_did_reproduce_this_iteration_p1 = True
    if parent_node.amount_token_used_for_reproduction == parent_node.amount_token_before_p1:
        parent_node.bool_died_because_all_tokens_used_for_reproduction_during_p1 = True
    child_node = all_nodes[cur_repro[1]]
    child_node.bool_spawned_this_iteration_p1 = True
for cur_node_id in ids_nodes_before_p1:
    cur_node = all_nodes[cur_node_id]
    cur_node.amount_token_used_for_reproduction = amount_token_used_for_reproduction or 0
    cur_node.bool_did_reproduce_this_iteration_p1 = cur_node.bool_did_reproduce_this_iteration_p1 or False
    cur_node.bool_died_because_all_tokens_used_for_reproduction_during_p1 = cur_node.bool_died_because_all_tokens_used_for_reproduction_during_p1 or False

for cur_node_id in list(set()):
    cur_node = all_nodes[cur_node_id]
    cur_node.bool_spaw< ned_this_iteration_p1 = cur_node.bool_spawned_this_iteration_p1 or False



for cur_node_id in ids_unique_nodes:
    all_nodes[cur_node_id].bool_did_plant = cur_node_id in cur_step.planted_children_p1

amount_token_before_p1: int = None
amount_token_used_for_reproduction: int = None
amount_token_received_during_p1_through_redistribution: int = None # TODO
bool_did_plant: bool = None
amount_token_after_p1: int = None
amount_token_used_for_defense: int = None # TODO
dict_token_used_for_node_attacks: dict = None # TODO
amount_token_used_for_attack: int = None # TODO
dict_blotto_competition_here: dict = None # TODO
amount_total_competition_token: int = None # TODO
amount_token_that_winner_has_allocated: int = None # TODO
bool_did_win_at_home: bool = None # TODO
bool_did_win_elsewhere: bool = None # TODO
ids_did_win_at_these_nodes = None # TODO
amount_wins_somewhere = None # TODO
amount_wins_elsewhere = None # TODO

amount_token_after_p2: int = None


bool_spawned_this_iteration_p1: bool = None
bool_did_reproduce_this_iteration_p1: bool = None
bool_died_because_all_tokens_used_for_reproduction_during_p1: bool = None
bool_died_instantly_after_spawning_during_p1: bool = None # TODO
bool_died_because_it_was_part_of_smaller_fragmented_network_p1: bool = None # TODO
bool_died_during_p1: bool = None # TODO

bool_died_because_of_competition_p2: bool = None # TODO
bool_died_because_it_was_part_of_smaller_fragmented_network_p2: bool = None # TODO
bool_died_during_p2: bool = None # TODO



def get_distribution_amount_token_used_for_reproduction_(reproduced_nodes_p1):
    counter_dict_to_fill = {}
    for cur_reproduced_node in reproduced_nodes_p1:
        token = cur_reproduced_node[2]
        add_to_counter_dict(counter_dict_to_fill, token)
    counter_dict_to_fill = dict(sorted(counter_dict_to_fill.items()))
    return counter_dict_to_fill


distribution_amount_token_used_for_reproduction_ = get_distribution_amount_token_used_for_reproduction_(cur_step.reproduced_nodes_p1)

#TODO info of how much each node attacked others and defended itself

#sanity check 1
if (amount_node_before_p1 + amount_reproduced_nodes_p1 - amount_node_deaths_after_p1 == amount_node_after_p1) == False:
    print("Sanity Check  1 failed, this should not be true, find mistake")
#sanity check 2
if (amount_node_after_p1 - amount_node_deaths_after_p2 == amount_node_after_p2) == False:
    print("Sanity Check  2 failed, this should not be true, find mistake")
#sanity check 3
if (amount_newly_reproduced_node_that_instantly_died + amount_original_node_that_died_after_p1 == amount_node_deaths_after_p1) == False:
    print("Sanity Check  3 failed, this should not be true, find mistake")



def get_token_distribution(node_tokens):
    counter_dict_to_fill = {}
    for cur_node_token in node_tokens:
        token = cur_node_token[1]
        add_to_counter_dict(counter_dict_to_fill, token)
    counter_dict_to_fill = dict(sorted(counter_dict_to_fill.items()))

    return counter_dict_to_fill

distribution_token_before_p1_counter_dict = get_token_distribution(last_step.node_tokens_p2)
distribution_token_after_p1_counter_dict =  get_token_distribution(cur_step.node_tokens_p1)
distribution_token_after_p2_counter_dict =  get_token_distribution(cur_step.node_tokens_p2)

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

distribution_walker_concentration_before_p1_counter_dict = get_walker_concentration_distribution(last_step.node_walkers_p2)
distribution_walker_concentration_after_p1_counter_dict = get_walker_concentration_distribution(cur_step.node_walkers_p1)
distribution_walker_concentration_after_p2_counter_dict = get_walker_concentration_distribution(cur_step.node_walkers_p2)


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

distribution_link_amount_before_p1_counter_dict = get_link_amount_distribution(last_step.edges_p2)
distribution_link_amount_after_p1_counter_dict = get_link_amount_distribution(cur_step.edges_p1)
distribution_link_amount_after_p2_counter_dict = get_link_amount_distribution(cur_step.edges_p2)


def get_node_age_distribution(node_ages):
    counter_dict_to_fill = {}
    for cur_node_age in node_ages:
        add_to_counter_dict(counter_dict_to_fill, cur_node_age[1])
    counter_dict_to_fill = dict(sorted(counter_dict_to_fill.items()))
    return counter_dict_to_fill

distribution_node_age_before_p1_counter_dict = get_node_age_distribution(last_step.node_ages_p2)
distribution_node_age_after_p2_counter_dict = get_node_age_distribution(cur_step.node_ages_p2)

def get_behavior_age_distribution(behavior_ages):
    counter_dict_to_fill = {}
    for cur_node_age in behavior_ages:
        add_to_counter_dict(counter_dict_to_fill, cur_node_age[1])
    counter_dict_to_fill = dict(sorted(counter_dict_to_fill.items()))
    return counter_dict_to_fill

distribution_behavior_age_before_p1_counter_dict = get_node_age_distribution(last_step.behavior_ages_p2)
distribution_behavior_age_after_p2_counter_dict = get_node_age_distribution(cur_step.behavior_ages_p2)

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

distribution_token_attack_along_edge_before_p1_counter_dict = get_token_attack_along_edge_distribution(last_step.token_attacks_p2)
distribution_token_attack_along_edge_after_p2_counter_dict = get_token_attack_along_edge_distribution(cur_step.token_attacks_p2)

