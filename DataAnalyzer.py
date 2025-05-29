import pickle
from Data import *
from SimOptions import *
import networkx as nx


# TODO Recreate current diagrams

# TODO seperately try to understand p1 and p2 maybe?

# TODO dimensionality

# TODO make analysis per node

# TODO track inactive links

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

    amount_token_before_p1: int = None
    amount_token_used_for_reproduction: int = None

    id_of_child: int = None
    id_of_parent: int = None

    amount_token_received_during_p1_through_redistribution: int = None
    bool_did_plant: bool = None
    amount_token_after_p1: int = None
    amount_token_used_for_defense: int = None
    dict_token_used_for_node_attacks: dict = None
    amount_token_used_for_attack: int = None
    dict_blotto_competition_here: dict = None
    amount_total_competition_token: int = None
    amount_token_that_winner_has_allocated: int = None
    bool_did_win_at_home: bool = None
    bool_did_win_elsewhere: bool = None
    ids_did_win_at_these_nodes = None
    amount_wins_somewhere = None
    amount_wins_elsewhere = None

    amount_token_after_p2: int = None

    bool_spawned_this_iteration_p1: bool = None
    bool_did_reproduce_this_iteration_p1: bool = None
    bool_died_because_all_tokens_used_for_reproduction_during_p1: bool = None
    bool_died_instantly_after_spawning_during_p1: bool = None
    bool_died_because_it_was_part_of_smaller_fragmented_network_p1: bool = None
    bool_died_during_p1: bool = None

    bool_died_because_of_competition_p2: bool = None
    bool_died_because_it_was_part_of_smaller_fragmented_network_p2: bool = None
    bool_died_during_p2: bool = None
    amount_token_received_during_p2_through_redistribution: int = None # TODO


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

    float_hubs_score_after_p1: float = None # TODO
    float_hubs_score_after_p2: float = None # TODO

    float_authority_score_after_p1: float = None # TODO
    float_authority_score_after_p2: float = None # TODO

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


closeness_centrality_after_p1 = nx.closeness_centrality(G_after_p1)
closeness_centrality_after_p2 = nx.closeness_centrality(G_after_p2)

betweenness_after_p1 = nx.betweenness_centrality(G_after_p1)
betweenness_after_p2 = nx.betweenness_centrality(G_after_p2)

eigenvector_after_p1 = nx.eigenvector_centrality(G_after_p1, max_iter=1000)
eigenvector_after_p2 = nx.eigenvector_centrality(G_after_p2, max_iter=1000)

clustering_after_p1 = nx.clustering(G_after_p1)
clustering_after_p2 = nx.clustering(G_after_p2)

k_core_after_p1 = nx.core_number(G_after_p1)
k_core_after_p2 = nx.core_number(G_after_p2)

local_efficiency_after_p1 = {n: nx.local_efficiency(G_after_p1.subgraph(G_after_p1.neighbors(n)).copy()) for n in G_after_p1.nodes()}
local_efficiency_after_p2 = {n: nx.local_efficiency(G_after_p2.subgraph(G_after_p2.neighbors(n)).copy()) for n in G_after_p2.nodes()}

eccentricity_after_p1 = nx.eccentricity(G_after_p1)
eccentricity_after_p2 = nx.eccentricity(G_after_p2)

avg_shortest_path_len_after_p1 = {n: sum(nx.single_source_shortest_path_length(G_after_p1, n).values()) / (len(G_after_p1) - 1) for n in G_after_p1.nodes()}
avg_shortest_path_len_after_p2 = {n: sum(nx.single_source_shortest_path_length(G_after_p2, n).values()) / (len(G_after_p2) - 1) for n in G_after_p2.nodes()}

assortativity_after_p1 = nx.degree_assortativity_coefficient(G_after_p1)
assortativity_after_p2 = nx.degree_assortativity_coefficient(G_after_p2)

max_r_after_p1 = max(dict(nx.eccentricity(G_after_p1)).values())
max_r_after_p2 = max(dict(nx.eccentricity(G_after_p2)).values())
def compute_fractal_dimensionality(G, max_r):
    fractal_dim = {n: [] for n in G.nodes()}
    for node in G.nodes():
        for r in range(max_r + 1):
            nodes_within_r = nx.single_source_shortest_path_length(G, node, cutoff=r)
            fractal_dim[node].append(len(nodes_within_r))
    return fractal_dim

fractal_dim_list_after_p1 = compute_fractal_dimensionality(G_after_p1, max_r_after_p1)
fractal_dim_list_after_p2 = compute_fractal_dimensionality(G_after_p2, max_r_after_p2)

from networkx.algorithms import community, hits, communicability
from scipy.stats import entropy

communities_after_p1 = list(community.greedy_modularity_communities(G_after_p1))
community_membership_after_p1 = {}
for i, comm in enumerate(communities_after_p1):
    for node in comm:
        community_membership_after_p1[node] = i
communities_after_p2 = list(community.greedy_modularity_communities(G_after_p2))
community_membership_after_p2 = {}
for i, comm in enumerate(communities_after_p2):
    for node in comm:
        community_membership_after_p2[node] = i

hubs_after_p1, authorities_after_p1 = hits(G_after_p1, max_iter=1000)
hubs_after_p2, authorities_after_p2 = hits(G_after_p2, max_iter=1000)

def shannon_entropy_per_node(G):
    A = nx.to_numpy_array(G)
    D_inv = np.linalg.inv(np.diag(A.sum(axis=1)))
    P = D_inv @ A  # Transition matrix
    entropies = {}
    for i, node in enumerate(G.nodes()):
        p_i = P[i]
        entropies[node] = entropy(p_i[p_i > 0], base=2)
    return entropies

shannon_ent_after_p1 = shannon_entropy_per_node(G_after_p1)
shannon_ent_after_p2 = shannon_entropy_per_node(G_after_p2)



def safe_setattr(node_obj, attr, value):
    if node_obj is not None:
        setattr(node_obj, attr, value)

# Replace dict_name and attribute_name accordingly
def assign_from_dict(data_dict, attr_name):
    for node_id, value in data_dict.items():
        node = all_nodes.get(node_id)
        safe_setattr(node, attr_name, value)

# Assign from all dictionaries (after_p1)
assign_from_dict(closeness_centrality_after_p1, 'float_closeness_centrality_after_p1')
assign_from_dict(betweenness_after_p1, 'float_betweenness_centrality_after_p1')
assign_from_dict(eigenvector_after_p1, 'float_eigenvector_centrality_after_p1')
assign_from_dict(clustering_after_p1, 'float_clustering_coefficient_after_p1')
assign_from_dict(k_core_after_p1, 'float_k_core_number_after_p1')
assign_from_dict(local_efficiency_after_p1, 'float_local_efficiency_after_p1')
assign_from_dict(eccentricity_after_p1, 'float_eccentricity_after_p1')
assign_from_dict(avg_shortest_path_len_after_p1, 'float_average_shortest_path_length_after_p1')
assign_from_dict(shannon_ent_after_p1, 'float_shannon_entropy_after_p1')
assign_from_dict(hubs_after_p1, 'float_hubs_score_after_p1')
assign_from_dict(authorities_after_p1, 'float_authority_score_after_p1')
assign_from_dict(community_membership_after_p1, 'float_community_membership_after_p1')

# Do the same for after_p2
assign_from_dict(closeness_centrality_after_p2, 'float_closeness_centrality_after_p2')
assign_from_dict(betweenness_after_p2, 'float_betweenness_centrality_after_p2')
assign_from_dict(eigenvector_after_p2, 'float_eigenvector_centrality_after_p2')
assign_from_dict(clustering_after_p2, 'float_clustering_coefficient_after_p2')
assign_from_dict(k_core_after_p2, 'float_k_core_number_after_p2')
assign_from_dict(local_efficiency_after_p2, 'float_local_efficiency_after_p2')
assign_from_dict(eccentricity_after_p2, 'float_eccentricity_after_p2')
assign_from_dict(avg_shortest_path_len_after_p2, 'float_average_shortest_path_length_after_p2')
assign_from_dict(shannon_ent_after_p2, 'float_shannon_entropy_after_p2')
assign_from_dict(hubs_after_p2, 'float_hubs_score_after_p2')
assign_from_dict(authorities_after_p2, 'float_authority_score_after_p2')
assign_from_dict(community_membership_after_p2, 'float_community_membership_after_p2')


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
    child_node.id_of_parent = cur_repro[0]
    child_node.bool_died_instantly_after_spawning_during_p1 = False
    child_node.bool_died_because_it_was_part_of_smaller_fragmented_network_p1 = False
for cur_node_id in ids_nodes_before_p1:
    cur_node = all_nodes[cur_node_id]
    cur_node.amount_token_used_for_reproduction = cur_node.amount_token_used_for_reproduction or 0
    cur_node.bool_did_reproduce_this_iteration_p1 = cur_node.bool_did_reproduce_this_iteration_p1 or False
    cur_node.bool_died_because_all_tokens_used_for_reproduction_during_p1 = cur_node.bool_died_because_all_tokens_used_for_reproduction_during_p1 or False
for cur_node_id in list(set(ids_nodes_after_p1 + ids_nodes_after_p2)):
    cur_node = all_nodes[cur_node_id]
    cur_node.bool_spawned_this_iteration_p1 = cur_node.bool_spawned_this_iteration_p1 or False
for cur_node_id in ids_newly_reproduced_nodes_that_instantly_died:
    cur_node = all_nodes[cur_node_id]
    cur_node.bool_died_instantly_after_spawning_during_p1 = True
    cur_node.bool_died_because_it_was_part_of_smaller_fragmented_network_p1 = True

for cur_node_id in cur_step.dead_nodes_p1:
    all_nodes[cur_node_id].bool_died_during_p1 = True
for cur_node_id in list(set(ids_nodes_before_p1 + ids_newly_reproduced_nodes)):
    cur_node = all_nodes[cur_node_id]
    cur_node.bool_died_during_p1 = cur_node.bool_died_during_p1 or False

for cur_node_id in cur_step.dead_nodes_p2:
    all_nodes[cur_node_id].bool_died_during_p2 = True
for cur_node_id in ids_nodes_after_p1:
    cur_node = all_nodes[cur_node_id]
    cur_node.bool_died_during_p2 = cur_node.bool_died_during_p2 or False


for cur_node_id in cur_step.planted_children_p1:
    all_nodes[cur_node_id].bool_did_plant = True
for cur_node_id in ids_nodes_before_p1:
    cur_node = all_nodes[cur_node_id]
    cur_node.bool_did_plant = cur_node.bool_did_plant or False

for cur_node_id in ids_nodes_after_p1:
    cur_node = all_nodes[cur_node_id]
    cur_node.dict_blotto_competition_here = {}
    cur_node.dict_token_used_for_node_attacks = {}
    cur_node.amount_token_used_for_attack = 0
    cur_node.amount_token_used_for_defense = 0
    cur_node.amount_total_competition_token = 0
    cur_node.bool_died_because_it_was_part_of_smaller_fragmented_network_p2 = False
for cur_token_attack in cur_step.token_attacks_p2:

    attacker_node_id = cur_token_attack[0]
    attacked_node_id = cur_token_attack[1]
    amount_token_attack = cur_token_attack[2]

    attacker_node = all_nodes[attacker_node_id]
    attacked_node = all_nodes[attacked_node_id]

    attacked_node.dict_blotto_competition_here[attacker_node_id] = amount_token_attack
    attacker_node.dict_token_used_for_node_attacks[attacked_node_id] = amount_token_attack
    if attacker_node_id == attacked_node_id:
        attacker_node.amount_token_used_for_defense += amount_token_attack
    else:
        attacker_node.amount_token_used_for_attack += amount_token_attack
    attacked_node.amount_total_competition_token += amount_token_attack
for cur_node_id in ids_nodes_after_p1:
    cur_node = all_nodes[cur_node_id]
    # TODO this is only true if WINNER_TOKEN_GO_TO_COMPETITION_PLACE = True
    if cur_node.amount_total_competition_token == 0:
        cur_node.bool_died_because_of_competition_p2 = True
    else:
        cur_node.bool_died_because_of_competition_p2 = False
        if cur_node_id in ids_nodes_death_during_p2:
            cur_node.bool_died_because_it_was_part_of_smaller_fragmented_network_p2 = True




for cur_node_id in ids_nodes_after_p1:
    cur_node = all_nodes[cur_node_id]
    cur_node.bool_did_win_elsewhere = False
    cur_node.ids_did_win_at_these_nodes = []
    cur_node.amount_wins_somewhere = 0
    cur_node.amount_wins_elsewhere = 0
for cur_game_winner in cur_step.game_winner_p2:
    competition_node_id = cur_game_winner[0]
    winner_node_id = cur_game_winner[1]

    competition_node = all_nodes[competition_node_id]
    winner_node = all_nodes[winner_node_id]

    competition_node.amount_token_that_winner_has_allocated = competition_node.dict_blotto_competition_here[winner_node_id]
    if competition_node_id == winner_node_id:
        competition_node.bool_did_win_at_home = True
    else:
        competition_node.bool_did_win_at_home = False
        winner_node.bool_did_win_elsewhere = True
        winner_node.amount_wins_elsewhere += 1
    winner_node.amount_wins_somewhere += 1
    winner_node.ids_did_win_at_these_nodes.append(competition_node_id)

for cur_node_id in ids_nodes_after_p1:
    cur_node = all_nodes[cur_node_id]
    if cur_node.bool_spawned_this_iteration_p1:
        reproduction_tokens = all_nodes[cur_node.id_of_parent].amount_token_used_for_reproduction
        cur_node.amount_token_received_during_p1_through_redistribution = cur_node.amount_token_after_p1 - reproduction_tokens
    else:
        reproduction_tokens = cur_node.amount_token_used_for_reproduction
        token_amount_before = cur_node.amount_token_before_p1
        token_amount_after = cur_node.amount_token_after_p1
        cur_node.amount_token_received_during_p1_through_redistribution = token_amount_after - token_amount_before + reproduction_tokens

for cur_node_id in ids_nodes_after_p2:
    # TODO this is only true if WINNER_TOKEN_GO_TO_COMPETITION_PLACE = True
    cur_node = all_nodes[cur_node_id]
    token_after = cur_node.amount_token_after_p2
    competition_token = cur_node.amount_total_competition_token
    cur_node.amount_token_received_during_p2_through_redistribution = token_after - competition_token

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

