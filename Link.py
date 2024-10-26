
from LinkNode import *
from SimOptions import *

class Link:
    """
    The link between particles
    """

    node1: LinkNode = None
    node2: LinkNode = None

    age = None
    active_value = None
    new_link = None
    is_active = None

    vis_max_color_value = None


    def __init__(self, particle1, particle2):
        self.node1 = LinkNode()
        self.node2 = LinkNode()
        if particle1 is None or particle2 is None:
            print("WTF1124")
        self.node1.initialize(particle=particle1, link=self, other_node=self.node2)
        self.node2.initialize(particle=particle2, link=self, other_node=self.node1)
        self.age = 0
        self.active_value = 0
        self.is_new_link = True
        self.vis_max_color_value = 0
        self.is_active = False



    def check_inactivity(self, sim_options, all_links, data, dead_links):
        if not self.is_active:
            data.inactive_links_history[-1] += 1.0

            self.kill_link(sim_options=sim_options, all_links=all_links, dead_links=dead_links)


    def prepare(self, sim_options):
        """
        Prepares information for next iteration.
        :param sim_options: Options of Simulation
        """
        self.vis_max_color_value = self.active_value

        self.node1.prepare(sim_options=sim_options)
        self.node2.prepare(sim_options=sim_options)
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
            data.swap_percentage_history[-1] += 2.0

            self.node1.particle.behavior, self.node2.particle.behavior = self.node2.particle.behavior, \
                                                                         self.node1.particle.behavior


    def kill_link(self, sim_options, all_links, dead_links):
        """
        Kills the link and shifts all the links of the vanishing particle to the surviving particle.
        :param sim_options: Options of Simulation
        :param all_links: all_links for reference
        :param dead_links: dead links for reference
        """

        self.node1.disconnect()
        self.node2.disconnect()
        all_links.remove(self)
        dead_links.append(self)



