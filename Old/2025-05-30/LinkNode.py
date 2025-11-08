


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
    information = None
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

    def prepare(self, sim_options):
        """
        Prepares information for next iteration.
        :param sim_options: Options of Simulation.
        """
        self.is_shifting = False
        self.is_swapping = False
        self.is_reconnecting = False

    def reset_info(self):
        """
        Resets the info if node or other_node has reconnected and information no longer represents the last interaction.
        """

        self.reconnection_rating = 0.0
        self.is_shifting = False
        self.is_swapping = False
        self.is_reconnecting = False
