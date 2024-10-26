import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt

black_color = [0, 0, 0, 1]

class VisOptions:

    backgroundcolor = (1, 1, 1, 1)
    backgroundcolor2 = (1, 1, 1, 1)

    cmap = mcolors.LinearSegmentedColormap.from_list("black_colormap", [black_color, black_color], N=256)
    cmap_edge = mcolors.LinearSegmentedColormap.from_list("black_colormap", [black_color, black_color], N=256)
    max_size_edge = 7.0
    min_size_edge = 3.5
    max_size_node = 1200
    min_size_node = 600
    mutate_rgb = 0.05

    def randomize(self):
        self.backgroundcolor = plt.cm.binary(np.random.uniform(0.0, 1.0))
        self.backgroundcolor2 = plt.cm.binary(np.random.uniform(0.0, 1.0))
        self.cmap = np.random.choice([plt.cm.binary, plt.cm.gist_rainbow, plt.cm.cool,
                                      plt.cm.summer, plt.cm.autumn, plt.cm.Wistia, plt.cm.winter, plt.cm.hot,
                                      plt.cm.Spectral, plt.cm.RdYlGn, plt.cm.rainbow, plt.cm.RdYlBu,
                                      plt.cm.viridis, plt.cm.plasma, plt.cm.coolwarm, plt.cm.twilight])
        self.cmap_edge = np.random.choice([plt.cm.binary, plt.cm.gist_rainbow, plt.cm.cool,
                                      plt.cm.summer, plt.cm.autumn, plt.cm.Wistia, plt.cm.winter, plt.cm.hot,
                                      plt.cm.Spectral, plt.cm.RdYlGn, plt.cm.rainbow, plt.cm.RdYlBu,
                                      plt.cm.viridis, plt.cm.plasma, plt.cm.coolwarm, plt.cm.twilight])

        self.max_size_edge = np.random.uniform(2, 5.5)
        self.min_size_edge = np.random.uniform(0, 0.5)

        self.max_size_node = np.random.uniform(30, 120)
        self.min_size_node = self.max_size_node*self.min_size_edge/self.max_size_edge
