import matplotlib.pyplot as plt
import matplotlib.colors as colors


def plot_heatmap_basic(data, title, x_ticks = [], is_show_plot=True):
    # This dictionary defines the colormap
    cdict = {'red': ((0.0, 0.0, 0.0),  # no red at 0
                     (0.5, 1.0, 1.0),  # all channels set to 1.0 at 0.5 to create white
                     (1.0, 0.8, 0.8)),  # set to 0.8 so its not too bright at 1

             'green': ((0.0, 0.8, 0.8),  # set to 0.8 so its not too bright at 0
                       (0.5, 1.0, 1.0),  # all channels set to 1.0 at 0.5 to create white
                       (1.0, 0.0, 0.0)),  # no green at 1

             'blue': ((0.0, 0.0, 0.0),  # no blue at 0
                      (0.5, 1.0, 1.0),  # all channels set to 1.0 at 0.5 to create white
                      (1.0, 0.0, 0.0))  # no blue at 1
             }

    # Create the colormap using the dictionary
    GnRd = colors.LinearSegmentedColormap('GnRd', cdict)

    # Make a figure and axes
    fig, ax = plt.subplots(1)

    # Plot the fake data
    p = ax.pcolormesh(data, cmap=GnRd, vmin=0, vmax=1)
    ax.invert_yaxis()

    ax.set_xticks(x_ticks)
    ax.set_yticks(x_ticks)

    # Make a colorbar
    fig.colorbar(p, ax=ax)

    fig.suptitle(title)

    fig.savefig(title + '.png', bbox_inches='tight')
    if is_show_plot:
        plt.show()
