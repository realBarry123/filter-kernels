import matplotlib.pyplot as plt

def heatmap_2d(x, xlabel="x", ylabel="y", title=None, cmap="viridis", vmin=None, vmax=None, save_file=None):
    if x.dim() != 2:
        raise ValueError(f"plot_heatmap_2d expects input of shape (t, d), got {tuple(x.shape)}")

    # Transpose so imshow's x-axis corresponds to the 0th dimension of the input
    data = x.detach().cpu().numpy()  # shape -> (d, t)

    fig, ax = plt.subplots()
    im = ax.imshow(data, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("value")
    if save_file is not None:
        plt.savefig(save_file)
    plt.show()
    return ax
