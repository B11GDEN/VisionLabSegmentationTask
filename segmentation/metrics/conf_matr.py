import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(
        conf_mat: np.ndarray,
        hide_spines: bool = False,
        hide_ticks: bool = False,
        figsize: tuple[int, int] | None = None,
        cmap=None,
        colorbar: bool = False,
        show_absolute: bool = True,
        show_normed: bool = False,
        norm_colormap: bool | None = None,
        class_names: list[str] | None = None,
        figure=None,
        axis=None,
        fontcolor_threshold: float = 0.5,
        show_in_percent: bool = True
):
    """
    Plot a confusion matrix via matplotlib.

    Args:
        conf_mat (np.ndarray): array-like, shape = [n_classes, n_classes].
        hide_spines (bool): Hides axis spines if True. Default  is `False`.
        hide_ticks (bool): Hides axis ticks if True. Default is `False`.
        figsize (tuple[int, int] | None): Height and width of the figure, Default is `None`.
        cmap: Matplotlib colormap, Default is `None`. Uses matplotlib.pyplot.cm.Blues if `None`
        colorbar (bool): Shows a colorbar if True. Default is `False`.
        show_absolute (bool): Shows absolute confusion matrix coefficients if `True`.
            At least one of  `show_absolute` or `show_normed` must be `True`. Default is `True`.
        show_normed (bool): Shows normed confusion matrix coefficients if True.
            The normed confusion matrix coefficients give the proportion of training examples per class that are
            assigned the correct label. At least one of  `show_absolute` or `show_normed` must be `True`.
        norm_colormap (bool | None): Matplotlib color normalization object to normalize the
            color scale, e.g., `matplotlib.colors.LogNorm()`. Default is `None`.
        class_names (list[str] | None): List of class names. If not `None`, ticks will be set to these values.
            Default is `None`.
        figure: `None` or Matplotlib figure. If `None` will create a new figure. Default is `None`.
        axis: `None` or Matplotlib figure axis. If `None` will create a new axis. Default is `None`.
        fontcolor_threshold (float): Sets a threshold for choosing black and white font colors
            for the cells. By default all values larger than 0.5 times the maximum cell value are converted to white,
            and everything equal or smaller than 0.5 times the maximum cell value are converted to black.
            Default to 0.5.
        show_in_percent (bool): show numbers in percent

    Returns:
        fig, ax: matplotlib.pyplot subplot objects. Figure and axis elements of the subplot.
    """
    if not (show_absolute or show_normed):
        raise AssertionError("Both show_absolute and show_normed are False")
    if class_names is not None and len(class_names) != len(conf_mat):
        raise AssertionError(
            "len(class_names) should be equal to number of" "classes in the dataset"
        )

    total_samples = conf_mat.sum(axis=1)[:, np.newaxis]
    if show_in_percent:
        normed_conf_mat = conf_mat.astype("float") / total_samples * 100
    else:
        normed_conf_mat = conf_mat.astype("float") / total_samples

    if figure is None and axis is None:
        fig, ax = plt.subplots(figsize=figsize)
    elif axis is None:
        fig = figure
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig, ax = figure, axis

    ax.grid(False)
    if cmap is None:
        cmap = plt.cm.Blues

    if figsize is None:
        figsize = (len(conf_mat) * 1.25, len(conf_mat) * 1.25)

    if show_normed:
        matshow = ax.matshow(normed_conf_mat, cmap=cmap, norm=norm_colormap)
    else:
        matshow = ax.matshow(conf_mat, cmap=cmap, norm=norm_colormap)

    if colorbar:
        fig.colorbar(matshow)

    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            cell_text = ""
            if show_absolute:
                num = conf_mat[i, j].astype(np.int64)
                cell_text += format(num, "d")
                if show_normed:
                    cell_text += "\n" + "("
                    cell_text += format(normed_conf_mat[i, j], ".2f") + ")"
            else:
                cell_text += format(normed_conf_mat[i, j], ".2f")

            if show_normed:
                ax.text(
                    x=j,
                    y=i,
                    s=cell_text,
                    va="center",
                    ha="center",
                    color=(
                        "white"
                        if normed_conf_mat[i, j] > 1 * fontcolor_threshold
                        else "black"
                    ),
                )
            else:
                ax.text(
                    x=j,
                    y=i,
                    s=cell_text,
                    va="center",
                    ha="center",
                    color="white"
                    if conf_mat[i, j] > np.max(conf_mat) * fontcolor_threshold
                    else "black",
                )
    if class_names is not None:
        tick_marks = np.arange(len(class_names))
        plt.xticks(
            tick_marks, class_names, rotation=45, ha="right", rotation_mode="anchor"
        )
        plt.yticks(tick_marks, class_names)

    if hide_spines:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    if hide_ticks:
        ax.axes.get_yaxis().set_ticks([])
        ax.axes.get_xaxis().set_ticks([])

    plt.xlabel("predicted label")
    plt.ylabel("true label")
    return fig, ax
