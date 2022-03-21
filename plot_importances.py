import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json


STACKED = True
HORIZONTAL = True

results_file = "./CAVE/output/rocksdb/test.json"
output_image = "./CAVE/output/rocksdb/test.eps"

# Random forest importance values from HyperMapper. We can always find these in the log file. Take the latest iteration.
RF_data = [
        0.13446159744632505,
        0.1817525262705587,
        0.004084893380879819,
        0.007882267131421995,
        0.0023343928478808933,
        0.0017898764404960029,
        0.0011783854525174173,
        0.4638587006111535,
        0.09149861391671249,
        0.09322824184027148,
        0.0018488469147009707,
        0.014663264080834409,
        0.0014183936662473,
    ]
title = "test"

parameters = [
    "block_size",
    "cache_index_and_filter_blocks",
    "compaction_readahead_size",
    "compression_type",
    "level0_file_num_compaction_trigger",
    "level0_slowdown_writes_trigger",
    "level0_stop_writes_trigger",
    "max_background_compactions",
    "max_background_flushes",
    "max_bytes_for_level_multiplier",
    "max_write_buffer_number",
    "min_write_buffer_number_to_merge",
    "write_buffer_size",
    ]



def match_parameters(d, positive_normalization=False):
    new = d.copy()
    for x in d:

        # round numbers to 2 signifcant figures
        new[x] = round(new[x], 2)

        if x not in parameters:
            del new[x]

    for x in parameters:
        if x not in d:
            new[x] = 0.0

    weights = np.array(list(dict(sorted(new.items())).values()))
    # return weights.tolist()

    # In Ablation Analysis, the values returned are improvement percentages. 
    # If no parameter leads to an improvement, then these values may be negative.
    # We can choose to give them a value of zero.
    if positive_normalization:
        for i in range(len(weights)):
            if weights[i] < 0.0:
                weights[i] = 0.0
    weights /= weights.sum() # Normalize
    return weights.tolist()


#################### PLOT FEATURE IMPORTANCES ####################
with open(results_file) as f:
    data = json.load(f)

    AA = match_parameters(data["ablation"]["imp"], positive_normalization=True)
    fANOVA = match_parameters(data["fanova"]["imp"])
    LPI = match_parameters(data["lpi"]["imp"])


imps = [RF_data, LPI, AA, fANOVA]

threshold = 0.02
colors = ["firebrick", "forestgreen", "royalblue", "purple"]
hatches = ["\\\\", "||", "//", "OO"]
labels = ["RF", "LPI", "AA", "fANOVA"]


def plot_stacked_bar(
    data,
    series_labels,
    title=None,
    category_labels=None,
    show_values=False,
    value_format="{}",
    y_label=None,
    colors=None,
    hatches=None,
    grid=True,
    reverse=False,
):
    """Plots a stacked bar chart with the data and labels provided.

    Keyword arguments:
    data            -- 2-dimensional numpy array or nested list
                       containing data for each series in rows
    series_labels   -- list of series labels (these appear in
                       the legend)
    category_labels -- list of category labels (these appear
                       on the x-axis)
    show_values     -- If True then numeric value labels will
                       be shown on each bar
    value_format    -- Format string for numeric value labels
                       (default is "{}")
    y_label         -- Label for y-axis (str)
    colors          -- List of color labels
    grid            -- If True display grid
    reverse         -- If True reverse the order that the
                       series are displayed (left-to-right
                       or right-to-left)
    """

    ny = len(data[0])
    ind = list(range(ny))

    axes = []
    cum_size = np.zeros(ny)

    data = np.array(data)

    if reverse:
        data = np.flip(data, axis=1)
        category_labels = reversed(category_labels)

    for i, row_data in enumerate(data):
        color = colors[i] if colors is not None else None
        hatch = hatches[i] if hatches is not None else None
        if HORIZONTAL:
            axes.append(
                plt.barh(
                    ind,
                    row_data,
                    left=cum_size,
                    label=series_labels[i],
                    hatch=hatch,
                    color=color,
                )
            )
        else:
            axes.append(
                plt.bar(
                    ind,
                    row_data,
                    bottom=cum_size,
                    label=series_labels[i],
                    hatch=hatch,
                    color=color,
                )
            )
        cum_size += row_data

    if category_labels:
        if HORIZONTAL:
            plt.yticks(ind, category_labels, rotation=0)
        else:
            plt.xticks(ind, category_labels, rotation=90)

    if y_label:
        if HORIZONTAL:
            plt.xlabel(y_label)
        else:
            plt.ylabel(y_label)

    plt.legend()

    if title:
        plt.title(title)

    if grid:
        plt.grid()

    if show_values:
        for axis in axes:
            for bar in axis:
                w, h = bar.get_width(), bar.get_height()
                plt.text(
                    bar.get_x() + w / 2,
                    bar.get_y() + h / 2,
                    value_format.format(h),
                    ha="center",
                    va="center",
                )


plt.figure(figsize=(6, 4))
plot_stacked_bar(
    imps,
    labels,
    title,
    category_labels=parameters,
    show_values=False,
    value_format="{:.1f}",
    colors=colors,
    hatches=hatches,
    grid=False,
    y_label="Importance weight",
)

plt.savefig(output_image, bbox_inches="tight", format="eps") #  eps format is nice
