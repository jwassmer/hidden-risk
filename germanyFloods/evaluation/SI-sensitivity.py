# %%
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from germanyFloods import readFiles as rf

from src import Plotting as pl

from tqdm import tqdm

import matplotlib.lines as mlines

all_catchment_gdf = gpd.read_file(
    "data_LFS/basin-polygons/NUTS_hydro_divisions_1010.geojson"
)
all_catchment_gdf.to_crs(epsg=4326, inplace=True)


# %%


gammas = [0, 0.02, 0.04, 0.06, 0.08, 0.1]

catchment = "all"


if catchment != "all":
    catchment_gdf = all_catchment_gdf[all_catchment_gdf["name"] == catchment]
else:
    catchment_gdf = all_catchment_gdf

affected_threshold = 1e-2

gdf_list = []

# mean_dist_list = []
for gamma in gammas:
    hospital_gdf = rf.read_hospital_catchement3(gamma=gamma, catchment=catchment)
    unique_hospitals = hospital_gdf.drop_duplicates(subset=["node"], keep="last")
    print("There are ", unique_hospitals.shape[0], " unique hospitals")
    # unique_hospitals = hospital_gdf.drop_duplicates(subset=["node"], keep="last")
    affected = hospital_gdf[
        hospital_gdf["population_percentual_diff"] > affected_threshold
    ]
    unique_affected = affected.drop_duplicates(subset=["node"], keep="last")
    # mean_diff = affected["population_percentual_diff"].mean()
    gdf_list.append(hospital_gdf)
    mean_dist = affected["min_dist"].mean()

    weighted_dist = np.average(
        affected["min_dist"], weights=affected["population_percentual_diff"]
    )
    # mean_dist_list.append(weighted_dist)

    print(f"Gamma: {gamma}")
    print(f"Unique affected: {unique_affected.shape[0]}")
    print(f"Total affected: {affected.shape[0]}")
    print(f"Mean distance: {mean_dist}")
    print(f"Weighted distance: {weighted_dist}")
    # print(f"Mean diff: {mean_diff}")
# %%


weighted_dist_list = []
mean_dist_list = []
num_affected_list = []
max_dist_list = []
std_dist_list = []
percentile_25_list = []
percentile_75_list = []

distance_values_per_gamma = []

for j, gamma in enumerate(gammas):
    # for i, catchment in enumerate(catchment_list):
    hospital_gdf = gdf_list[j]
    affected = hospital_gdf[
        hospital_gdf["population_percentual_diff"] > affected_threshold
    ]

    # Collect raw distances for this gamma
    distance_values = affected["min_dist"].values
    distance_values_per_gamma.append(distance_values)

    mean_dist = affected["min_dist"].mean()
    weighted_dist = np.average(
        affected["min_dist"], weights=affected["population_percentual_diff"]
    )
    num_affected = affected.shape[0]
    max_dist = affected["min_dist"].max()
    standard_deviation = affected["min_dist"].std()
    percentile_25 = affected["min_dist"].quantile(0.25)
    percentile_75 = affected["min_dist"].quantile(0.75)
    # dist_deviation = affected["min_dist"] - mean_dist

    weighted_dist_list.append(weighted_dist)
    max_dist_list.append(max_dist)
    mean_dist_list.append(mean_dist)
    std_dist_list.append(standard_deviation)
    num_affected_list.append(num_affected)
    percentile_25_list.append(percentile_25)
    percentile_75_list.append(percentile_75)


# %%

fig, ax = plt.subplots(1, 1, figsize=(8, 6))


xpos = np.array(gammas) / 0.8
bp = ax.boxplot(
    distance_values_per_gamma,
    positions=xpos,
    widths=0.01,  # adjust as needed
    patch_artist=True,
    showfliers=False,
    boxprops=dict(facecolor="lightblue", color="blue"),
    medianprops=dict(color="black"),
    whiskerprops=dict(color="blue"),
    capprops=dict(color="blue"),
    showmeans=True,
    meanline=True,
    meanprops={"color": "red", "linewidth": 4, "linestyle": "-"},
)
mean_line = mlines.Line2D([], [], color="red", linewidth=2, label="Mean")

ax.set_yticklabels([f"{g:.0f}" for g in ax.get_yticks()])

ax.set_xlim(-0.025, 0.15)
ax.set_xticks(xpos)
ax.set_xticklabels([f"{g:.3f}" for g in xpos])

ax.grid()

ax.legend(handles=[mean_line], loc="upper left")

for i, d in enumerate(distance_values_per_gamma, start=1):
    count = len(d)
    # print(count)
    x = xpos[i - 1]
    q3 = np.percentile(d, 75)
    ax.text(
        x + 0.003,
        q3 + 2500,
        f"n={count}",
        ha="center",
        va="bottom",
        fontsize=20,
        rotation=270,
    )

ax.set_xlabel(r"traffic magnitude $\gamma$")
ax.set_ylabel("Distance to nearest inundation [m]")


fig.savefig("germanyFloods/figs/SI-traffic-sensitivity-boxplot.pdf", dpi=300)

# %%
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

labels = [r"\textbf{a}", r"\textbf{b}"]

ax = axs[0]
ax.text(
    0.05, 1.06, labels[0], transform=ax.transAxes, fontsize=26, va="top", ha="right"
)
# ax.plot(gammas, mean_dist_list / np.max(mean_dist_list), label="Mean distance")
ax.plot(
    np.array(gammas) / 0.8,
    num_affected_list,
    label="# affected hospitals",
    marker="o",
    linewidth=4,
    markersize=12,
)
ax.set_xlabel(r"traffic magnitude $\gamma$")
ax.set_ylabel("Number of affected hospitals")
ax.grid()

ax1 = axs[1]
ax1.text(
    0.05, 1.06, labels[1], transform=ax1.transAxes, fontsize=26, va="top", ha="right"
)


xpos = np.array(gammas) / 0.8
ax1.boxplot(
    distance_values_per_gamma,
    positions=xpos,
    widths=0.01,  # adjust as needed
    patch_artist=True,
    showfliers=False,
    boxprops=dict(facecolor="lightblue", color="blue"),
    medianprops=dict(color="black"),
    whiskerprops=dict(color="blue"),
    capprops=dict(color="blue"),
    # flierprops=dict(marker='o', markerfacecolor='red', markersize=6, linestyle='none')
)

ax1.plot(
    xpos,
    mean_dist_list,
    label="Mean",
    marker="o",
    linewidth=4,
    markersize=12,
)

for ax in axs:
    ax.set_xlim(-0.025, 0.15)
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{g:.3f}" for g in xpos], rotation=90)

ax1.legend(loc="upper left")
ax1.grid()
ax1.set_xlabel(r"traffic magnitude $\gamma$")
ax1.set_ylabel("Distance to nearest inundation [m]")

fig.savefig("germanyFloods/figs/SI-traffic-sensitivity.png", dpi=300)
fig.savefig("germanyFloods/figs/SI-traffic-sensitivity.pdf", dpi=300)


# %%


fig, axs = plt.subplots(2, 3, figsize=(12, 10))

vmax = max([gdf["population_percentual_diff"].max() for gdf in gdf_list])
vmin = 1e-2

cmap = plt.cm.get_cmap("Reds")
cmap.set_under("white")
norm = plt.cm.colors.LogNorm(vmin=vmin, vmax=vmax)


for i, ax in enumerate(axs.flatten()):
    gdf = gdf_list[i]
    gdf.plot(ax=ax, column="population_percentual_diff", cmap=cmap, norm=norm)
    catchment_gdf.plot(ax=ax, color="none", edgecolor="black")
    ax.set_title(f"Gamma={gammas[i]}")


# %%

fig.savefig("germanyFloods/figs/SI-traffic-sensitivity-map.pdf")

# %%
