# %%
import os
import sys
import pickle

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import osmnx as ox
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FormatStrFormatter

from tqdm import tqdm
from src import FloodRaster as fr

# from src import Plotting as pl
from germanyFloods import readFiles as rf
from scipy.stats import linregress

# pl.mpl_params(fontsize=22)


catchment_gdf = gpd.read_file(
    "data_LFS/basin-polygons/NUTS_hydro_divisions_1010.geojson"
)

path = "germanyFloods/data/"

# %%
gdf = rf.read_hospital_catchement3(
    catchment="all", population_kwd="prob_service_population", gamma=0.08
)

# %%
# agg_gdf = rf.aggregate_hospitals(
#    gdf, col="population_percentual_diff", new_index="idxmax"
# )
# agg_gdf.sort_values("population_percentual_diff", ascending=True, inplace=True)


agg_gdf = gdf.sort_values("population_percentual_diff", ascending=True)
agg_gdf = agg_gdf.drop_duplicates(subset=["node"], keep="last")
agg_gdf = agg_gdf[agg_gdf["population_percentual_diff"] > 0]


agg_gdf["catchment_code"] = agg_gdf["catchment"].astype("category").cat.codes
d = dict(enumerate(agg_gdf["catchment"].astype("category").cat.categories))
for key, value in d.items():
    if "_" in value:
        str0, str1 = value.split("_")
        d[key] = f"{str1} {str0}"
    # Capitalize all values
    for key, value in d.items():
        d[key] = value.title()

d[0] = "Danube"

tail_gdf = agg_gdf.drop_duplicates(subset=["amenity_name"], keep="last").tail(9)

# %%
tab_cmap = mpl.cm.get_cmap("tab10")
colors = tab_cmap.colors
new_order = [
    colors[7],
    colors[1],
    colors[2],
    colors[3],
    colors[8],
    colors[0],
    colors[-1],
]
cmap = ListedColormap(new_order)
norm = mpl.colors.Normalize(vmin=0, vmax=7)


ycol = "population_percentual_diff"

agg_gdf[ycol] = agg_gdf[ycol].replace(0.0, np.nan)
agg_gdf["min_dist"] = agg_gdf["min_dist"].replace(0.0, np.nan)
agg_gdf.dropna(subset=["min_dist", ycol], inplace=True)
agg_gdf.sort_values(ycol, inplace=True)


fig, ax = plt.subplots(figsize=(12, 8))

ax.scatter(
    agg_gdf["min_dist"],
    agg_gdf[ycol],
    c=agg_gdf["catchment_code"],
    # label="Hospitals",
    s=40,
    marker="+",
    # alpha=0.75,
    cmap=cmap,
    norm=norm,
)
ax.scatter([], [], c="grey", s=150, marker="+", label="Hospital")


ax.grid()
ax.set_yscale("log")
ax.set_xscale("log")


ax.set_xlabel(
    r"Euclidean distance to the nearest inundation [m]"
)  # , $\mathrm{dist}_E(k_H, u_{kl})$ [m]"
ax.set_ylabel(
    r"Maximum change in service population max($\Delta N(k_H)$)"
)  # , max$(\Delta N(k_H))$")


markers = ["o", "s", "D", "v", "^", "p", "*", ">", "<"]
for i, (j, row) in enumerate(tail_gdf[::-1].iterrows()):
    print(row["catchment"], row["event"])
    c = cmap(row["catchment_code"])
    ax.scatter(
        row["min_dist"],
        row[ycol],
        c=row["catchment_code"],
        s=250,
        marker=markers[i],
        cmap=cmap,
        norm=norm,
        edgecolor="black",
        zorder=3,
    )


pos_gdf = agg_gdf[agg_gdf[ycol] > 0]
x = np.log10(pos_gdf["min_dist"])
y = np.log10(pos_gdf[ycol])

slope, intercept, r_value, p_value, std_err = linregress(x, y)

# Calculate points for the trend line
x_trend = np.linspace(min(x) - 0.1, max(x), 100)
y_trend = slope * x_trend + intercept
trendline = intercept + slope * x_trend

# Convert the trend line points back from log scale to original scale
ax.plot(
    10 ** (x_trend),
    10 ** (y_trend),
    "black",
    linestyle="--",
    linewidth=3,
    label=rf"$y = {10**intercept:.0f} \cdot x^{{{slope:.2f}}}$",  # x^{slope:.2f}",
)

ci = 1.96 * 10 ** (std_err)  # 95% confidence interval multiplier for standard error
upper_bound = trendline + ci
lower_bound = trendline - ci

ax.plot(x_trend, 10**intercept * x_trend**slope, "red", linestyle="--", linewidth=5)

thirty_percent_line = 0.3
ten_km_line = 10_000


# ax.axvline(
#    x=ten_km_line,
#    color="grey",
#    linestyle=":",
#    linewidth=3,
#    label="$\mathrm{dist}_E(k_H, u_{kl})=$10 km",
#    zorder=2,
# )

# ax.axhline(
#    y=thirty_percent_line,
#    color="grey",
#    linestyle="-",
#    linewidth=3,
#    label="$\Delta N(k_H) = 0.3$",
#    zorder=0,
# )


xvals = np.linspace(min(x_trend), max(x_trend), 1000)
ax.fill_between(
    10**xvals,
    thirty_percent_line,
    1000,
    where=(10**xvals >= (ten_km_line)),
    color="none",  # or simply omit this line
    hatch="//",
    edgecolor="black",  # Ensures the hatch lines are black
    alpha=0.8,
    label="Hidden risk",
    zorder=0,
)


# ax.fill_between(x, lower_bound, upper_bound, color='red', alpha=0.2)
# ax.fill_between(
#    sorted(10 ** (x_trend)),
#    10 ** (lower_bound[np.argsort(x_trend)]),
#    10 ** (upper_bound[np.argsort(x_trend)]),
#    color="black",
#    alpha=0.2,
#    label="95\% CI",
#    zorder=0,
# )

ax.set_ylim(1e-5, 2e1)
ax.set_xlim(70 * min(x_trend), 10 ** max(x_trend))
ax.legend(loc="lower left")
# ax.set_title("Hospital catchment area change vs. distance to flooded area")
cbar = plt.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=ax,
    orientation="vertical",
    label="Analysis basin",
    shrink=0.75,
    pad=0.01,
)
cbar.set_ticks(np.arange(0.5, 7.5, 1))
try:
    cbar.set_ticklabels(d.values(), rotation=-45)
except:
    pass

# ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))


# %%

fig.savefig(
    f"germanyFloods/figs/RESULTS-hidden-risk.png",
    bbox_inches="tight",
    dpi=300,
)

fig.savefig(
    f"germanyFloods/figs/RESULTS-hidden-risk.pdf",
    bbox_inches="tight",
    dpi=300,
)
# %%


# count number of hospitals in hidden risk window
hidden_risk = agg_gdf[
    (agg_gdf["min_dist"] >= ten_km_line) & (agg_gdf[ycol] >= thirty_percent_line)
]


hidden_risk = hidden_risk[
    ["amenity_name", "catchment", "min_dist", "population_percentual_diff"]
].sort_values("population_percentual_diff", ascending=False)


hidden_risk["min_dist"] = hidden_risk["min_dist"].round(2).astype(str)
hidden_risk["population_percentual_diff"] = (
    hidden_risk["population_percentual_diff"].round(2).astype(str)
)

print("There are", len(hidden_risk), "hospitals in the hidden risk window.")

# %%
tex_table = hidden_risk.to_latex(
    index=False,
    escape=False,
)

print(tex_table)


# %%

# %%
