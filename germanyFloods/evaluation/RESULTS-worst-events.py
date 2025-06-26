# %%
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import geopandas as gpd
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import ListedColormap

from src import FloodRaster as fr
from src import Plotting as pl
from src import EmergencyModule as em

from germanyFloods import readFiles as rf

RUN_AGG_EVENTS = True

gamma = 0.08

raster_path = "data_LFS/haz/rim2019/0303_downscale_20240629"
path = "germanyFloods/data"
population_kwd = "prob_service_population"

if RUN_AGG_EVENTS:
    access_gdf = rf.read_hospital_catchement3(
        catchment="all", population_kwd=population_kwd, gamma=gamma
    )
    # sort by population percentual difference
    access_gdf.sort_values(by=["population_percentual_diff", "min_dist"], inplace=True)

    # drop dupes
    access_no_duplicates = access_gdf.drop_duplicates(
        subset=["node"], inplace=False, keep="last"
    )
    access_no_duplicates = access_no_duplicates.dropna(
        subset="population_percentual_diff"
    )  # WHERE DO NA COME FROM?

    tail = access_no_duplicates.tail(9)
    tail.sort_values(by="population_percentual_diff", ascending=False, inplace=True)
    tail.to_csv(f"{path}/worst_9_events.csv", index=False)
else:
    tail = pd.read_csv(f"{path}/worst_9_events.csv")
    tail = tail.sort_values(by="population_percentual_diff", ascending=False)


# %%
catchment_dict = {
    "donau": "Danube",
    "rhine_lower": "Lower Rhine",
    "rhine_upper": "Upper Rhine",
    "elbe_lower": "Lower Elbe",
    "elbe_upper": "Upper Elbe",
    "ems": "Ems",
    "weser": "Weser",
}
# %%


catchment_code = [4, 0, 6, 4, 4, 6, 6, 4, 4]
tail["catchment_code"] = catchment_code
tail["catchment_code"] = tail["catchment_code"].astype("category")
catchments = tail["catchment"].values
events = tail["event"].values
hospital_names = tail["amenity_name"].values
osmids = tail["node"].values

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
tab_cmap = ListedColormap(new_order)
tab_norm = mpl.colors.Normalize(vmin=0, vmax=7)

# %%


hospital_list = []
nodes_pre_list = []
nodes_list = []
edges_list = []
raster_list = []
access_gdf_list = []
access_gdf_r_list = []

for j, (catchment, event, hospital_name, osmid) in enumerate(
    zip(catchments, events, hospital_names, osmids)
):
    print(j, catchment, event, hospital_name)

    nodes = gpd.read_file(
        f"germanyFloods/data/{catchment}/{event}/gamma_{gamma}/{hospital_name}/nodes.geojson"
    ).set_index("osmid")
    nodes_pre = gpd.read_file(
        f"germanyFloods/data/{catchment}/{event}/gamma_{gamma}/{hospital_name}/nodes_pre.geojson"
    ).set_index("osmid")
    edges = gpd.read_file(
        f"germanyFloods/data/{catchment}/{event}/gamma_{gamma}/{hospital_name}/edges.geojson"
    )
    eps = 0.1
    bounds = nodes.total_bounds
    bounds = (bounds[0] + eps, bounds[1] + eps, bounds[2] - eps, bounds[3] - eps)
    raster = fr.read_total_event(
        catchment, event, bounds=bounds, rescale=0.5, path=raster_path
    )
    access_gdf = gpd.read_file(
        f"{path}/{catchment}/gamma_{gamma}/hospital-access.geojson"
    )
    access_gdf["node"] = access_gdf["node"].astype(int).astype(object)
    access_gdf = access_gdf.set_index("node")

    access_gdf_r = gpd.read_file(
        f"{path}/{catchment}/{event}/gamma_{gamma}/hospital-access.geojson"
    )
    access_gdf_r["node"] = access_gdf_r["node"].astype(int).astype(object)
    access_gdf_r = access_gdf_r.set_index("node")

    access_gdf_r["population_pre_flood"] = access_gdf[population_kwd]
    access_gdf_r.rename(columns={population_kwd: "population_post_flood"}, inplace=True)
    access_gdf_r["population_diff"] = (
        access_gdf_r["population_post_flood"] - access_gdf_r["population_pre_flood"]
    )
    access_gdf_r["population_percentual_diff"] = (
        access_gdf_r["population_post_flood"] - access_gdf_r["population_pre_flood"]
    ) / access_gdf_r["population_pre_flood"]

    # hospital = access_gdf[access_gdf["amenity_name"] == hospital_name]
    hospital_r = access_gdf_r.loc[osmid]

    hospital_list.append(hospital_r)
    nodes_pre_list.append(nodes_pre)
    nodes_list.append(nodes)
    edges_list.append(edges)
    raster_list.append(raster)
    access_gdf_list.append(access_gdf)
    access_gdf_r_list.append(access_gdf_r)


# %%

labels = [
    r"\textbf{a}",
    r"\textbf{b}",
    r"\textbf{c}",
    r"\textbf{d}",
    r"\textbf{e}",
    r"\textbf{f}",
    r"\textbf{g}",
    r"\textbf{h}",
    r"\textbf{i}",
]

markers = ["o", "s", "D", "v", "^", "p", "*", ">", "<"]

listed_cmap = mpl.colors.ListedColormap(["None", cfeature.COLORS["water"]])
# listed_cmap = mpl.colors.ListedColormap(["None", "magenta"])
river_norm = mpl.colors.Normalize(vmin=1e-3, vmax=1)

Blues_cmap = plt.cm.Blues

colors = Blues_cmap(np.linspace(0.5, 1.0, 256))
raster_cmap = LinearSegmentedColormap.from_list("custom_blue", colors)
raster_cmap = plt.cm.cividis_r
raster_cmap.set_bad("none")
raster_cmap.set_under("none")
raster_norm = mpl.colors.LogNorm(vmin=0.3, vmax=0.31)


fig, axs = plt.subplots(
    nrows=3,
    ncols=3,
    figsize=(24, 16),
    constrained_layout=True,
    subplot_kw={"projection": ccrs.PlateCarree()},
)

axs = axs.flatten()
# fig.subplots_adjust(hspace=0.1, wspace=0.0)
for j, (catchment, event, hospital_name, osmid) in enumerate(
    zip(catchments, events, hospital_names, osmids)
):
    print(j, catchment, event, hospital_name)

    ax = axs[j]

    ax.text(
        0.007,
        0.99,
        labels[j],
        transform=ax.transAxes,
        fontsize=36,
        va="top",
        ha="left",
        zorder=5,
    )

    nodes = nodes_list[j]
    edges = edges_list[j]
    raster = raster_list[j]
    access_gdf = access_gdf_list[j]
    access_gdf_r = access_gdf_r_list[j]

    if "upper" in catchment:
        major_catchment = catchment.replace("_upper", "")
    elif "lower" in catchment:
        major_catchment = catchment.replace("_lower", "")
    else:
        major_catchment = catchment
    print(major_catchment)
    hospital = access_gdf.loc[osmid:osmid]

    hospital_r = access_gdf_r.loc[osmid:osmid]

    x_off = 0
    if j == 7:
        x_off = +0.05

    lat, lon = (
        hospital_r.geometry.centroid.y.values[0],
        hospital_r.geometry.centroid.x.values[0] + x_off,
    )
    delta_lon = 0.3  # longitude span remains constant
    ratio = (1 + np.sqrt(5)) / 2

    # Adjust delta_lat based on the cosine of the latitude to maintain more consistent visual sizes
    delta_lat = delta_lon / ratio * np.cos(np.radians(lat))
    xmin, xmax = lon - delta_lon, lon + delta_lon
    ymin, ymax = lat - delta_lat, lat + delta_lat

    river_raster = fr.read_raster(
        f"data_LFS/haz/rim2019/burned_domains/{major_catchment}.tif",
        bounds=[xmin, ymin, xmax, ymax],
    )
    river_raster.values = np.where(
        river_raster.values < 9999, np.nan, river_raster.values
    )
    # river_raster.values = np.where(river_raster.values == 9999, 1, river_raster.values)
    river_raster.plot(
        ax=ax,
        add_colorbar=False,
        add_labels=False,
        cmap=listed_cmap,
        norm=river_norm,
        rasterized=True,
    )

    if raster is not None:
        raster.plot(
            ax=ax,
            zorder=2,
            cmap=raster_cmap,
            norm=raster_norm,
            add_colorbar=False,
            add_labels=False,
            rasterized=True,
        )

    hospital_r.plot(
        ax=ax,
        # column="population_percentual_diff",
        zorder=1,
        color="lightgrey",
        alpha=0.75,
    )
    hospital.plot(
        ax=ax,
        # column="population_percentual_diff",
        zorder=1,
        color="black",
        alpha=0.75,
    )
    # access_gdf_r.boundary.plot(ax=ax, color="pink", zorder=4, linewidth=1)
    edges.plot(ax=ax, color="grey", linewidth=0.5, zorder=3)

    removed = edges[edges["removed"] == "1"]
    if len(removed) > 0:
        removed.plot(ax=ax, color="red", linewidth=1, zorder=3)

    hospital_r.boundary.plot(ax=ax, color="black", zorder=3, linewidth=1.5)
    hospital.boundary.plot(ax=ax, color="black", zorder=3, linewidth=1.5)
    hospital_nodes = nodes[nodes["amenity"] == "hospital"]
    hospital_nodes.loc[:, "min_dist"] = np.inf
    min_dist_list = hospital_nodes.geometry.apply(
        lambda pt: em.min_dist_to_raster(pt, raster)
    )
    hospital_nodes.loc[:, "min_dist"] = np.minimum(
        hospital_nodes["min_dist"], min_dist_list
    )
    hospital_nodes["population_percentual_diff"] = access_gdf_r[
        "population_percentual_diff"
    ]

    hospital_nodes[hospital_nodes["amenity_name"] != hospital_name].plot(
        ax=ax,
        color="lightgrey",
        zorder=3,
        markersize=250,
        marker="P",
        edgecolor="black",
        linewidth=0.5,
    )

    innudated_hospitals = hospital_nodes[hospital_nodes["min_dist"] <= 50]

    if len(innudated_hospitals) > 0:
        innudated_hospitals[innudated_hospitals["amenity_name"] != hospital_name].plot(
            ax=ax,
            color=cfeature.COLORS["water"],
            zorder=3,
            markersize=250,
            marker="P",
            edgecolor="black",
            linewidth=0.5,
            # alpha=0.7,
        )

    color = tab_cmap(tail["catchment_code"].values[j])
    delta_s = 0
    if j == 6:
        delta_s = 250
    hospital_nodes[hospital_nodes["amenity_name"] == hospital_name].plot(
        ax=ax,
        color=color,
        zorder=3,
        markersize=500 + delta_s,
        marker=markers[j],
        edgecolor="black",
        linewidth=0.5,
        # alpha=0.7,
    )

    # access_gdf_r.plot(ax=ax, color="black", zorder=0, alpha=0.5)

    # ax.set_xlim(xmin, xmax)
    # ax.set_ylim(ymin, ymax)
    ax.set_extent([xmin, xmax, ymin, ymax], crs=ccrs.PlateCarree())

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=1,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )

    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"rotation": 45}

    dist = hospital_r["min_dist"].values[0]

    pos_x, pos_y = hospital_r["hospital_location_x"], hospital_r["hospital_location_y"]

    lat, lon = round(pos_y.values[0], 2), round(pos_x.values[0], 2)

    if dist is None:
        dist = 0
    textstr = "\n".join(
        (
            f"Loc: {lon}°E, {lat}°N",
            f"{catchment_dict[catchment]}, {event}",
            f"$d_E(k_H,u_{{kl}})=${dist:.0f}m",
            f"$\Delta N(k_H)=${hospital_r['population_percentual_diff'].values[0]:.2f}%",
            # rf"$N_{{\mathrm{{pre}}}}:$ {hospital_r['population_pre_flood'].values[0]:.0f}",
            # rf"$N_{{\mathrm{{post}}}}:$ {hospital_r['population_post_flood'].values[0]:.0f}",
        )
    )
    props = dict(boxstyle="square", facecolor="white")

    hospital_name = hospital_name.replace("+", "\n \& ")
    hospital_name = hospital_name.replace("oich - ", "oich - \n ")
    hospital_name = hospital_name.replace(" für ", "\n für  ")
    hospital_name = hospital_name.replace(" an ", "\n an  ")

    inset_ax = inset_axes(
        ax,
        width="100%",
        height="100%",
        loc="upper left",
        bbox_to_anchor=(-0.04, 1.03, 0.1, 0.1),
        bbox_transform=ax.transAxes,
    )

    inset_ax.scatter(
        0.0,
        0.0,
        color=color,
        edgecolor="black",
        marker=markers[j],
        s=500 + delta_s,
    )
    inset_ax.axis("off")

    ax.text(
        0.97,
        0.97,
        textstr,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=props,
        fontsize=26,
    )
    ax.set_title(f"{hospital_name}", fontsize=26)

    # ax.scatter(1, 1, markers[j])


# ticks = cbar.get_ticks()
# ticklabels = [f"{tick*100:.0f}$\%$" for tick in ticks]
# cbar.ax.set_xticklabels(ticklabels)

# cbar2 = fig.colorbar(
#    mpl.cm.ScalarMappable(norm=raster_norm, cmap=raster_cmap),
#    ax=axs,
#    pad=0.01,
#    aspect=40,
#    shrink=1 / 2,
#    extend="both",
# )
# cbar2.ax.set_ylabel("Water surface height (WSH) $u_{kl}$ [m]", fontsize=26)
# cbar2.ax.tick_params(labelsize=26)


flood_patch = mpatches.Patch(
    facecolor=raster_cmap(255),
    # edgecolor="lightblue",
    label="Overbank flooding",
    linewidth=0,
)

water_patch = mpatches.Patch(
    facecolor=cfeature.COLORS["water"],
    # edgecolor="lightblue",
    label="Permanent water",
    linewidth=0,
)
road_patch = mlines.Line2D(
    [],
    [],
    color="grey",
    label="Road",
    linestyle="-",
    linewidth=1,
)
removed_patch = mlines.Line2D(
    [],
    [],
    color="red",
    label="Disrupted road",
    linestyle="-",
    linewidth=1,
)
poly_patch1 = mpatches.Patch(
    facecolor="lightgrey",
    edgecolor="black",
    label="Post-event service area",
    linewidth=1,
)

poly_patch0 = mpatches.Patch(
    facecolor="black",
    edgecolor="black",
    label="Pre-event service area",
    linewidth=1,
)

hospital = ax.scatter(
    [], [], color="white", edgecolor="black", label=r"Hospital $k_H$", s=150, marker="P"
)
fig.legend(
    handles=[
        poly_patch0,
        poly_patch1,
        road_patch,
        removed_patch,
        water_patch,
        flood_patch,
        hospital,
    ],
    # loc="upper right",
    bbox_to_anchor=(0.67 + 0.16, 0.01),
    framealpha=0.5,
    fontsize=26,
    ncol=4,
    # alpha=0.8,
)
# %%
fig.savefig("germanyFloods/figs/RESULTS-worst_events.pdf", bbox_inches="tight", dpi=72)


# %%
fig.savefig("germanyFloods/figs/RESULTS-worst_events.png", bbox_inches="tight", dpi=300)
# fig.savefig("germanyFloods/figs/worst_events.pdf", dpi=300)


# %%

# %%
fig.savefig("germanyFloods/figs/RESULTS-worst_events.svg", dpi=300)

# %%


for j, (catchment, event, hospital_name, osmid) in enumerate(
    zip(catchments, events, hospital_names, osmids)
):
    print(j, catchment, event, hospital_name)
    if j == 6:
        break

nodes = gpd.read_file(
    f"germanyFloods/data/{catchment}/{event}/gamma_{gamma}/{hospital_name}/nodes.geojson"
).set_index("osmid")
nodes_pre = gpd.read_file(
    f"germanyFloods/data/{catchment}/{event}/gamma_{gamma}/{hospital_name}/nodes_pre.geojson"
).set_index("osmid")
edges = gpd.read_file(
    f"germanyFloods/data/{catchment}/{event}/gamma_{gamma}/{hospital_name}/edges.geojson"
)


# %%
eps = 0.1
bounds = nodes.total_bounds
bounds = (bounds[0] + eps, bounds[1] + eps, bounds[2] - eps, bounds[3] - eps)

raster = fr.read_total_event(
    catchment, event, bounds=bounds, rescale=0.5, path=raster_path
)

# %%
raster.plot()
# %%


# %%
