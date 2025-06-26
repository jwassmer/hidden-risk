# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import geopandas as gpd
import cartopy.crs as ccrs

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER
from matplotlib.ticker import FuncFormatter

from src import Plotting as pl
from src import GeoModule as gm
from src import FloodRaster as fr
from src import RoadNetwork2 as rn
from src import WetRoads as wr


driving_tags = [
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    # "residential",
]

west, east = 7.24, 7.34
south, north = 52.68, 52.73
buffer = 30

bbox = gm.bbox(west=west, east=east, south=south, north=north)

# %%


basin_gdf = gpd.read_file(
    "data_LFS/basin-polygons/NUTS_hydro_divisions_1010.geojson"
).to_crs("EPSG:4326")

# path_ds_wsh = "data_LFS/haz/rim2019/downscale/rim2019_wd_ems_day-20141_realisation-13_raster_index-33_wse_781f853c/wse2_clip_fdsc-r05.tif"
path_ds = "germanyFloods/data/wse2_clip_fdsc-r05_wsh.tif"
path = "germanyFloods/data/rim2019_wd_ems_day-20141_realisation-13_raster_index-33.tif"


# raster_ds_wsh = fr.read_raster(path_ds_wsh)
raster_ds = fr.read_raster(path_ds)
raster = fr.read_raster(path)

rasters = [raster, raster_ds]
# %%

region = rn.RoadNetwork(
    osmpbf="ger-buffered-200km.osm.pbf",
    highway_filter=f"w/highway={','.join(driving_tags)}",
    gdf=bbox.buffer(buffer),
)


key = "amenity"
tag = "hospital"
region.add_pois(key, tag)

edges = region.edges

bridge_gdf = wr.get_bridge_polys(region)
tunnel_gdf = wr.get_tunnel_polys(region)
# %%
cmap = plt.get_cmap("cividis_r")
cmap.set_under("None")
norm = mpl.colors.Normalize(vmin=1e-3, vmax=3)

fig, axs = plt.subplots(
    1, 2, subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(13, 8)
)
scale = ["normal", "downscale"]
labels = [r"\textbf{a}", r"\textbf{b}"]


for i, raster in enumerate(rasters):
    axs[i].text(0.01, 1.01, labels[i], transform=axs[i].transAxes, fontsize=26)
    axs[i].set_extent(
        [bbox.west, bbox.east, bbox.south, bbox.north], crs=ccrs.PlateCarree()
    )
    gl = axs[i].gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=1,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = False
    if i == 1:
        gl.left_labels = False
    # gl.xformatter = LONGITUDE_FORMATTER
    # axs[i].set_global()
    # axs[i].set_xticks(, crs=ccrs.PlateCarree())
    xticks = [7.245, 7.275, 7.305, 7.335]
    axs[i].set_xticks(xticks)
    axs[i].xaxis.set_major_formatter(LONGITUDE_FORMATTER)

    bridge_gdf.plot(ax=axs[i], color="black", zorder=4, linewidth=4, label="Bridge")
    # tunnel_gdf.plot(ax=axs[i], color="black", zorder=4, linewidth=4, label="Tunnel")

    region_c = region.copy()

    region_c.water_depth(raster, polygon=bbox.shape)
    wet_roads = region_c.edges[region_c.edges["water_depth"] > 0.3].index
    print("There are", len(wet_roads), "wet roads")

    region_c.edges.loc[wet_roads].plot(
        ax=axs[i], color="red", zorder=3, linewidth=2, label="Disrupted"
    )

    region_c.edges.plot(
        ax=axs[i],
        color="grey",
        alpha=0.5,
        zorder=1,
        linewidth=2,
        label="Road",
    )

    rasters[i].plot(
        ax=axs[i],
        cmap=cmap,
        norm=norm,
        add_colorbar=False,
        add_labels=False,
        rasterized=True,
    )


cbar = fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=axs,
    shrink=1 / 2,
    orientation="vertical",
    extend="both",
    pad=0.01,
    aspect=25,
)
cbar.ax.set_ylabel(r"Water surface height (WSH) $w_{kl}$ [m]")
axs[1].legend(loc="upper right", bbox_to_anchor=[1.1, 1.1])

# %%

fig.savefig(
    "germanyFloods/figs/SI-downscaled-water-depth-bridges.png",
    dpi=300,
    bbox_inches="tight",
)


# %%
fig.savefig(
    "germanyFloods/figs/SI-downscaled-water-depth-bridges.pdf",
    dpi=300,
    bbox_inches="tight",
)

# %%
