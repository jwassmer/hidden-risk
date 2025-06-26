# %%
import geopandas as gpd
import pandas as pd
from src import Plotting as pl
import matplotlib.pyplot as plt
import numpy as np

from germanyFloods import readFiles as rf

# %%
catchemnt_dict = {
    "elbe_upper": "Upper Elbe",
    "elbe_lower": "Lower Elbe",
    "ems": "Ems",
    "rhine_upper": "Upper Rhine",
    "rhine_lower": "Lower Rhine",
    "weser": "Weser",
    "donau": "Danube",
}


catchment_gdf = gpd.read_file(
    "data_LFS/basin-polygons/NUTS_hydro_divisions_1010.geojson"
)
path = "germanyFloods/data"

gdf = rf.read_hospital_catchement3(
    catchment="all", population_kwd="prob_service_population", gamma=0.08
)

gdf.sort_values("population_percentual_diff", ascending=True, inplace=True)
# %%

europe_gdf = gpd.read_file("data/europe.geojson")
europe_gdf = europe_gdf.to_crs(epsg=4326)
ger_gdf = europe_gdf[europe_gdf["NAME"] == "Germany"]

# %%
# max_hospitals = gdf.drop_duplicates(subset="node", keep="last")
hospitals_affected = gdf[gdf["population_percentual_diff"] >= 0.01]


hospitals_affected.drop_duplicates(subset="node", keep="first", inplace=True)

# Set new geometry from hospital_location_x and y
hospitals_affected["location"] = gpd.points_from_xy(
    hospitals_affected["hospital_location_x"], hospitals_affected["hospital_location_y"]
)

hospitals_affected.set_geometry("location", inplace=True)

# %%

hospitals_outside_germany = hospitals_affected[
    ~hospitals_affected.within(ger_gdf.unary_union)
]
# hospitals_outside_germany.plot(ax=plt.gca(), color="red")
# ger_gdf.boundary.plot(ax=plt.gca(), color="black")


# %%
print("There are", len(hospitals_outside_germany), "hospitals outside of Germany")
print("they lie in:")

for cntry_name, geom in zip(europe_gdf["NAME"], europe_gdf["geometry"]):
    amount = len(hospitals_outside_germany[hospitals_outside_germany.within(geom)])
    if amount > 0:
        print(cntry_name, amount)


# %%

affected_df = pd.DataFrame(columns=gdf.catchment.unique())
unique_gdf = gdf.drop_duplicates(subset="node", keep="last")

for k in [0.1, 0.3, 0.5, 1]:
    affected_hospitals = unique_gdf[unique_gdf["population_percentual_diff"] > k]
    grouped = affected_hospitals.groupby("catchment").count()["node"]

    affected_df.loc[k] = grouped

affected_df.replace(np.nan, 0, inplace=True)

affected_df = (
    affected_df.astype(int)
    .astype(str)
    .reset_index()
    .rename(columns={"index": "Population Difference"})
)


tex_table = affected_df.to_latex(
    index=False,
    column_format="lrrrll",
    escape=False,
)

print(tex_table)
affected_df

# %%

# catchment perspective
affected_hospitals = gdf[gdf["population_percentual_diff"] > 0.3]

table = pd.DataFrame(
    {
        "Node": affected_hospitals["node"],
        "Node2": affected_hospitals["node"],
        "Name": affected_hospitals["amenity_name"],
        "Min Distance": affected_hospitals["min_dist"],
        "Population Difference": affected_hospitals["population_percentual_diff"],
        "Event": affected_hospitals["event"],
        "Catchment": affected_hospitals["catchment"],
    }
)

grouped = (
    table.groupby("Catchment")
    .agg(
        {
            "Node": "count",
            "Node2": lambda x: len(set(x)),
            "Event": lambda x: len(set(x)),
            "Population Difference": "mean",
            "Min Distance": "mean",
        }
    )
    .reset_index()
)
grouped["Population Difference"] = (
    round(grouped["Population Difference"], 2)
    .astype(str)
    .str.rstrip("0")
    .str.rstrip(".")
)
grouped["Min Distance"] = (
    (round(grouped["Min Distance"], 2)).astype(str).str.rstrip("0").str.rstrip(".")
)

grouped = grouped.rename(
    columns={
        "Node": "Number of affected Hospitals",
        "Node2": "Number of unique affected Hospitals",
        "Event": "Number of Events",
        "Population Difference": "Mean Population Difference",
        "Min Distance": "Mean Flood Distance [m]",
    }
)
grouped["Catchment"] = grouped["Catchment"].map(catchemnt_dict)
grouped = grouped.sort_values(by="Number of affected Hospitals", ascending=False)
grouped


tex_table = grouped.to_latex(
    index=False,
    column_format="lrrrll",
    escape=False,
)

print(tex_table)
grouped


# %%

# event perspective

affected_hospitals = gdf[gdf["population_percentual_diff"] > 0.3]

table = pd.DataFrame(
    {
        "Node": affected_hospitals["node"],
        "Node2": affected_hospitals["node"],
        "Name": affected_hospitals["amenity_name"],
        "Min Distance": affected_hospitals["min_dist"],
        "Population Difference": affected_hospitals["population_percentual_diff"],
        "Event": affected_hospitals["event"],
        "Catchment": affected_hospitals["catchment"],
    }
)

grouped_events = (
    table.groupby(["Catchment", "Event"])
    .agg(
        {
            "Node": "count",
            "Population Difference": "mean",
            "Min Distance": "mean",
        }
    )
    .sort_values(by=["Node", "Population Difference"], ascending=False)
    .reset_index()
)
grouped_events["Population Difference"] = (
    round(grouped_events["Population Difference"], 2)
    .astype(str)
    .str.rstrip("0")
    .str.rstrip(".")
)
grouped_events["Min Distance"] = (
    (round(grouped_events["Min Distance"], 2))
    .astype(str)
    .str.rstrip("0")
    .str.rstrip(".")
)
grouped_events = grouped_events.rename(
    columns={
        "Node": "Number of affected Hospitals",
        "Population Difference": "Mean Population Difference",
        "Min Distance": "Mean Flood Distance [m]",
    }
)
grouped_events["Catchment"] = grouped_events["Catchment"].map(catchemnt_dict)
grouped_events

tex_table = grouped_events.to_latex(
    index=False,
    column_format="llrrl",
    escape=False,
)


# %%


gdf = gdf[gdf["population_percentual_diff"] >= 0.3]
gdf = gdf.drop_duplicates(subset="node", keep="last")

len(gdf)
# %%

no_dupes = gdf.drop_duplicates(subset=["node", "catchment"], keep="last", inplace=False)
# %%

no_dupes.groupby("catchment").count()["node"].sum()
# %%
