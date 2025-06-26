# %%
from germanyFloods import readFiles as rf
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from src import Plotting as pl
import pandas as pd

pl.mpl_params(fontsize=22)

# %%
path = "germanyFloods/data/"

gdf = rf.read_hospital_catchement3(
    catchment="all", population_kwd="prob_service_population", gamma=0.08
)
gdf.sort_values("population_percentual_diff", ascending=True, inplace=True)
# %%

pop_thresh = 0.3
distance_thresh = 10_000

hidden_df = gdf[
    (gdf["population_percentual_diff"] >= pop_thresh)
    & (gdf["min_dist"] >= distance_thresh)
]

pd.DataFrame(hidden_df.drop_duplicates(subset=["node"], keep="last"))


# %%


table = pd.DataFrame(
    {
        "Name": hidden_df["amenity_name"],
        "Count": hidden_df["amenity_name"],
        "Min Distance": hidden_df["min_dist"],
        "Population Difference": hidden_df["population_percentual_diff"],
        "Event": hidden_df["event"],
        "Catchment": hidden_df["catchment"],
    }
)


# %%

grouped_table = (
    table.groupby("Name")
    .agg(
        {
            "Catchment": lambda x: x.unique()[0],
            "Count": "count",
            "Min Distance": "mean",
            "Population Difference": "mean",
            # "Event": list,
        }
    )
    .sort_values("Count", ascending=False)
    .reset_index()
)

catchemnt_dict = {
    "elbe_upper": "Upper Elbe",
    "elbe_lower": "Lower Elbe",
    "ems": "Ems",
    "rhine_upper": "Upper Rhine",
    "rhine_lower": "Lower Rhine",
    "weser": "Weser",
    "donau": "Danube",
}

grouped_table["Name"] = grouped_table["Name"].str.replace(r"\+", r"\\\\", regex=True)
grouped_table["Name"] = grouped_table["Name"].str.replace(r" - ", r"\\\\", regex=True)
grouped_table["Name"] = grouped_table["Name"].str.replace(
    r"\bund\b", r"\\\\", regex=True
)
grouped_table["Name"] = grouped_table["Name"].apply(
    lambda x: r"\makecell[l]{" + x + "}"
)
grouped_table["Population Difference"] = (
    round(grouped_table["Population Difference"], 2)
    .astype(str)
    .str.rstrip("0")
    .str.rstrip(".")
)
grouped_table["Min Distance"] = (
    round(grouped_table["Min Distance"], 2).astype(str).str.rstrip("0").str.rstrip(".")
)


def format_events(events):
    events_chunks = [
        ", ".join(map(str, events[i : i + 3])) for i in range(0, len(events), 3)
    ]
    return r"\makecell{" + r"\\ ".join(events_chunks) + "}"


# grouped_table["Event"] = grouped_table["Event"].apply(format_events)

grouped_table["Catchment"] = grouped_table["Catchment"].map(catchemnt_dict)
# Convert the DataFrame to a LaTeX table
latex_table = grouped_table.to_latex(
    index=False,
    column_format="lrrll",
    header=[
        "Name",
        "Count",
        "Min Distance",
        "Population Difference",
        "Catchment",
    ],
    escape=False,
)
grouped_table
# %%
print(latex_table)
# %%
