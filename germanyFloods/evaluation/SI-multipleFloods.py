# %%
from germanyFloods import readFiles as rf
import geopandas as gpd
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
# %%
# gdf = gdf[gdf["population_pre_flood"] > 1_000]
unique_hospitals = len(gdf["amenity_name"].unique())
print("Amount of hospitals: ", unique_hospitals)
print("Amount of events: ", len(gdf["event"].unique()))


for k in [0.3, 0.5, 1]:
    unique_hospitals_affected = len(
        gdf[gdf["population_percentual_diff"] > k]["amenity_name"].unique()
    )
    print(
        f"Amount of unique hospitals affected by more than {100*k}%: ",
        unique_hospitals_affected,
    )


# gdf.describe()

# %%

fig, ax = plt.subplots(figsize=(12, 8))

ax.grid()
ks = [0.1, 0.3, 0.5, 0.75, 1]

for k in ks:
    affected_hospitals = gdf[gdf["population_percentual_diff"] > k]["amenity_name"]

    duplicate_counts = affected_hospitals.value_counts()

    y, x = np.histogram(duplicate_counts, bins=range(1, max(duplicate_counts) + 2))

    cumsum = np.cumsum(y[::-1])
    # print(x[:-1][::-1])
    # print(cumsum[-9])
    if k == 0.3:
        print(cumsum)
        print(duplicate_counts[duplicate_counts >= 10])

    ax.plot(
        x[:-1][::-1],
        cumsum / max(cumsum),
        label=rf"$\Delta N(k_H) \geq$  {100*k:.0f}\%, $\#k_h$={cumsum[-1]}",
        marker="o",
    )


ax.legend()
ax.set_xlabel(r"\#flood events")
ax.set_ylabel(r"Percentage of hospitals")
ax.set_yscale("log")
ax.set_xlim(1, 30)

ax.set_yticklabels([f"{(x*100)}\%" for x in ax.get_yticks()])

fig.savefig("germanyFloods/figs/SI-multipleFloods.png", bbox_inches="tight", dpi=300)
fig.savefig("germanyFloods/figs/SI-multipleFloods.pdf", bbox_inches="tight", dpi=300)


# %%


gdf.tail(10)
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


affected_hospitals = gdf[gdf["population_percentual_diff"] > 0.5]

table = pd.DataFrame(
    {
        "Name": affected_hospitals["amenity_name"],
        "Count": affected_hospitals["amenity_name"],
        "Min Distance": affected_hospitals["min_dist"],
        "Population Difference": affected_hospitals["population_percentual_diff"],
        "Event": affected_hospitals["event"],
        "Catchment": affected_hospitals["catchment"],
    }
)

grouped_table = (
    table.groupby("Name")
    .agg(
        {
            "Count": "count",
            "Min Distance": "mean",
            "Population Difference": "mean",
            "Event": list,
            "Catchment": lambda x: x.unique()[0],
        }
    )
    .sort_values("Count", ascending=False)
    .reset_index()
)

grouped_table = grouped_table[grouped_table["Count"] > 1]


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
    (round(grouped_table["Min Distance"], 2))
    .astype(str)
    .str.rstrip("0")
    .str.rstrip(".")
)


def format_events(events):
    events_chunks = [
        ", ".join(map(str, events[i : i + 3])) for i in range(0, len(events), 3)
    ]
    return r"\makecell{" + r"\\ ".join(events_chunks) + "}"


grouped_table["Event"] = grouped_table["Event"].apply(format_events)

grouped_table["Catchment"] = grouped_table["Catchment"].map(catchemnt_dict)
# Convert the DataFrame to a LaTeX table
latex_table = grouped_table.to_latex(
    index=False,
    column_format="lrrrll",
    header=[
        "Name",
        "Count",
        "Min Distance",
        "Population Difference",
        "Event",
        "Catchment",
    ],
    escape=False,
)

# Print the LaTeX table
sideways_table = (
    "\\begin{sidewaystable}\n\\centering\n" + latex_table + "\\end{sidewaystable}"
)

# Print the LaTeX table
print(latex_table)

# %%
grouped_table
# %%
