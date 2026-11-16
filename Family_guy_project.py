import pandas as pd
#import csv file:

df = pd.read_csv("Family-Guy-Project/Family Guy Dataset.csv")
print(df.head())

#investigate data:
print(df.columns)

#data cleaning
# Data cleaning
# Data cleaning
def tweak_df(df):
    return (
        df
        .rename(columns={
            "Written by": "writer",
            "Title of the Episode": "title"
        })
        .assign(
            air_date=lambda x: pd.to_datetime(
                x["Original Air Date"].str.extract(
                    r"([A-Za-z]+ \d{1,2}, \d{4})"
                )[0],
                format="%B %d, %Y",
                errors="coerce"
            )
        )
    )

# Apply the cleaning function
df = tweak_df(df)

# Check the results
print(df[["title", "Original Air Date", "air_date"]].head())
print("Missing dates:", df["air_date"].isna().sum())


# Research Question 3: Cloud coverage and IMDb ratings
import requests
url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": 41.824,
    "longitude": -71.412,
    "start_date": df["air_date"].min().strftime("%Y-%m-%d"),
"end_date": df["air_date"].max().strftime("%Y-%m-%d"),
    "daily": "cloud_cover_mean",
    "timezone": "America/New_York"
}

response = requests.get(url, params=params, timeout=60)
response.raise_for_status()

data = response.json()

weather = pd.DataFrame({
    "time": data["daily"]["time"],
    "cloud_cover_mean": data["daily"]["cloud_cover_mean"]
})

print(weather.head())
print("Number of weather days:", len(weather))

# Combine Family Guy episodes with weather data

# Convert weather dates to datetime
weather["time"] = pd.to_datetime(weather["time"])

# Match episodes to weather using their dates
df = df.merge(
    weather,
    left_on="air_date",
    right_on="time",
    how="left"
)

# Check the result
print(df[["title", "air_date", "IMDb Rating", "cloud_cover_mean"]].head())


#visualization:
#Is family guy getting worse over time?
import matplotlib.pyplot as plt
average_ratings = df.groupby("Season")["IMDb Rating"].mean()
average_ratings.plot(kind="line")

plt.title("Family Guy IMDb Ratings by Season")
plt.xlabel("Season")
plt.ylabel("Average IMDb Rating")
plt.xticks(average_ratings.index)
plt.grid(True)
plt.show()

#Has family guy lost viewers?
average_viewers = df.groupby("Season")["U.S. Viewers (Millions)"].mean()
average_viewers.plot(kind="line", marker="o")

plt.title("Family Guy: Average US Viewers by Season")
plt.xlabel("Season")
plt.ylabel("Average Viewers (Millions)")
plt.xticks(average_viewers.index)
plt.grid(True)
plt.show()

# Research Question 3: Cloud coverage, IMDb ratings and viewers

plt.figure(figsize=(10, 6))

plt.scatter(
    df["cloud_cover_mean"],
    df["IMDb Rating"],
    s=df["U.S. Viewers (Millions)"] * 20,
    color="purple",
    alpha=0.5
)

plt.title("Family Guy: Cloud Coverage, Ratings and Viewers")
plt.xlabel("Cloud Coverage (%)")
plt.ylabel("IMDb Rating")

plt.grid(True)
plt.show()

# Calculate correlation between cloud cover and IMDb rating

correlation = df["cloud_cover_mean"].corr(df["IMDb Rating"])

print("Correlation between cloud cover and IMDb rating:")
print(round(correlation, 3))

# Research Question 4:
# Is there a relationship between US viewers and IMDb ratings?

plt.figure(figsize=(10, 6))

plt.scatter(
    df["U.S. Viewers (Millions)"],
    df["IMDb Rating"],
    color="teal",
    alpha=0.5
)

plt.title("Family Guy: US Viewers vs IMDb Ratings")
plt.xlabel("US Viewers (Millions)")
plt.ylabel("IMDb Rating")

plt.grid(True)
plt.show()

correlation_viewers = df["U.S. Viewers (Millions)"].corr(
    df["IMDb Rating"]
)

print("Correlation between viewers and IMDb ratings:")
print(round(correlation_viewers, 3))

#for season
# Research Question 4b:
# Does season help explain the relationship?

plt.figure(figsize=(10, 6))

plt.scatter(
    df["U.S. Viewers (Millions)"],
    df["IMDb Rating"],
    c=df["Season"],
    cmap="viridis",
    alpha=0.7
)

plt.colorbar(label="Season")

plt.title("Family Guy: Viewers and IMDb Ratings by Season")
plt.xlabel("US Viewers (Millions)")
plt.ylabel("IMDb Rating")

plt.grid(True)
plt.show()

# Research Question 5:
# Which writers have written the most Family Guy episodes?

writer_stats = (
    df.groupby("writer")["IMDb Rating"]
    .agg(["count", "mean"])
    .rename(columns={
        "count": "episodes",
        "mean": "average_rating"
    })
)

# Sort writers by number of episodes
writer_stats = writer_stats.sort_values(
    "episodes",
    ascending=False
)

# Show the top 10
print(writer_stats.head(10))

# Research Question 5:
# Which writers have the highest ratings?

# Select writers with at least 5 episodes
writers_plot = writer_stats[
    writer_stats["episodes"] >= 5
].copy()

# Sort writers by average rating
writers_plot = writers_plot.sort_values(
    "average_rating",
    ascending=False
)

# Keep the top 15 writers
writers_plot = writers_plot.head(15)

# Create bubble chart
plt.figure(figsize=(14, 7))

plt.scatter(
    writers_plot.index,
    writers_plot["average_rating"],
    s=writers_plot["episodes"] * 30,
    color="purple",
    alpha=0.6
)

plt.title("Family Guy Writers: Ratings and Episode Counts")
plt.xlabel("Writer")
plt.ylabel("Average IMDb Rating")

plt.xticks(rotation=75, ha="right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Who is Family Guy?
# Find the highest-rated writer with at least 10 episodes

qualified_writers = writer_stats[
    writer_stats["episodes"] >= 10
]

winner = qualified_writers.sort_values(
    "average_rating",
    ascending=False
).iloc[0]

winner_name = qualified_writers["average_rating"].idxmax()

print("Who is Family Guy?")
print("Highest-rated writer:", winner_name)
print("Average IMDb rating:", round(winner["average_rating"], 2))
print("Episodes written:", int(winner["episodes"]))
# Research Question 6:
# Do Family Guy writers differ in their IMDb rating distributions?

import seaborn as sns

# Find the 10 writers with the most rated episodes
top_10_writers = writer_stats.head(10).index

# Select episodes credited to those writers
writer_data = df[df["writer"].isin(top_10_writers)]

# Create the box plot
plt.figure(figsize=(12, 7))

sns.boxplot(
    data=writer_data,
    x="IMDb Rating",
    y="writer",
    order=top_10_writers,
    hue="writer",
    palette="Set2",
    legend=False
)

plt.title("Who Is Family Guy? IMDb Ratings by Writer")
plt.xlabel("IMDb Rating")
plt.ylabel("Writer")

plt.tight_layout()
plt.show()
