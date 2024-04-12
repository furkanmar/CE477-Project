import pandas as pd

df = pd.read_csv("GameSales.csv")

#defining categorical variables
publisher = df["Publisher"]
genre = df["Genre"]
platform = df["Platform"]
name = df["Name"]

#mode
name_mode = name.mode().values[0]
platform_mode = platform.mode().values[0]
genre_mode = genre.mode().values[0]
publisher_mode = publisher.mode().values[0]

#sort and find median
name_median = name.sort_values().reset_index(drop=True)[len(name) // 2]
platform_median = platform.sort_values().reset_index(drop=True)[len(platform) // 2]
genre_median = genre.sort_values().reset_index(drop=True)[len(genre) // 2]
publisher_median = publisher.sort_values().reset_index(drop=True)[len(publisher) // 2]

#data frame
mode = {
    "Mode": [name_mode, platform_mode, genre_mode, publisher_mode],
    "Median": [name_median, platform_median, genre_median, publisher_median]
}
graph = pd.DataFrame(mode, index=["Name", "Platform", "Genre", "Publisher"])

#result
print(graph)
