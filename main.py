
import pandas as pd


def get_races(year: int):
    races = pd.read_csv("data/races.csv")
    return races[races["year"].isin([year])]


def main():
    results = pd.read_csv("data/results.csv")
    races = pd.read_csv("data/races.csv")

    
    print(results.head())
    print(races.head())

    results = results.merge(races, on="raceId", how="inner")

    results = results.query("year == 2024")
    print(results.head())


if __name__ == "__main__":
    main()