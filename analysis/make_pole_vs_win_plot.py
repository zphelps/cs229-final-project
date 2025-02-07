import pandas as pd
import matplotlib.pyplot as plt

def make_pole_vs_win_plot(results_csv="data/results.csv",
                          qualifying_csv="data/qualifying.csv",
                          races_csv="data/races.csv",
                          circuits_csv="data/circuits.csv"):
    """
    Reads F1 data from CSV files and creates a plot showing, for each circuit,
    how often the pole-sitter (driver in P1 in qualifying) goes on to win the race.
    """

    # 1) Load the data
    results = pd.read_csv(results_csv)
    qualifying = pd.read_csv(qualifying_csv)
    races = pd.read_csv(races_csv)
    circuits = pd.read_csv(circuits_csv)

    # 2) Identify the race winner from results (positionOrder=1).
    #    We'll keep only columns of interest: raceId, driverId, positionOrder
    #    and rename driverId to something like winnerDriverId for clarity.
    winners = (
        results.query("positionOrder == 1")
               .loc[:, ["raceId", "driverId"]]
               .rename(columns={"driverId": "winnerDriverId"})
    )

    # 3) Identify the pole-sitter from qualifying (position=1).
    #    We'll keep raceId, driverId, rename to poleDriverId for clarity.
    poles = (
        qualifying.query("position == 1")
                  .loc[:, ["raceId", "driverId"]]
                  .rename(columns={"driverId": "poleDriverId"})
    )

    # 4) Merge the winners and poles on raceId, so we know for each race:
    #    who got pole, who won.
    #    Then see if it's the same driver (poleDriverId == winnerDriverId).
    merged = pd.merge(winners, poles, on="raceId", how="inner")

    # 5) Merge in race info to get circuitId (so we can group by circuit).
    #    Then merge in circuit info to get circuit name or location.
    merged = pd.merge(merged, races.loc[:, ["raceId", "circuitId"]], 
                      on="raceId", how="inner")
    merged = pd.merge(merged, circuits.loc[:, ["circuitId", "name", "location"]], 
                      on="circuitId", how="inner")

    # 6) We create a column that is 1 if the pole-sitter also won, else 0.
    merged["pole_won"] = (merged["poleDriverId"] == merged["winnerDriverId"]).astype(int)

    # 7) Group by circuit name (or location), 
    #    then compute fraction or total times that the pole-sitter won.
    grouped = (
        merged.groupby("name")
              .agg(total_races=("raceId", "count"), 
                   pole_won_sum=("pole_won", "sum"))
              .reset_index()
    )
    grouped["pct_pole_won"] = grouped["pole_won_sum"] / grouped["total_races"] * 100

    # 8) Sort by descending fraction of times the pole-sitter also won
    grouped = grouped.sort_values("pct_pole_won", ascending=False)

    # 9) Plot a simple bar chart showing the fraction of times the pole-sitter also won
    plt.figure(figsize=(10, 6))
    plt.barh(grouped["name"], grouped["pct_pole_won"], color="skyblue")
    plt.xlabel("Percentage of Races Where Pole Sitter Won (%)")
    plt.ylabel("Circuit")
    plt.title("Correlation: Being on Pole and Winning the Race (by Circuit)")
    plt.gca().invert_yaxis()  # so that highest fraction is at top

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    make_pole_vs_win_plot() 