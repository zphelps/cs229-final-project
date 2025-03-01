import pandas as pd

def getPreviousRace(raceId, racesCSV="data/races.csv"):
    races = pd.read_csv(racesCSV)
    
    # Get round and year for given raceId
    round = races.loc[races['raceId'] == raceId, 'round']
    year = races.loc[races['raceId'] == raceId, 'year']
    
    # Check if round or year exists
    if round.empty or year.empty:
        return None
    
    round = round.values[0]
    year = year.values[0]

    # If first round, return 0
    if round == 1:
        return 0

    # Get previous raceId
    prevRaceId = races.loc[((races['round'] == (round - 1)) & 
                            (races['year'] == year)), 
                           'raceId']
    
    if prevRaceId.empty:
        return 0  # No previous race found
    
    prevRaceId = prevRaceId.values[0]

    return prevRaceId

def getDriverStandingPoints(raceId, driverId, driverStandingsCSV="data/driver_standings.csv"):
    driverStandings = pd.read_csv(driverStandingsCSV)
    prevRaceId = getPreviousRace(raceId)

    # Get driver points from the previous race
    driverPoints = driverStandings.loc[((driverStandings['driverId'] == driverId) & 
                                        (driverStandings['raceId'] == prevRaceId)),
                                       'points']
    
    return driverPoints.values[0] if not driverPoints.empty else 0

def getConstructorStandingPoints(raceId, constructorId, constructorStandingsCSV="data/constructor_standings.csv"):
    constructorStandings = pd.read_csv(constructorStandingsCSV)
    prevRaceId = getPreviousRace(raceId)

    # Get driver points from the previous race
    constructorPoints = constructorStandings.loc[((constructorStandings['constructorId'] == constructorId) & 
                                        (constructorStandings['raceId'] == prevRaceId)),
                                       'points']
    
    return constructorPoints.values[0] if not constructorPoints.empty else 0

