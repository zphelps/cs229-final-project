import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional, Set
import os


class F1DatasetGenerator:
    """
    A class to generate Formula 1 datasets with engineered features for machine learning.
    """
    
    def __init__(self, data_dir: str = "data", cleaned_dir: str = "cleaned"):
        """
        Initialize the dataset generator.
        
        Args:
            data_dir: Directory containing the raw F1 data files
            cleaned_dir: Directory for storing cleaned/processed data
        """
        self.data_dir = data_dir
        self.cleaned_dir = cleaned_dir
        
        # Create cleaned directory if it doesn't exist
        os.makedirs(cleaned_dir, exist_ok=True)
        
        # Load the base datasets
        self.races = pd.read_csv(f"{data_dir}/races.csv")
        self.results = pd.read_csv(f"{data_dir}/results.csv")
        self.drivers = pd.read_csv(f"{data_dir}/drivers.csv")
        self.driver_standings = pd.read_csv(f"{data_dir}/driver_standings.csv")
        self.constructor_standings = pd.read_csv(f"{data_dir}/constructor_standings.csv")
        self.constructors = pd.read_csv(f"{data_dir}/constructors.csv")
        self.circuits = pd.read_csv(f"{data_dir}/circuits.csv")
        
        # Process position data - convert string positions to numeric
        self.results['position'] = pd.to_numeric(self.results['position'], errors='coerce')
        
        # Create a merged base dataset with essential information
        self._create_base_dataset()
    
    def _create_base_dataset(self):
        """
        Create a base dataset with essential race and driver information.
        """
        # Merge results with races to get year and circuit information
        self.base_df = self.results.merge(self.races[['raceId', 'year', 'round', 'circuitId']], 
                                         on='raceId', how='inner')
        
        # Add driver information
        self.base_df = self.base_df.merge(self.drivers[['driverId', 'forename', 'surname', 'dob', 'nationality']], 
                                         on='driverId', how='inner')
        
        # Add constructor information
        self.base_df = self.base_df.merge(self.constructors[['constructorId', 'name']], 
                                         on='constructorId', how='inner')
        
        # Rename columns for clarity
        self.base_df = self.base_df.rename(columns={'name': 'constructor_name'})
        
        # Create a driver full name column
        self.base_df['driver_name'] = self.base_df['forename'] + ' ' + self.base_df['surname']
        
        # Calculate driver age at race time
        self.base_df['dob'] = pd.to_datetime(self.base_df['dob'])
        self.base_df['date'] = pd.to_datetime(self.races['date'])
        self.base_df['driver_age'] = (self.base_df['date'] - self.base_df['dob']).dt.days / 365.25
        
        # Define podium as position 1, 2, or 3
        self.base_df['podium'] = self.base_df['position'].apply(lambda x: 1 if x in [1, 2, 3] else 0)
        
        # Sort by year, round, and position for chronological ordering
        self.base_df = self.base_df.sort_values(['year', 'round', 'position'])
    
    def _add_career_stats_prior_to_season(self):
        """
        Add career statistics prior to the current season:
        - Total career wins
        - Total career podiums
        - Total career points
        - Number of seasons in F1
        - Championship titles won
        """
        # Create a temporary dataframe to calculate career stats
        career_stats = []
        
        # Group by driver and year to calculate stats
        for (driver_id, year), group in self.base_df.groupby(['driverId', 'year']):
            # Get all results for this driver before this year
            prior_results = self.base_df[(self.base_df['driverId'] == driver_id) & 
                                         (self.base_df['year'] < year)]
            
            if prior_results.empty:
                # No prior results, set all stats to 0
                career_stats.append({
                    'driverId': driver_id,
                    'year': year,
                    'career_wins': 0,
                    'career_podiums': 0,
                    'career_points': 0,
                    'career_seasons': 0,
                    'career_titles': 0
                })
            else:
                # Calculate career stats
                career_wins = len(prior_results[prior_results['position'] == 1])
                career_podiums = len(prior_results[prior_results['position'].isin([1, 2, 3])])
                career_points = prior_results['points'].sum()
                career_seasons = len(prior_results['year'].unique())
                
                # Calculate championship titles
                # Get the final standings for each year
                titles = 0
                for prev_year in prior_results['year'].unique():
                    # Get the final race of the season
                    final_race = self.races[self.races['year'] == prev_year]['raceId'].max()
                    
                    # Check if driver was champion (position 1 in final standings)
                    champion = self.driver_standings[
                        (self.driver_standings['raceId'] == final_race) & 
                        (self.driver_standings['driverId'] == driver_id) & 
                        (self.driver_standings['position'] == 1)
                    ]
                    
                    if not champion.empty:
                        titles += 1
                
                career_stats.append({
                    'driverId': driver_id,
                    'year': year,
                    'career_wins': career_wins,
                    'career_podiums': career_podiums,
                    'career_points': career_points,
                    'career_seasons': career_seasons,
                    'career_titles': titles
                })
        
        # Convert to DataFrame and merge with base_df
        career_stats_df = pd.DataFrame(career_stats)
        self.base_df = self.base_df.merge(career_stats_df, on=['driverId', 'year'], how='left')
    
    def _add_prior_season_performance(self):
        """
        Add prior season performance metrics:
        - Points in previous season
        - Wins in previous season
        - Final championship position in previous season
        - Points per race average in previous season
        """
        prior_season_stats = []
        
        # Group by driver and year
        for (driver_id, year), group in self.base_df.groupby(['driverId', 'year']):
            # Get results from previous season
            prev_year = year - 1
            prev_season_results = self.base_df[(self.base_df['driverId'] == driver_id) & 
                                              (self.base_df['year'] == prev_year)]
            
            if prev_season_results.empty:
                # No previous season data
                prior_season_stats.append({
                    'driverId': driver_id,
                    'year': year,
                    'prev_season_points': 0,
                    'prev_season_wins': 0,
                    'prev_season_position': None,
                    'prev_season_points_per_race': 0
                })
            else:
                # Calculate previous season stats
                prev_season_points = prev_season_results['points'].sum()
                prev_season_wins = len(prev_season_results[prev_season_results['position'] == 1])
                prev_season_races = len(prev_season_results)
                prev_season_points_per_race = prev_season_points / prev_season_races if prev_season_races > 0 else 0
                
                # Get final championship position
                final_race = self.races[self.races['year'] == prev_year]['raceId'].max()
                final_standing = self.driver_standings[
                    (self.driver_standings['raceId'] == final_race) & 
                    (self.driver_standings['driverId'] == driver_id)
                ]
                
                prev_season_position = final_standing['position'].values[0] if not final_standing.empty else None
                
                prior_season_stats.append({
                    'driverId': driver_id,
                    'year': year,
                    'prev_season_points': prev_season_points,
                    'prev_season_wins': prev_season_wins,
                    'prev_season_position': prev_season_position,
                    'prev_season_points_per_race': prev_season_points_per_race
                })
        
        # Convert to DataFrame and merge with base_df
        prior_season_df = pd.DataFrame(prior_season_stats)
        self.base_df = self.base_df.merge(prior_season_df, on=['driverId', 'year'], how='left')
    
    def _add_track_specific_performance(self):
        """
        Add track-specific historical performance:
        - Driver's average finish position at this circuit
        - Driver's best result at this circuit
        - Number of previous races at this circuit
        """
        track_stats = []
        
        # Sort by year and round to ensure chronological order
        sorted_df = self.base_df.sort_values(['year', 'round'])
        
        # Process each race
        for idx, race in sorted_df.iterrows():
            driver_id = race['driverId']
            circuit_id = race['circuitId']
            race_id = race['raceId']
            
            # Get all previous races at this circuit for this driver
            prev_circuit_races = sorted_df[
                (sorted_df['driverId'] == driver_id) & 
                (sorted_df['circuitId'] == circuit_id) & 
                (sorted_df['raceId'] < race_id)
            ]
            
            if prev_circuit_races.empty:
                # No previous races at this circuit
                track_stats.append({
                    'raceId': race_id,
                    'driverId': driver_id,
                    'circuit_avg_position': None,
                    'circuit_best_position': None,
                    'circuit_races_count': 0
                })
            else:
                # Calculate track-specific stats
                positions = prev_circuit_races['position'].dropna()
                
                if positions.empty:
                    avg_pos = None
                    best_pos = None
                else:
                    avg_pos = positions.mean()
                    best_pos = positions.min()
                
                track_stats.append({
                    'raceId': race_id,
                    'driverId': driver_id,
                    'circuit_avg_position': avg_pos,
                    'circuit_best_position': best_pos,
                    'circuit_races_count': len(prev_circuit_races)
                })
        
        # Convert to DataFrame and merge with base_df
        track_stats_df = pd.DataFrame(track_stats)
        self.base_df = self.base_df.merge(track_stats_df, on=['raceId', 'driverId'], how='left')
    
    def _add_temporal_performance_trends(self, n_races: int = 5):
        """
        Add temporal performance trends:
        - Points trend over last N races
        - Form indicator (average position in last N races)
        
        Args:
            n_races: Number of previous races to consider for trends
        """
        trend_stats = []
        
        # Sort by year, round to ensure chronological order
        sorted_df = self.base_df.sort_values(['year', 'round'])
        
        # Process each race
        for idx, race in sorted_df.iterrows():
            driver_id = race['driverId']
            race_id = race['raceId']
            
            # Get the N most recent races for this driver before the current race
            prev_races = sorted_df[
                (sorted_df['driverId'] == driver_id) & 
                (sorted_df['raceId'] < race_id)
            ].sort_values('raceId', ascending=False).head(n_races)
            
            if prev_races.empty:
                # No previous races
                trend_stats.append({
                    'raceId': race_id,
                    'driverId': driver_id,
                    f'last_{n_races}_races_points': 0,
                    f'last_{n_races}_races_avg_position': None,
                    f'last_{n_races}_races_count': 0
                })
            else:
                # Calculate trend stats
                recent_points = prev_races['points'].sum()
                
                positions = prev_races['position'].dropna()
                avg_position = positions.mean() if not positions.empty else None
                
                trend_stats.append({
                    'raceId': race_id,
                    'driverId': driver_id,
                    f'last_{n_races}_races_points': recent_points,
                    f'last_{n_races}_races_avg_position': avg_position,
                    f'last_{n_races}_races_count': len(prev_races)
                })
        
        # Convert to DataFrame and merge with base_df
        trend_stats_df = pd.DataFrame(trend_stats)
        self.base_df = self.base_df.merge(trend_stats_df, on=['raceId', 'driverId'], how='left')
    
    def generate_dataset(self, 
                         include_career_stats: bool = True,
                         include_prior_season: bool = True, 
                         include_track_stats: bool = True,
                         include_temporal_trends: bool = True,
                         n_races_trend: int = 5,
                         output_file: Optional[str] = None,
                         selected_features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Generate a dataset with the requested features.
        
        Args:
            include_career_stats: Whether to include career statistics
            include_prior_season: Whether to include prior season performance
            include_track_stats: Whether to include track-specific performance
            include_temporal_trends: Whether to include temporal performance trends
            n_races_trend: Number of races to consider for temporal trends
            output_file: Path to save the generated dataset (optional)
            selected_features: List of specific features to include (if None, include all)
            
        Returns:
            DataFrame with the generated dataset
        """
        # Add requested feature groups
        if include_career_stats:
            self._add_career_stats_prior_to_season()
        
        if include_prior_season:
            self._add_prior_season_performance()
        
        if include_track_stats:
            self._add_track_specific_performance()
        
        if include_temporal_trends:
            self._add_temporal_performance_trends(n_races=n_races_trend)
        
        # Select features for the final dataset
        if selected_features is None:
            # Default set of features
            selected_features = [
                'raceId', 'year', 'round', 'circuitId', 'driverId', 'constructorId',
                'grid', 'position', 'points', 'driver_name', 'constructor_name', 'podium'
            ]
            
            # Add career stats features if included
            if include_career_stats:
                selected_features.extend([
                    'career_wins', 'career_podiums', 'career_points', 
                    'career_seasons', 'career_titles'
                ])
            
            # Add prior season features if included
            if include_prior_season:
                selected_features.extend([
                    'prev_season_points', 'prev_season_wins', 
                    'prev_season_position', 'prev_season_points_per_race'
                ])
            
            # Add track-specific features if included
            if include_track_stats:
                selected_features.extend([
                    'circuit_avg_position', 'circuit_best_position', 'circuit_races_count'
                ])
            
            # Add temporal trend features if included
            if include_temporal_trends:
                selected_features.extend([
                    f'last_{n_races_trend}_races_points',
                    f'last_{n_races_trend}_races_avg_position',
                    f'last_{n_races_trend}_races_count'
                ])
        
        # Filter columns that exist in the dataframe
        existing_features = [f for f in selected_features if f in self.base_df.columns]
        final_df = self.base_df[existing_features].copy()
        
        # Save to file if requested
        if output_file:
            output_path = f"{self.cleaned_dir}/{output_file}"
            final_df.to_csv(output_path, index=False)
            print(f"Dataset saved to {output_path}")
        
        return final_df
    
    def add_custom_feature(self, feature_name: str, feature_function, *args, **kwargs):
        """
        Add a custom feature to the dataset using a provided function.
        
        Args:
            feature_name: Name of the new feature
            feature_function: Function that calculates the feature values
            *args, **kwargs: Additional arguments to pass to the feature function
        """
        self.base_df[feature_name] = feature_function(self.base_df, *args, **kwargs)
        return self


# Example usage
if __name__ == "__main__":
    # Initialize the dataset generator
    generator = F1DatasetGenerator()
    
    # Generate a dataset with all features
    dataset = generator.generate_dataset(
        include_career_stats=True,
        include_prior_season=True,
        include_track_stats=True,
        include_temporal_trends=True,
        n_races_trend=5,
        output_file="f1_podium_prediction_dataset.csv"
    )
    
    # # Example of generating a dataset with specific features
    # custom_features = [
    #     'raceId', 'year', 'round', 'driverId', 'constructorId', 
    #     'grid', 'position', 'podium', 'career_wins', 'career_podiums',
    #     'prev_season_points', 'circuit_best_position'
    # ]
    
    # custom_dataset = generator.generate_dataset(
    #     selected_features=custom_features,
    #     output_file="f1_custom_features_dataset.csv"
    # )
    
    # # Example of adding a custom feature
    # def calculate_grid_to_position_delta(df):
    #     """Calculate the difference between grid position and final position"""
    #     return df['grid'] - df['position']
    
    # # Add the custom feature and generate a new dataset
    # generator.add_custom_feature('grid_position_delta', calculate_grid_to_position_delta)
    
    # enhanced_dataset = generator.generate_dataset(
    #     output_file="f1_enhanced_dataset.csv"
    # )
    
    print("Dataset generation complete!")
