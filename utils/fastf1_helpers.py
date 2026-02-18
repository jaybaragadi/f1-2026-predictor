"""
FastF1 Helpers - LAP-BASED DATA EXTRACTION
Uses lap data instead of deprecated Ergast results
"""

import fastf1
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Enable FastF1 cache
fastf1.Cache.enable_cache('cache')


def get_race_results_from_laps(year, race_name):
    """
    Extract race results from lap data
    This is the workaround for empty results.Position
    
    Parameters:
    -----------
    year : int
        Season year
    race_name : str or int
        Race name or round number
    
    Returns:
    --------
    pd.DataFrame : Race results with finishing positions
    """
    try:
        # Load race session
        race = fastf1.get_session(year, race_name, 'R')
        race.load(telemetry=False, weather=False, messages=False)  # Faster loading
        
        # Get laps data
        laps = race.laps
        
        if laps is None or len(laps) == 0:
            print(f"  No lap data for {year} - {race_name}")
            return None
        
        # Get the finishing order from the last lap of each driver
        # Group by driver and get their last lap
        last_laps = laps.groupby('Driver').last().reset_index()
        
        # Get starting grid from first lap or qualifying
        first_laps = laps.groupby('Driver').first().reset_index()
        
        # Try to get results object for Points (even though Position is empty)
        try:
            results = race.results
            points_map = dict(zip(results['Abbreviation'], results['Points']))
        except:
            points_map = {}
        
        # Create race data DataFrame
        race_data = pd.DataFrame({
            'Year': year,
            'RaceName': race.event['EventName'],
            'Round': race.event['RoundNumber'],
            'DriverCode': last_laps['Driver'],
            'Team': last_laps['Team'],
        })
        
        # Get FullName from results if available
        if hasattr(race, 'results') and race.results is not None:
            driver_names = dict(zip(race.results['Abbreviation'], race.results['FullName']))
            race_data['DriverName'] = race_data['DriverCode'].map(driver_names)
        else:
            race_data['DriverName'] = race_data['DriverCode']
        
        # Get driver numbers
        driver_numbers = dict(zip(last_laps['Driver'], last_laps['DriverNumber']))
        race_data['DriverNumber'] = race_data['DriverCode'].map(driver_numbers).astype(str)
        
        # Calculate finishing position from LapNumber (drivers who completed more laps finish higher)
        # Then sort by total race time for drivers with same lap count
        race_data['LapsCompleted'] = last_laps['LapNumber'].values
        race_data['TotalTime'] = last_laps['Time'].values
        
        # Sort by laps completed (descending), then by total time (ascending)
        race_data = race_data.sort_values(
            ['LapsCompleted', 'TotalTime'], 
            ascending=[False, True]
        ).reset_index(drop=True)
        
        # Assign finishing positions
        race_data['Position'] = range(1, len(race_data) + 1)
        
        # Get starting grid position from first lap
        grid_map = dict(zip(first_laps['Driver'], first_laps['Position']))
        race_data['GridPosition'] = race_data['DriverCode'].map(grid_map)
        
        # Add points (if available from results)
        race_data['Points'] = race_data['DriverCode'].map(points_map).fillna(0)
        
        # Calculate points if not available (standard F1 points system)
        if race_data['Points'].sum() == 0:
            points_system = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 
                           6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
            race_data['Points'] = race_data['Position'].map(points_system).fillna(0)
        
        # Add status (assume Finished for now)
        race_data['Status'] = 'Finished'
        
        # Select final columns
        race_data = race_data[[
            'Year', 'RaceName', 'Round', 'DriverNumber', 'DriverCode', 
            'DriverName', 'Team', 'GridPosition', 'Position', 'Points', 'Status'
        ]]
        
        # Clean up
        race_data['Position'] = race_data['Position'].astype(int)
        race_data['GridPosition'] = pd.to_numeric(race_data['GridPosition'], errors='coerce')
        
        return race_data
        
    except Exception as e:
        print(f"  Error fetching {year} - {race_name}: {str(e)}")
        return None


def get_qualifying_results(year, race_name):
    """
    Fetch qualifying results
    
    Parameters:
    -----------
    year : int
        Season year
    race_name : str or int
        Race name or round number
    
    Returns:
    --------
    pd.DataFrame : Qualifying results
    """
    try:
        # Load qualifying session
        quali = fastf1.get_session(year, race_name, 'Q')
        quali.load(telemetry=False, weather=False, messages=False)
        
        # Get results
        results = quali.results
        
        if results is None or len(results) == 0:
            # Try from laps as fallback
            laps = quali.laps
            if laps is None or len(laps) == 0:
                return None
            
            # Get best lap per driver
            best_laps = laps.loc[laps.groupby('Driver')['LapTime'].idxmin()].copy()
            best_laps = best_laps.sort_values('LapTime').reset_index(drop=True)
            
            quali_data = pd.DataFrame({
                'Year': year,
                'RaceName': quali.event['EventName'],
                'Round': quali.event['RoundNumber'],
                'DriverNumber': best_laps['DriverNumber'].astype(str),
                'DriverCode': best_laps['Driver'],
                'QualifyingPosition': range(1, len(best_laps) + 1),
                'Q1': best_laps['LapTime'].astype(str),
                'Q2': None,
                'Q3': None,
            })
        else:
            # Use results if available
            quali_data = pd.DataFrame({
                'Year': year,
                'RaceName': quali.event['EventName'],
                'Round': quali.event['RoundNumber'],
                'DriverNumber': results.index.astype(str),
                'DriverCode': results['Abbreviation'],
                'QualifyingPosition': results['Position'],
                'Q1': results['Q1'].astype(str),
                'Q2': results['Q2'].astype(str),
                'Q3': results['Q3'].astype(str),
            })
        
        return quali_data
        
    except Exception as e:
        print(f"  Error fetching qualifying {year} - {race_name}: {str(e)}")
        return None


def get_season_schedule(year):
    """Get the race schedule for a season"""
    try:
        schedule = fastf1.get_event_schedule(year)
        return schedule
    except Exception as e:
        print(f"Error fetching schedule for {year}: {str(e)}")
        return None


def collect_full_season(year, delay=3):
    """
    Collect all race and qualifying data for a full season
    
    Parameters:
    -----------
    year : int
        Season year
    delay : int
        Delay between API calls (seconds)
    
    Returns:
    --------
    tuple : (race_data_df, quali_data_df)
    """
    print(f"\n{'='*60}")
    print(f"Collecting data for {year} season...")
    print(f"{'='*60}\n")
    
    # Get schedule
    schedule = get_season_schedule(year)
    
    if schedule is None:
        return None, None
    
    # Filter only Grand Prix events
    races = schedule[schedule['EventFormat'] != 'testing']
    
    all_race_data = []
    all_quali_data = []
    
    # Iterate through races
    for idx, race in tqdm(races.iterrows(), total=len(races), desc=f"{year} Season"):
        round_num = race['RoundNumber']
        event_name = race['EventName']
        
        print(f"  Collecting {event_name}...")
        
        # Get race results from laps
        race_results = get_race_results_from_laps(year, round_num)
        if race_results is not None and len(race_results) > 0:
            all_race_data.append(race_results)
            print(f"    ✓ Race: {len(race_results)} drivers")
        else:
            print(f"    ⚠️  Race: No data")
        
        # Get qualifying results
        quali_results = get_qualifying_results(year, round_num)
        if quali_results is not None and len(quali_results) > 0:
            all_quali_data.append(quali_results)
            print(f"    ✓ Quali: {len(quali_results)} drivers")
        else:
            print(f"    ⚠️  Quali: No data")
        
        # Delay to avoid overwhelming the API
        time.sleep(delay)
    
    # Combine all data
    race_df = pd.concat(all_race_data, ignore_index=True) if all_race_data else pd.DataFrame()
    quali_df = pd.concat(all_quali_data, ignore_index=True) if all_quali_data else pd.DataFrame()
    
    print(f"\n✓ Collected {len(race_df)} race results and {len(quali_df)} qualifying results for {year}")
    
    # Show sample if we have data
    if len(race_df) > 0:
        print(f"\nSample data from {year}:")
        sample = race_df.iloc[0]
        print(f"  Driver: {sample['DriverName']}")
        print(f"  Position: {sample['Position']}")
        print(f"  Grid: {sample['GridPosition']}")
        print(f"  Points: {sample['Points']}")
    
    return race_df, quali_df


def get_driver_standings(year):
    """
    Get driver championship standings
    Calculated from race results
    """
    try:
        # We'll calculate this from race data instead
        # This function can be simplified or removed
        return pd.DataFrame()  # Placeholder
        
    except Exception as e:
        print(f"Error fetching standings for {year}: {str(e)}")
        return None


def get_winter_testing_data(year):
    """Get winter testing data (when available)"""
    try:
        schedule = get_season_schedule(year)
        testing = schedule[schedule['EventFormat'] == 'testing']
        
        if len(testing) == 0:
            print(f"No winter testing data available for {year}")
            return None
        
        # Simplified - return None for now
        return None
        
    except Exception as e:
        print(f"Winter testing data not available for {year}: {str(e)}")
        return None