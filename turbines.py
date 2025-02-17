import os
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
"""
Assumptions
1. All 15 Wind turbines are located on the same wind farm
2. Sensors for each turbine measure metrics in the same way
3. Missing values are defined as nulls or are defined as a missed hourly timestamp between the start and end date
"""

# 1. Load the data
def load_data(csvs):
    """
    1. Loads data from the specified file paths.
    2. Reads each file and appends its contents to a single dataframe.
    3. Returns the combined dataframe.
    """
    df_list = []

    for path in csvs:
        try:
            df_temp = pd.read_csv(path)
            df_temp["file_name"] = os.path.basename(path)  # Track file source
            df_list.append(df_temp)
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            raise ValueError(f"Error loading '{path}': {str(e)}")
        except Exception as e:
            raise ValueError(f"Unexpected error loading '{path}': {str(e)}")

    if not df_list:
        raise ValueError("No valid files were loaded. Please check file formats.")

    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"Number of files loaded: {combined_df['file_name'].nunique()}")

    return combined_df


import pandas as pd

def fill_missing_values(df):
    """
    1. Logs the percentage of missing values for the wind_speed, wind_direction, and power_output columns.
    2. Ensures that for each turbine, all hourly timestamps between the minimum and maximum timestamp exist.
    3. Fills missing values for each turbine based on the mean values of the other turbines on the same timestamp.
    4. Adds a new column 'fill_missing_value' that tracks which rows were imputed (1 if originally missing or added).
    """
    # Ensures 'timestamp' is in datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Creates a new column for the hourly timestamp 
    df['timestamp_hour'] = df['timestamp'].dt.floor('h')
    
    # Generates a complete hourly time series for each turbine
    turbines = df['turbine_id'].unique()
    df_list = []
    
    for turbine in turbines:
        # Selects data for this turbine
        df_turb = df[df['turbine_id'] == turbine].copy()
        
        # Determines the complete hourly range for this turbine
        min_time = df_turb['timestamp_hour'].min()
        max_time = df_turb['timestamp_hour'].max()
        full_range = pd.date_range(start=min_time, end=max_time, freq='h')
        
        # Creates a DataFrame with the full hourly range for this turbine
        full_df = pd.DataFrame({'timestamp_hour': full_range})
        full_df['turbine_id'] = turbine
        
        # Merges the complete hourly timestamps with the turbine data
        # Uses a left join so that any missing timestamps become rows with NaN values
        df_merged = pd.merge(full_df, df_turb, on=['turbine_id', 'timestamp_hour'], how='left')
        
        # If the original 'timestamp' column is missing (because of a new row), fill it with the timestamp_hour value
        df_merged['timestamp'] = df_merged['timestamp'].combine_first(df_merged['timestamp_hour'])
        
        df_list.append(df_merged)
    
    # Concatenates the data for all turbines into one DataFrame
    df_full = pd.concat(df_list, ignore_index=True)
    
    # Columns for which sensor values should be filled
    columns_to_fill = ["wind_speed", "wind_direction", "power_output"]
    
    # Calculates and log the percentage of missing values for the specified columns (before imputation)
    missing_values = df_full[columns_to_fill].isnull().mean() * 100
    missing_values_df = pd.DataFrame({
        "column_name": columns_to_fill,
        "missing_percentage": missing_values.values
    })
    
    print("Missing Values Before Filling:")
    print(missing_values_df)
    
    # Creates a new column to indicate whether the row had any missing sensor value originally.
    # This includes rows that were added due to missing timestamps.
    df_full['fill_missing_value'] = df_full[columns_to_fill].isnull().any(axis=1).astype(int)
    
    # Fills missing sensor values using the mean of other turbines at the same hourly timestamp.
    # The groupby on 'timestamp_hour' ensures that each hourly timestamp is treated separately.
    for column in columns_to_fill:
        df_full[column] = df_full.groupby('timestamp_hour')[column].transform(lambda x: x.fillna(x.mean()))
    
    # Drops the temporary 'timestamp_hour' column to restore the original structure
    df_full.drop(columns=['timestamp_hour'], inplace=True)
    
    return df_full, missing_values_df




import pandas as pd

def deg_to_compass(deg):
    """
    1. Converts an integer degree value into a descriptive string (e.g., "north", "north east", etc.) based on predefined groupings.
    2. Aids visualisation for BI developers by providing a more intuitive understanding of wind direction.
    """
    # Defines 16 compass directions
    directions = [
        "north", "north north east", "north east", "east north east", 
        "east", "east south east", "south east", "south south east",
        "south", "south south west", "south west", "west south west",
        "west", "west north west", "north west", "north north west"
    ]
    if pd.isna(deg):
        return None
    # Each direction covers 22.5 degrees
    idx = int((deg + 11.25) / 22.5) % 16
    return directions[idx]


def detect_and_fill_outliers(df):
    """
    1. Targets outliers in the wind_speed column.
    2. Outliers in the wind_direction column are not addressed because wind direction values often show large, inconsistent fluctuations and are less likely to affect power output (as wind turbines can pivot).
    3. For each timestamp, outliers in wind speed are detected using the IQR method relative to the other turbines observed at that time.
    4. If a wind speed value is determined to be an outlier, it is flagged in a new column called detect_wind_speed_outlier.
    5. Two dataframes are ultimately returned: one with the cleaned, hourly view of the data, and another aggregate table that highlights power anomalies over a 24‑hour period.ghlights power anomalies over a 24‑hour period.
    """
    # Ensurs 'timestamp' is in datetime format using day-first format
    df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)
    
    # Columns to check for outliers (currently only checking 'wind_speed')
    columns_to_check = ["wind_speed"]
    
    # Initialises a column to flag wind_speed outliers with 0 [not an outlier]
    df['detect_wind_speed_outlier'] = 0
    
    # Track outlier counts
    outlier_counts = {col: 0 for col in columns_to_check}
    total_rows = len(df)
    
    # 1. Detects Outliers Per Timestamp Using IQR excluding the current row)
    for timestamp in df['timestamp'].unique():
        subset = df[df['timestamp'] == timestamp]
        for idx, row in subset.iterrows():
            # Creates a subset excluding the current row
            subset_excl = subset.drop(idx)
            # If there's no other rows to compare, skip this row.
            if subset_excl.empty:
                continue
            for column in columns_to_check:
                Q1 = subset_excl[column].quantile(0.25)
                Q3 = subset_excl[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Check if the current row's wind_speed is an outlier relative to the rest of the subset.
                if row[column] < lower_bound or row[column] > upper_bound:
                    # Updates outlier count.
                    outlier_counts[column] += 1
                    # Flags this row as an outlier.
                    df.loc[idx, 'detect_wind_speed_outlier'] = 1
                    # Replaces the outlier value with NaN.
                    df.loc[idx, column] = None
    
    # 2. Calculates Outlier Percentages
    outlier_percentages = {col: (count / total_rows) * 100 for col, count in outlier_counts.items()}
    
    # 3. Logs Outlier Counts & Percentages
    outlier_summary_df = pd.DataFrame({
        "column_name": list(outlier_counts.keys()),
        "outlier_count": list(outlier_counts.values()),
        "outlier_percentage": list(outlier_percentages.values())
    })
    
    print("Outliers Detected and Replaced with NaN:")
    print(outlier_summary_df)
    
    # Step 4: Fills Outliers Using Mean of Other Turbines at the Same Timestamp
    for column in columns_to_check:
        df[column] = df.groupby('timestamp')[column].transform(lambda x: x.fillna(x.mean()))
    
    # Step 5: Converts wind_direction degrees to compass direction strings.
    df['wind_direction_str'] = df['wind_direction'].apply(deg_to_compass)
    
    return df, outlier_summary_df




def detect_energy_anomalies_24h(df):
    """
    1. Groups the data by wind turbine and date.
    2. Aggregates the power output for each turbine over the entire day.
    3. Compares each turbine’s daily power output to the mean power output of the other turbines for that day.
    4. Flags an anomaly if the deviation exceeds two standard deviations.
    5. Outputs the original cleaned dataframe and the anomaly summary table
    """
    # Creates a copy of the original dataframe to return unchanged
    original_df = df.copy()
    
    # Ensure proper datetime conversion.
    df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)
    # s a date column representing the 24-hour period.
    df['date'] = df['timestamp'].dt.date
    
    # Creates a daily summary by turbine:
    # Total power_output over the day (sum)
    # Average wind_speed over the day (mean)
    # Average wind_direction over the day (mean)
    daily_summary = df.groupby(['date', 'turbine_id']).agg({
        'power_output': 'sum',
        'wind_speed': 'mean',
        'wind_direction': 'mean'
    }).reset_index()
    
    # Defines a function to flag anomalies for each day based on the daily total power_output
    def flag_anomaly(subgroup):
        anomaly_flags = []
        for _, row in subgroup.iterrows():
            # Exclude the current turbine to compute the baseline.
            baseline = subgroup[subgroup['turbine_id'] != row['turbine_id']]
            # If no other turbines or no variation in baseline, flag anomaly as 0
            if baseline.empty or baseline['power_output'].std() == 0:
                anomaly_flags.append(0)
            else:
                mean_other = baseline['power_output'].mean()
                std_other = baseline['power_output'].std()
                # Flags anomaly if deviation exceeds 2 standard deviations
                if abs(row['power_output'] - mean_other) > 2 * std_other:
                    anomaly_flags.append(1)
                else:
                    anomaly_flags.append(0)
        subgroup = subgroup.copy()
        subgroup['power_anomoly_detected'] = anomaly_flags
        return subgroup
    
    # Apply the anomaly flag function for each date
    daily_summary = daily_summary.groupby('date', group_keys=False).apply(flag_anomaly)
    
    # Add a new column with the wind direction string based on the average wind_direction.
    daily_summary['wind_direction_str'] = daily_summary['wind_direction'].apply(deg_to_compass)
    
    # Reorder columns as required.
    daily_summary = daily_summary[['date', 'turbine_id', 'wind_speed', 'wind_direction', 'wind_direction_str',
                                   'power_output', 'power_anomoly_detected']]
    
    return original_df, daily_summary




def process_turbine_data(csv_paths):
    """
    1. Main function that orchestrates loading, cleaning, and analysing data from turbine CSVs.
    2. After processing, it outputs CSV files for the cleaned data and the daily anomaly summary.
    """
    # 1. Loads raw data
    raw_df = load_data(csv_paths)

    # 2. Fills missing values
    df_filled, missing_summary = fill_missing_values(raw_df)

    # 3. Detects and fill outliers
    df_cleaned, outlier_summary = detect_and_fill_outliers(df_filled)

    # 4. Detects anomalies in power output over a 24-hour rolling period
    original_df, daily_summary = detect_energy_anomalies_24h(df_cleaned)

    # 5. Prints summaries to console
    print("Missing Value Summary:")
    print(missing_summary)

    print("\nOutlier Summary:")
    print(outlier_summary)

    print("\nEnergy Anomaly Summary:")
    print(daily_summary)

    # 6. Creates output CSV files
    original_df.to_csv("cleaned.csv", index=False)
    daily_summary.to_csv("daily_summary.csv", index=False)
    
    print("CSV output files 'cleaned.csv' and 'daily_summary.csv' created successfully.")

    return raw_df, df_filled, missing_summary, df_cleaned, outlier_summary, original_df, daily_summary


if __name__ == "__main__":
   
    csv_files = [
        "data_group_1.csv",
        "data_group_2.csv",
        "data_group_3.csv"
    ]

    raw_df, df_filled, missing_summary, df_cleaned, outlier_summary, original_df, daily_summary = process_turbine_data(csv_files)
