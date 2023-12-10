import pandas as pd
import numpy as np
import datetime

df = pd.read_csv(r'c:\Users\Yash\Downloads\dataset-3.csv')

def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Create a pivot table with 'id_start' as index and 'id_end' as columns, and 'distance' as values
    distance_pivot = df.pivot(index='id_start', columns='id_end', values='distance').fillna(np.inf)
    
    # Fill diagonal with 0s for the distance from each ID to itself
    np.fill_diagonal(distance_pivot.values, 0)
    
    # Ensure the pivot table is square by adding any missing columns or rows
    all_ids = np.union1d(distance_pivot.index, distance_pivot.columns)
    distance_pivot = distance_pivot.reindex(index=all_ids, columns=all_ids, fill_value=np.inf)
    
    # Use the Floyd-Warshall algorithm to find all-pairs shortest paths
    for k in all_ids:
        for i in all_ids:
            for j in all_ids:
                if distance_pivot.at[i, j] > distance_pivot.at[i, k] + distance_pivot.at[k, j]:
                    distance_pivot.at[i, j] = distance_pivot.at[i, k] + distance_pivot.at[k, j]
    
    # Replace np.inf with 0 for non-connected pairs
    distance_pivot.replace(np.inf, 0, inplace=True)
    
    # Ensure the matrix is symmetric
    df = distance_pivot + distance_pivot.T - np.diag(np.diag(distance_pivot))
    
    return df

df1 = calculate_distance_matrix(df)

def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Initialize a list to store the unrolled data
    unrolled_data = []
    
    # Iterate over the matrix to unroll it into a list of dictionaries
    for i in df.index:
        for j in df.columns:
            if i != j:  # Exclude same id_start to id_end
                unrolled_data.append({'id_start': i, 'id_end': j, 'distance': df.at[i, j]})
    
    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(unrolled_data)
    
    return df

df2 = unroll_distance_matrix(df1)

def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Calculate the average distance of the reference ID
    reference_avg_distance = df[df['id_start'] == reference_id]['distance'].mean()
    
    # Define the upper and lower bounds (10% threshold)
    lower_bound = reference_avg_distance * 0.9
    upper_bound = reference_avg_distance * 1.1
    
    # Find IDs within the 10% threshold
    result_df = df.groupby('id_start').filter(lambda x: lower_bound <= x['distance'].mean() <= upper_bound)
    
    # Sort the result by id_start
    result_df = result_df.sort_values(by='id_start')
    
    # Return the sorted DataFrame
    return result_df

#Q3 ex: ind_ids_within_ten_percentage_threshold(df2, 1001400)

def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Define rate coefficients
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    
    # Calculate toll rates for each vehicle type
    for vehicle_type, rate in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate
    
    return df

def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Initialize an empty list to store the rows
    rows_list = []
    # Create a new DataFrame to store the results
    result_df = pd.DataFrame()
    
    # Define the discount factors for weekdays and weekends
    discount_factors = {
        'weekday': {
            '00:00:00-10:00:00': 0.8,
            '10:00:00-18:00:00': 1.2,
            '18:00:00-23:59:59': 0.8
        },
        'weekend': {
            '00:00:00-23:59:59': 0.7
        }
    }
    
    # Define the days of the week
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Iterate over each unique (id_start, id_end) pair
    for (id_start, id_end), group_df in df.groupby(['id_start', 'id_end']):
        # Iterate over each day of the week
        for day in days_of_week:
            # Determine if it's a weekday or weekend
            day_type = 'weekend' if day in ['Saturday', 'Sunday'] else 'weekday'
            # Iterate over each time interval
            for time_range, discount_factor in discount_factors[day_type].items():
                start_time_str, end_time_str = time_range.split('-')
                # Create a row for the time interval
                row = {
                    'id_start': id_start,
                    'id_end': id_end,
                    'start_day': day,
                    'start_time': datetime.datetime.strptime(start_time_str, '%H:%M:%S').time(),
                    'end_day': day,
                    'end_time': datetime.datetime.strptime(end_time_str, '%H:%M:%S').time()
                }
                # Apply the discount factor to each vehicle column
                for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
                    row[vehicle] = group_df[vehicle].mean() * discount_factor
                # Append the row dictionary to the rows_list
                rows_list.append(row)
    
    # Convert the list of dictionaries into a DataFrame
    result_df = pd.DataFrame(rows_list)
    
    return result_df