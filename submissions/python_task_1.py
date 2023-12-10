import pandas as pd
import numpy as np

df = pd.read_csv(r'c:\Users\Yash\Downloads\dataset-1.csv')
df1 = pd.read_csv(r'c:\Users\Yash\Downloads\dataset-2.csv')

def generate_car_matrix(df)->pd.DataFrame:
    """
    Creates a DataFrame  for id combinations.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Matrix generated with 'car' values, 
                          where 'id_1' and 'id_2' are used as indices and columns respectively.
    """
    df = df.pivot(index='id_1', columns='id_2', values='car').fillna(0)
    
    # Set the diagonal values to 0
    np.fill_diagonal(df.values, 0)

    return df

def get_type_count(df)->dict:
    """
    Categorizes 'car' values into types and returns a dictionary of counts.

    Args:
        df (pandas.DataFrame)

    Returns:
        dict: A dictionary with car types as keys and their counts as values.
    """
    def categorize_car(value):
        if value <= 15:
            return 'low'
        elif 15 < value <= 25:
            return 'medium'
        else:
            return 'high'
    
    # Apply the categorization function to the 'car' column
    df['car_type'] = df['car'].apply(categorize_car)
    
    # Count the occurrences of each car type
    type_counts = df['car_type'].value_counts().to_dict()

    return dict(sorted(type_counts.items()))

def get_bus_indexes(df)->list:
    """
    Returns the indexes where the 'bus' values are greater than twice the mean.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of indexes where 'bus' values exceed twice the mean.
    """
    # Calculate the mean of the 'bus' column
    bus_mean = df['bus'].mean()
    
    # Identify the indices where 'bus' values are greater than twice the mean
    condition = df['bus'] > 2 * bus_mean
    indices = df[condition].index
    
    # Sort the indices in ascending order
    indices.sort_values()

    return list(indices)

def filter_routes(df)->list:
    """
    Filters and returns routes with average 'truck' values greater than 7.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of route names with average 'truck' values greater than 7.
    """
    route_means = df.groupby('route')['truck'].mean()
    
    # Filter routes with average 'truck' values greater than 7
    filtered_routes = route_means[route_means > 7].index

    return list(filtered_routes.sort_values())

def multiply_matrix(matrix)->pd.DataFrame:
    """
    Multiplies matrix values with custom conditions.

    Args:
        matrix (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Modified matrix with values multiplied based on custom conditions.
    """
    def custom_multiply(value):
        if value > 20:
            return round(value * 0.75, 1)
        else:
            return round(value * 1.25, 1)
    
    # Apply the custom function to each element in the DataFrame
    matrix = matrix.map(custom_multiply)

    return matrix

matrix = generate_car_matrix(df)

def time_check(df)->pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Specify the format of your date and time strings
    datetime_format = '%A %H:%M:%S'  # Example format: 'Monday 23:59:59'

    # Combine 'startDay', 'startTime', 'endDay', 'endTime' into full timestamps
    df['startTimestamp'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'], format=datetime_format)
    df['endTimestamp'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'], format=datetime_format)
    
    # Function to check completeness for a single group
    def is_incomplete(group):
        hours_covered = set(group['startTimestamp'].dt.hour) | set(group['endTimestamp'].dt.hour)
        days_covered = set(group['startTimestamp'].dt.dayofweek) | set(group['endTimestamp'].dt.dayofweek)
        return not (len(hours_covered) == 24 and len(days_covered) == 7)
    
    # Group by 'id' and 'id_2' and apply the completeness check
    incomplete_series = df.groupby(['id', 'id_2']).apply(is_incomplete)
    
    return pd.Series(incomplete_series)

