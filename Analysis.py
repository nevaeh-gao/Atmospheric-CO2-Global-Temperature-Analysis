import pandas as pd
import math


# Load CO2 data

def load_co2_data(filename):
    """
    Load the daily CO2 dataset.

    Returns a pandas DataFrame containing:
    Year, Month, Day, Decimal_date, co2_mol_frac
    """
    co2_data = pd.read_csv(
        filename,
        names=['Year', 'Month', 'Day', 'Decimal_date', 'co2_mol_frac'],
        skiprows=32,
        na_values=[-999, -999.0, -999.00, -999.000000]
    )

    return co2_data


# Calculate basic statistics

def calculate_basic_statistics(co2_data, start_year, end_year):
    """
    Calculate mean, standard deviation, minimum, maximum,
    and count of available CO2 measurements.
    """

    # Do not modify original dataframe
    data = co2_data.copy()

    # Select requested years
    mask = (
        (data['Year'] >= start_year) &
        (data['Year'] <= end_year)
    )

    values = data.loc[mask, 'co2_mol_frac'].dropna()

    mean = values.mean()

    # Population standard deviation
    std = values.std(ddof=0)

    minimum = values.min()
    maximum = values.max()
    count = values.count()

    return [
        float(mean),
        float(std),
        float(minimum),
        float(maximum),
        float(count)
    ]


# Decimal date

def to_decimal_date(year, month_of_year, day_of_month):
    """
    Convert a calendar date into decimal-year format.
    """

    # Determine whether year is a leap year
    leap_year = (
        year % 400 == 0 or
        (year % 4 == 0 and year % 100 != 0)
    )

    if leap_year:
        days_in_month = [
            31, 29, 31, 30, 31, 30,
            31, 31, 30, 31, 30, 31
        ]
        days_in_year = 366
    else:
        days_in_month = [
            31, 28, 31, 30, 31, 30,
            31, 31, 30, 31, 30, 31
        ]
        days_in_year = 365

    # Number of complete days before the target date
    days_passed = sum(days_in_month[:month_of_year - 1])
    days_passed += day_of_month - 1

    return year + days_passed / days_in_year


# Estimate linear trend

def estimate_linear_trend(co2_data, start_year, end_year):
    """
    Estimate the linear relationship between decimal date
    and CO2 concentration.

    Returns [slope, intercept].
    """

    data = co2_data.copy()

    mask = (
        (data['Year'] >= start_year) &
        (data['Year'] <= end_year)
    )

    filtered = data.loc[
        mask,
        ['Decimal_date', 'co2_mol_frac']
    ].dropna()

    t = filtered['Decimal_date']
    x = filtered['co2_mol_frac']

    t_bar = t.mean()
    x_bar = x.mean()

    numerator = ((t - t_bar) * (x - x_bar)).sum()
    denominator = ((t - t_bar) ** 2).sum()

    slope = numerator / denominator
    intercept = x_bar - slope * t_bar

    return [float(slope), float(intercept)]


# Extract seasonal component

def extract_seasonal_component(co2_data, start_year, end_year):
    """
    Remove the estimated linear trend and calculate the
    average detrended CO2 concentration for each month.
    """

    data = co2_data.copy()

    slope, intercept = estimate_linear_trend(
        data,
        start_year,
        end_year
    )

    mask = (
        (data['Year'] >= start_year) &
        (data['Year'] <= end_year)
    )

    filtered = data.loc[
        mask,
        ['Month', 'Decimal_date', 'co2_mol_frac']
    ].dropna().copy()

    # Predicted CO2 from linear trend
    trend = (
        slope * filtered['Decimal_date']
        + intercept
    )

    # Remove trend
    filtered['detrended'] = (
        filtered['co2_mol_frac'] - trend
    )

    seasonal = []

    for month in range(1, 13):

        month_values = filtered.loc[
            filtered['Month'] == month,
            'detrended'
        ]

        seasonal.append(float(month_values.mean()))

    return seasonal


# Linear prediction

def predict_co2_future_linear(
        co2_data,
        start_year,
        end_year,
        target_year,
        target_month,
        target_day):
    """
    Predict CO2 using only the fitted linear trend.
    """

    slope, intercept = estimate_linear_trend(
        co2_data,
        start_year,
        end_year
    )

    target_date = to_decimal_date(
        target_year,
        target_month,
        target_day
    )

    prediction = slope * target_date + intercept

    return float(prediction)


# Linear + seasonal prediction

def predict_co2_future_linear_and_seasonal(
        co2_data,
        start_year,
        end_year,
        target_year,
        target_month,
        target_day):
    """
    Predict CO2 using both the linear trend and
    the average seasonal component.
    """

    linear_prediction = predict_co2_future_linear(
        co2_data,
        start_year,
        end_year,
        target_year,
        target_month,
        target_day
    )

    seasonal = extract_seasonal_component(
        co2_data,
        start_year,
        end_year
    )

    # January is index 0, February index 1, etc.
    seasonal_component = seasonal[target_month - 1]

    return float(linear_prediction + seasonal_component)


# Load temperature data

def load_temperature_data(filename):
    """
    Load monthly temperature data.

    The file is whitespace separated. The first three columns
    contain year, month and the temperature observation needed
    for Q5b.
    """

    temperature_data = pd.read_csv(
        filename,
        sep=r'\s+',
        header=None,
        usecols=[0, 1, 2],
        names=['Year', 'Month', 'Temp'],
        na_values=[-999, -999.0, -999.000000]
    )

    return temperature_data


# CO2-temperature correlation

def compute_correlation_co2_temperature(
        co2_data,
        temperature_data,
        start_year,
        end_year):
    """
    Calculate Pearson correlation between monthly-average
    CO2 concentration and monthly temperature.
    """

    # Do not modify the supplied dataframes
    co2_df = co2_data.copy()
    temp_df = temperature_data.copy()

    # Restrict CO2 data to requested years FIRST
    co2_mask = (
        (co2_df['Year'] >= start_year) &
        (co2_df['Year'] <= end_year)
    )

    co2_df = co2_df.loc[
        co2_mask,
        ['Year', 'Month', 'co2_mol_frac']
    ].dropna()

    # Restrict temperature data to requested years
    temp_mask = (
        (temp_df['Year'] >= start_year) &
        (temp_df['Year'] <= end_year)
    )

    temp_df = temp_df.loc[
        temp_mask,
        ['Year', 'Month', 'Temp']
    ].dropna()

    # If either dataset contains no observations
    if co2_df.empty or temp_df.empty:
        return math.nan

    # Calculate one average CO2 concentration per year/month
    monthly_co2 = (
        co2_df
        .groupby(['Year', 'Month'], as_index=False)
        ['co2_mol_frac']
        .mean()
    )

    # Match observations using BOTH year and month
    merged = pd.merge(
        monthly_co2,
        temp_df,
        on=['Year', 'Month'],
        how='inner'
    )

    # Correlation cannot be calculated with fewer than 2 pairs
    if len(merged) < 2:
        return math.nan

    x = merged['co2_mol_frac']
    y = merged['Temp']

    x_bar = x.mean()
    y_bar = y.mean()

    numerator = (
        (x - x_bar) *
        (y - y_bar)
    ).sum()

    sum_x_squared = ((x - x_bar) ** 2).sum()
    sum_y_squared = ((y - y_bar) ** 2).sum()

    denominator = math.sqrt(
        sum_x_squared * sum_y_squared
    )

    # Avoid division by zero
    if denominator == 0:
        return math.nan

    correlation = numerator / denominator

    return float(correlation)