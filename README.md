# Atmospheric-CO2-Global-Temperature-Analysis
A Python data analysis project investigating long-term atmospheric CO₂ trends, seasonal variation, and the relationship between CO₂ concentration and global temperature.

# Project Overview

This project analyses daily atmospheric CO₂ measurements from Mauna Loa Observatory together with monthly global temperature observations. The analysis covers data preprocessing, statistical analysis, trend estimation, seasonal pattern extraction, forecasting, and correlation analysis.

The project was originally developed as part of coursework at the Australian National University (ANU) and has been organised here as a standalone data analysis project.

# Key Features
Load and clean daily CO₂ and monthly temperature datasets
Calculate descriptive statistics over user-specified time periods
Convert calendar dates to decimal-year representations
Estimate long-term CO₂ trends using simple linear regression implemented from scratch
Remove linear trends to identify monthly seasonal patterns
Forecast future CO₂ concentrations using trend and seasonal components
Aggregate daily CO₂ observations into monthly averages
Align CO₂ and temperature observations by year and month
Compute Pearson correlation coefficients from scratch to examine the relationship between CO₂ concentration and temperature

# Technologies
Python
Pandas
Math

# Data
*Atmospheric CO₂

Daily atmospheric CO₂ measurements are based on observations from the NOAA Global Monitoring Laboratory at Mauna Loa Observatory, Hawaii. The recovered NOAA dataset contains daily observations beginning in 1974.

*Global Temperature

The temperature dataset contains monthly global temperature observations beginning in 1850. The analysis uses the monthly temperature series and matches observations with monthly-average CO₂ concentrations by year and month.

# Methods

The project implements the main statistical calculations directly rather than relying on specialised statistical or machine-learning libraries. These include:

Descriptive statistics
Simple linear regression
Detrending and seasonal decomposition
Trend-based forecasting
Pearson correlation

This approach provides practice in both implementing statistical methods mathematically and applying them to real-world environmental datasets.
