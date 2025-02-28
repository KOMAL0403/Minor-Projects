# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample weather data
def generate_sample_data(n_days=365):
    dates = [datetime(2023, 1, 1) + timedelta(days=x) for x in range(n_days)]
    
    # Generate realistic-looking weather data
    temperature = np.random.normal(20, 8, n_days)  # Mean 20°C with variation
    humidity = np.random.normal(65, 15, n_days)    # Mean 65% with variation
    precipitation = np.random.exponential(5, n_days)  # Exponential distribution for rainfall
    wind_speed = np.random.normal(15, 5, n_days)     # Mean 15 km/h with variation
    
    # Create DataFrame
    weather_data = pd.DataFrame({
        'date': dates,
        'temperature': temperature,
        'humidity': humidity,
        'precipitation': precipitation,
        'wind_speed': wind_speed
    })
    
    # Add some missing values
    weather_data.loc[np.random.choice(weather_data.index, 10), 'temperature'] = np.nan
    weather_data.loc[np.random.choice(weather_data.index, 10), 'humidity'] = np.nan
    
    return weather_data

# Generate sample data
weather_data = generate_sample_data()

# 1. Basic Data Analysis
print("\n=== Basic Data Information ===")
print(weather_data.info())
print("\n=== First Few Rows ===")
print(weather_data.head())

# 2. Data Cleaning
print("\n=== Missing Values Before Cleaning ===")
print(weather_data.isnull().sum())

# Fill missing values
weather_data = weather_data.fillna(weather_data.mean())

print("\n=== Missing Values After Cleaning ===")
print(weather_data.isnull().sum())

# 3. Descriptive Statistics
print("\n=== Descriptive Statistics ===")
print(weather_data.describe())

# 4. Visualizations

# Temperature over time
plt.figure(figsize=(12, 6))
plt.plot(weather_data['date'], weather_data['temperature'])
plt.title('Temperature Over Time')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Distribution plots
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.histplot(weather_data['temperature'], kde=True)
plt.title('Temperature Distribution')

plt.subplot(1, 3, 2)
sns.histplot(weather_data['humidity'], kde=True)
plt.title('Humidity Distribution')

plt.subplot(1, 3, 3)
sns.histplot(weather_data['wind_speed'], kde=True)
plt.title('Wind Speed Distribution')

plt.tight_layout()
plt.show()

# Correlation heatmap
correlation_matrix = weather_data.drop('date', axis=1).corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

# Monthly averages
weather_data['month'] = weather_data['date'].dt.month
monthly_avg = weather_data.groupby('month')[['temperature', 'humidity', 'precipitation']].mean()

plt.figure(figsize=(12, 6))
monthly_avg.plot(kind='bar', width=0.8)
plt.title('Monthly Averages')
plt.xlabel('Month')
plt.legend(title='Weather Parameters')
plt.tight_layout()
plt.show()

# Print summary statistics by month
print("\n=== Monthly Statistics ===")
print(monthly_avg)

# Additional analysis: Weather parameter relationships
plt.figure(figsize=(8, 6))
sns.scatterplot(data=weather_data, x='temperature', y='humidity')
plt.title('Temperature vs Humidity')
plt.show()

# Print key findings
print("\n=== Key Findings ===")
print(f"Average Temperature: {weather_data['temperature'].mean():.2f}°C")
print(f"Average Humidity: {weather_data['humidity'].mean():.2f}%")
print(f"Average Wind Speed: {weather_data['wind_speed'].mean():.2f} km/h")
print(f"Total Precipitation: {weather_data['precipitation'].sum():.2f} mm")