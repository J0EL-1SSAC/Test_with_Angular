import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Configuration
stations = [
    "Wimco Nagar Depot", "Wimco Nagar", "Tiruvottiyur", "Tiruvottiyur Theradi", "Kaladipet",
    "Tollgate", "Tondiarpet", "Sir Theagaraya College", "Washermanpet", "Mannadi",
    "High Court", "Puratchi Thalaivar Dr. M.G. Ramachandran Central", "Government Estate",
    "LIC", "Thousand Lights", "AG-DMS", "Teynampet", "Nandanam", "Saidapet", "Little Mount",
    "Guindy", "Alandur", "Nanganallur Road", "Meenambakkam", "Chennai International Airport",
    "St. Thomas Mount", "Ashok Nagar", "Ekkattuthangal", "Arumbakkam",
    "Vadapalani", "Koyambedu", "Thirumangalam", "Anna Nagar Tower",
    "Anna Nagar East", "Shenoy Nagar", "Pachaiyappa's College", "Kilpauk", "Nehru Park",
    "Egmore", "Puratchi Thalaivi Dr. J. Jayalalithaa CMBT"
]
n_samples = 50000
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)

data = []

# Helper function to determine crowd level based on various factors
def get_crowd_level(hour, day_of_week, is_holiday, is_peak_hour):
    base_crowd = 0

    # Time of day
    if 7 <= hour < 11 or 16 <= hour < 20: # Peak hours
        base_crowd += random.uniform(0.6, 1.0)
    elif 11 <= hour < 16: # Mid-day
        base_crowd += random.uniform(0.3, 0.6)
    else: # Off-peak
        base_crowd += random.uniform(0.1, 0.4)

    # Day of week
    if day_of_week >= 5:  # Weekend
        base_crowd += random.uniform(0.1, 0.3)
    else: # Weekday
        base_crowd += random.uniform(0.0, 0.1)

    # Special conditions
    if is_holiday:
        base_crowd += random.uniform(0.2, 0.4) # Higher crowd on holidays

    if is_peak_hour:
         base_crowd *= 1.2 # Amplify crowd during peak hours

    # Add random noise
    base_crowd += np.random.normal(0, 0.05)

    # Clamp the value between 0 and 1
    return max(0, min(1, base_crowd))


print("Generating synthetic metro data...")

for _ in range(n_samples):
    # Generate random timestamp
    time_offset = random.randint(0, int((end_date - start_date).total_seconds()))
    timestamp = start_date + timedelta(seconds=time_offset)

    station = random.choice(stations)
    hour = timestamp.hour
    day_of_week = timestamp.weekday() # Monday = 0, Sunday = 6
    month = timestamp.month
    is_weekend = 1 if day_of_week >= 5 else 0

    # Simple logic for holidays (e.g., Pongal, Diwali)
    is_holiday = 1 if month in [1, 10, 11, 12] and random.random() < 0.1 else 0

    # Determine if it's a peak hour
    is_peak_hour = 1 if (7 <= hour < 11) or (16 <= hour < 20) else 0

    # Get the crowd level
    crowd_level = get_crowd_level(hour, day_of_week, is_holiday, is_peak_hour)

    data.append({
        "station": station,
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
        "is_weekend": is_weekend,
        "is_holiday": is_holiday,
        "is_peak_hour": is_peak_hour,
        "crowd_level": crowd_level
    })

df = pd.DataFrame(data)
output_path = "metro_data.csv"
df.to_csv(output_path, index=False)

print(f"Successfully generated {len(df)} samples.")
print(f"Dataset saved to {output_path}")

