import pandas as pd
from datetime import datetime
from meteostat import Point, Hourly

start = datetime(2022, 1, 1)
end = datetime(2024, 1, 1)

regions_coordinates = {
    "California": [
        (40.0000, -120.0000),
        (36.7783, -119.4179),
        (34.0522, -118.2437)
    ],
    "Carolinas": [
        (35.7596, -79.0193),
        (35.2271, -80.8431),
        (33.8361, -81.1637)
    ],
    "Central": [
        (39.0119, -98.4842),
        (38.5767, -92.1735),
        (41.4925, -99.9018)
    ],
    "Florida": [
        (30.3322, -81.6557),
        (28.5383, -81.3792),
        (25.7617, -80.1918)
    ],
    "Mid-Atlantic": [
        (37.4316, -78.6569),
        (39.0458, -76.6413),
        (41.2033, -77.1945)
    ],
    "Midwest": [
        (40.6331, -89.3985),
        (40.4173, -82.9071),
        (44.3148, -85.6024)
    ],
    "New England": [
        (42.4072, -71.3824),
        (44.5588, -72.5778),
        (45.2538, -69.4455)
    ],
    "Northwest": [
        (47.7511, -120.7401),
        (43.8041, -120.5542),
        (44.0682, -114.7420)
    ],
    "New York": [
        (42.8864, -78.8784),
        (43.0481, -76.1474),
        (40.7128, -74.0060)
    ],
    "Southeast": [
        (32.1656, -82.9001),
        (32.3182, -86.9023),
        (32.3547, -89.3985)
    ],
    "Southwest": [
        (34.0489, -111.0937),
        (34.5199, -105.8701),
        (38.8026, -116.4194)
    ],
    "Tennessee": [
        (35.2010, -89.9711),
        (36.1627, -86.7816),
        (35.9606, -83.9207)
    ],
    "Texas": [
        (32.7767, -96.7970),
        (30.2672, -97.7431),
        (29.4241, -98.4936)
    ]
}

weather = pd.DataFrame()
for loc, latlongs in regions_coordinates.items():
    for latlong in latlongs:
        point = Point(latlong[0], latlong[1])
        data = Hourly(point, start, end)
        tmp = pd.DataFrame(data.fetch().reset_index())
        tmp['coord'] = f"{latlong[0]},{latlong[1]}"
        tmp['location'] = loc
        weather = pd.concat([weather, tmp])

weather.to_feather('data/weather.feather')