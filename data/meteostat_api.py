import pandas as pd
from datetime import datetime
from meteostat import Point, Hourly

start = datetime(2022, 1, 1)
end = datetime(2024, 1, 1)

locs = {
    "California": (36.7783, -119.4179),
    "Carolinas": (34.0007, -81.0348),
    "Central": (39.8283, -98.5795),
    "Florida": (27.9944, -81.7603),
    "Mid-Atlantic": (39.9526, -75.1652),
    "Midwest": (41.8781, -87.6298),
    "New England": (42.4072, -71.3824),
    "Northwest": (45.5051, -122.6750),
    "New York": (40.7128, -74.0060),
    "Southeast": (33.7490, -84.3880),
    "Southwest": (34.0489, -111.0937),
    "Tennessee": (35.5175, -86.5804),
    "Texas": (31.9686, -99.9018)
}

weather = pd.DataFrame()
for loc, latlong in locs.items():
    point = Point(latlong[0], latlong[1])
    data = Hourly(point, start, end)
    tmp = pd.DataFrame(data.fetch().reset_index())
    tmp['location'] = loc
    weather = pd.concat([weather, tmp])

weather.to_feather('data/weather.feather')