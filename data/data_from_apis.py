import requests as req
import pandas as pd
import time

ninja_token = "78bb218e335bbfb2ebc6d2650f553ac50b72a082"
ninja_api_base = "https://www.renewables.ninja/api/"

s = req.Session()
s.headers = {"Authorization": f"Token {ninja_token}"}
# pv_url = ninja_api_base + "data/pv"
wind_url = ninja_api_base + 'data/wind'

delay = 3600 / 50

country_lat_lon = {
    "Spain": (40.4168, -3.7038),
    "Austria": (47.5162, 14.5501),
    "Portugal": (39.3999, -8.2245),
    "Albania": (41.1533, 20.1683),
    "Netherlands": (52.1326, 5.2913),
    "Italy": (41.8719, 12.5674),
    "Romania": (45.9432, 24.9668),
    "Czech Republic": (49.8175, 15.4730),
    "Ireland": (53.1424, -7.6921),
    "Belgium": (50.8503, 4.3517),
    "Switzerland": (46.8182, 8.2275),
    "Germany": (51.1657, 10.4515),
    "Great Britain": (55.3781, -3.4360),
    "Slovenia": (46.1512, 14.9955),
    "Macedonia": (41.6086, 21.7453),
    "Greece": (39.0742, 21.8243),
    "Norway": (60.4720, 8.4689),
    "Hungary": (47.1625, 19.5033),
    "Bulgaria": (42.7339, 25.4858),
    "Montenegro": (42.7087, 19.3744),
    "Croatia": (45.1, 15.2),
    "France": (46.6034, 1.8883),
    "Moldova": (47.4116, 28.3699),
    "Serbia": (44.0165, 21.0059),
    "Poland": (51.9194, 19.1451),
    "Estonia": (58.5953, 25.0136),
    "Luxembourg": (49.8153, 6.1296),
    "Latvia": (56.8796, 24.6032),
    "Lithuania": (55.1694, 23.8813),
    "Bosnia and Herzegovina": (43.9159, 17.6791),
    "Finland": (61.9241, 25.7482),
    "Malta": (35.9375, 14.3754),
    "Slovakia": (48.6690, 19.6990),
    "Denmark": (56.2639, 9.5018),
    "Cyprus": (35.1264, 33.4299),
    "Sweden": (60.1282, 18.6435),
}

years = [2018, 2019, 2020,]

pv_df = pd.DataFrame()
wind_df = pd.DataFrame()

i = 0
for c, ll in country_lat_lon.items():
    for y in years:
        i += 1
        print(i, c, y)
        pv_args = {
            "lat": ll[0],
            "lon": ll[1],
            "date_from": f"{y}-01-01",
            "date_to": f"{y}-12-31",
            "dataset": "merra2",
            "capacity": 1.0,
            "system_loss": 0.1,
            "tracking": 0,
            "tilt": 35,
            "azim": 180,
            "format": "json",
            "raw": True,
        }

        pv_r = s.get(pv_url, params=pv_args)

        # Check the status code and content of the responses
        if pv_r.status_code == 200:
            pv_txt = pv_r.json()
            tmp_df = pd.DataFrame.from_dict(pv_txt['data'], orient='index')
            tmp_df = tmp_df.assign(country=c)
            pv_df = pd.concat([pv_df, tmp_df])
        else:
            print(f"PV Request failed with status code: {pv_r.status_code}")
            print(f"PV Response Text: {pv_r.text}")
            continue
        print('loading into PV pickle')
        pv_df.to_pickle('pv_data.pkl')
        
        time.sleep(10)

        wind_args = {
            'lat': ll[0],
            'lon': ll[1],
            'date_from': f'{y}-01-01',
            'date_to': f'{y}-12-31',
            'capacity': 1.0,
            'height': 100,
            'turbine': 'Vestas V80 2000',
            'format': 'json'
        }

        wind_r = s.get(wind_url, params=wind_args)

        if wind_r.status_code == 200:
            wind_txt = wind_r.json()
            tmp_df = pd.DataFrame.from_dict(wind_txt['data'], orient='index')
            tmp_df = tmp_df.assign(country=c)
            wind_df = pd.concat([wind_df, tmp_df])
        else:
            print(f"Wind Request failed with status code: {wind_r.status_code}")
            print(f"Wind Response Text: {wind_r.text}")
            continue
        print('loading into Wind pickle')
        wind_df.to_pickle('wind_data.pkl')        
        time.sleep(delay)