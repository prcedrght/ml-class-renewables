import requests as req
import pandas as pd
import json

ninja_token = '78bb218e335bbfb2ebc6d2650f553ac50b72a082'
ninja_api_base = 'https://www.renewables.ninja/api/'

s = req.Session()
s.headers = {'Authorization': f'Token {ninja_token}'}
# country_ids = 'US'
pv_url = ninja_api_base + 'data/pv'
# wind_url = api_base + 'data/wind'

pv_args = {
    'lat': 40.33,
    'lon': -60.63,
    'date_from': '2023-01-01',
    'date_to': '2023-12-31',
    'dataset': 'merra2',
    'capacity': 1.0,
    'system_loss': 0.1,
    'tracking': 0,
    'tilt': 35,
    'azim': 180,
    'format': 'json',
    'raw': True
}

pv_r = s.get(pv_url, params=pv_args)

# Check the status code and content of the responses
if pv_r.status_code == 200:
    pv_txt = pv_r.json()
    print(pv_txt['data'])
else:
    print(f"PV Request failed with status code: {pv_r.status_code}")
    print(f"PV Response Text: {pv_r.text}")

# wind_args = {
#     'lat': 34.125,
#     'lon': 39.814,
#     'date_from': '2015-01-01',
#     'date_to': '2015-12-31',
#     'capacity': 1.0,
#     'height': 100,
#     'turbine': 'Vestas V80 2000',
#     'format': 'json'
# }


# wind_r = s.get(wind_url, params=wind_args)

# models_r = s.get(ninja_api_base + 'models')
# print(models_r.json())
# if wind_r.status_code == 200:
#     try:
#         wind_pv = json.loads(wind_r.text)
#         wind_data = pd.read_json(wind_pv['data'], orient='index')
#         wind_metadata = wind_pv['metadata']
#         print(wind_data.head())
#     except json.JSONDecodeError as e:
#         print(f"Error decoding Wind JSON: {e}")
#         print(f"Wind Response Text: {wind_r.text}")
# else:
#     print(f"Wind Request failed with status code: {wind_r.status_code}")
#     print(f"Wind Response Text: {wind_r.text}")