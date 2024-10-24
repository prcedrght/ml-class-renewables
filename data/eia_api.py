import os
import requests as req
import pandas as pd
import time
import math
import json

# Constants
num_records = 1757256
recs_per_request = 5000
max_requests_per_hour = 9000
burst_rate = 5

total_pages = math.ceil(num_records/recs_per_request)

offset = 0
# Manually load environment variables from .env file
def load_env_file(filepath):
    with open(filepath) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

load_env_file('.env')
EIA_API_KEY = os.getenv("EIA_API_KEY")
if not EIA_API_KEY:
    raise ValueError("API_KEY not found in environment variables")

s = req.Session()

eia = pd.DataFrame()

for page in range(total_pages):
    # Update offset
    offset = page * recs_per_request
    if offset == 0:
        eia_url = f"https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/?api_key={EIA_API_KEY}&frequency=hourly&data[0]=value&facets[respondent][]=CAL&facets[respondent][]=CAR&facets[respondent][]=CENT&facets[respondent][]=FLA&facets[respondent][]=MIDA&facets[respondent][]=MIDW&facets[respondent][]=NE&facets[respondent][]=NW&facets[respondent][]=NY&facets[respondent][]=SE&facets[respondent][]=SW&facets[respondent][]=TEN&facets[respondent][]=TEX&start=2020-01-01T00&end=2022-12-31T00&sort[0][column]=period&sort[0][direction]=desc&offset={offset}&length=5000"
    else:
        offset += 1
        eia_url = f"https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/?api_key={EIA_API_KEY}&frequency=hourly&data[0]=value&facets[respondent][]=CAL&facets[respondent][]=CAR&facets[respondent][]=CENT&facets[respondent][]=FLA&facets[respondent][]=MIDA&facets[respondent][]=MIDW&facets[respondent][]=NE&facets[respondent][]=NW&facets[respondent][]=NY&facets[respondent][]=SE&facets[respondent][]=SW&facets[respondent][]=TEN&facets[respondent][]=TEX&start=2020-01-01T00&end=2022-12-31T00&sort[0][column]=period&sort[0][direction]=desc&offset={offset}&length=5000"
    print(offset, page)
    eia_r = s.get(eia_url)
    data = eia_r.json()['response']['data']
    eia = pd.concat([eia, pd.DataFrame.from_records(data)])
    time.sleep(1 / burst_rate)

    if (page + 1) % max_requests_per_hour == 0:
        print(f"Sleeping for 1 hour before {page + 1} requests")
        time.sleep(3600)
# print("All Records Fetched")

eia.to_feather('data/eia.feather')

num_records = 7344

total_pages = math.ceil(num_records/recs_per_request)

offset = 0

state_to_region = {
    "California": "California",
    "North Carolina": "Carolinas",
    "South Carolina": "Carolinas",
    "Iowa": "Central",
    "Kansas": "Central",
    "Minnesota": "Central",
    "Missouri": "Central",
    "Nebraska": "Central",
    "North Dakota": "Central",
    "South Dakota": "Central",
    "Florida": "Florida",
    "Delaware": "Mid-Atlantic",
    "District of Columbia": "Mid-Atlantic",
    "Maryland": "Mid-Atlantic",
    "New Jersey": "Mid-Atlantic",
    "Pennsylvania": "Mid-Atlantic",
    "Virginia": "Mid-Atlantic",
    "West Virginia": "Mid-Atlantic",
    "Illinois": "Midwest",
    "Indiana": "Midwest",
    "Michigan": "Midwest",
    "Ohio": "Midwest",
    "Wisconsin": "Midwest",
    "Connecticut": "New England",
    "Maine": "New England",
    "Massachusetts": "New England",
    "New Hampshire": "New England",
    "Rhode Island": "New England",
    "Vermont": "New England",
    "Idaho": "Northwest",
    "Montana": "Northwest",
    "Oregon": "Northwest",
    "Washington": "Northwest",
    "Wyoming": "Northwest",
    "New York": "New York",
    "Alabama": "Southeast",
    "Georgia": "Southeast",
    "Mississippi": "Southeast",
    "Arizona": "Southwest",
    "Colorado": "Southwest",
    "Nevada": "Southwest",
    "New Mexico": "Southwest",
    "Utah": "Southwest",
    "Tennessee": "Tennessee",
    "Texas": "Texas",
    "Oklahoma": "Central",
    "Louisiana": "Southeast",
    "Arkansas": "Central",
    "Kentucky": "Central",
}

demand = pd.DataFrame()

for page in range(total_pages):
    # Update offset
    offset = page * recs_per_request
    if offset == 0:
        demand_sales_url = f"https://api.eia.gov/v2/electricity/retail-sales/data/?api_key={EIA_API_KEY}&frequency=monthly&data[0]=customers&data[1]=price&data[2]=revenue&data[3]=sales&facets[stateid][]=AK&facets[stateid][]=AL&facets[stateid][]=AR&facets[stateid][]=AZ&facets[stateid][]=CA&facets[stateid][]=CO&facets[stateid][]=CT&facets[stateid][]=DC&facets[stateid][]=DE&facets[stateid][]=FL&facets[stateid][]=GA&facets[stateid][]=HI&facets[stateid][]=IA&facets[stateid][]=ID&facets[stateid][]=IL&facets[stateid][]=IN&facets[stateid][]=KS&facets[stateid][]=KY&facets[stateid][]=LA&facets[stateid][]=MA&facets[stateid][]=MD&facets[stateid][]=ME&facets[stateid][]=MI&facets[stateid][]=MN&facets[stateid][]=MO&facets[stateid][]=MS&facets[stateid][]=MT&facets[stateid][]=NC&facets[stateid][]=ND&facets[stateid][]=NE&facets[stateid][]=NH&facets[stateid][]=NJ&facets[stateid][]=NM&facets[stateid][]=NV&facets[stateid][]=NY&facets[stateid][]=OH&facets[stateid][]=OK&facets[stateid][]=OR&facets[stateid][]=PA&facets[stateid][]=RI&facets[stateid][]=SC&facets[stateid][]=SD&facets[stateid][]=TN&facets[stateid][]=TX&facets[stateid][]=UT&facets[stateid][]=VA&facets[stateid][]=VT&facets[stateid][]=WA&facets[stateid][]=WI&facets[stateid][]=WV&facets[stateid][]=WY&start=2020-01&end=2022-12&sort[0][column]=period&sort[0][direction]=desc&offset={offset}&length=5000"
    else:
        offset += 1
        demand_sales_url = f"https://api.eia.gov/v2/electricity/retail-sales/data/?api_key={EIA_API_KEY}&frequency=monthly&data[0]=customers&data[1]=price&data[2]=revenue&data[3]=sales&facets[stateid][]=AK&facets[stateid][]=AL&facets[stateid][]=AR&facets[stateid][]=AZ&facets[stateid][]=CA&facets[stateid][]=CO&facets[stateid][]=CT&facets[stateid][]=DC&facets[stateid][]=DE&facets[stateid][]=FL&facets[stateid][]=GA&facets[stateid][]=HI&facets[stateid][]=IA&facets[stateid][]=ID&facets[stateid][]=IL&facets[stateid][]=IN&facets[stateid][]=KS&facets[stateid][]=KY&facets[stateid][]=LA&facets[stateid][]=MA&facets[stateid][]=MD&facets[stateid][]=ME&facets[stateid][]=MI&facets[stateid][]=MN&facets[stateid][]=MO&facets[stateid][]=MS&facets[stateid][]=MT&facets[stateid][]=NC&facets[stateid][]=ND&facets[stateid][]=NE&facets[stateid][]=NH&facets[stateid][]=NJ&facets[stateid][]=NM&facets[stateid][]=NV&facets[stateid][]=NY&facets[stateid][]=OH&facets[stateid][]=OK&facets[stateid][]=OR&facets[stateid][]=PA&facets[stateid][]=RI&facets[stateid][]=SC&facets[stateid][]=SD&facets[stateid][]=TN&facets[stateid][]=TX&facets[stateid][]=UT&facets[stateid][]=VA&facets[stateid][]=VT&facets[stateid][]=WA&facets[stateid][]=WI&facets[stateid][]=WV&facets[stateid][]=WY&start=2020-01&end=2022-12&sort[0][column]=period&sort[0][direction]=desc&offset={offset}&length=5000"
    
    demand_r = s.get(demand_sales_url)
    data = demand_r.json()['response']['data']
    tmp = pd.DataFrame.from_records(data)
    tmp['eia_region'] = tmp['stateDescription'].map(state_to_region)
    demand = pd.concat([demand, tmp])

demand.to_feather('data/demand.feather')