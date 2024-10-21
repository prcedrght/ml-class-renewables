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
        eia_url = f"https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/?api_key={EIA_API_KEY}&frequency=hourly&data[0]=value&facets[respondent][]=CAL&facets[respondent][]=CAR&facets[respondent][]=CENT&facets[respondent][]=FLA&facets[respondent][]=MIDA&facets[respondent][]=MIDW&facets[respondent][]=NE&facets[respondent][]=NW&facets[respondent][]=NY&facets[respondent][]=SE&facets[respondent][]=SW&facets[respondent][]=TEN&facets[respondent][]=TEX&start=2022-01-01T00&end=2024-01-01T00&sort[0][column]=period&sort[0][direction]=desc&offset={offset}&length=5000"
    else:
        offset += 1
        eia_url = f"https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/?api_key={EIA_API_KEY}&frequency=hourly&data[0]=value&facets[respondent][]=CAL&facets[respondent][]=CAR&facets[respondent][]=CENT&facets[respondent][]=FLA&facets[respondent][]=MIDA&facets[respondent][]=MIDW&facets[respondent][]=NE&facets[respondent][]=NW&facets[respondent][]=NY&facets[respondent][]=SE&facets[respondent][]=SW&facets[respondent][]=TEN&facets[respondent][]=TEX&start=2022-01-01T00&end=2024-01-01T00&sort[0][column]=period&sort[0][direction]=desc&offset={offset}&length=5000"
    print(offset, page)
    eia_r = s.get(eia_url)
    data = eia_r.json()['response']['data']
    eia = pd.concat([eia, pd.DataFrame.from_records(data)])
    time.sleep(1 / burst_rate)

    if (page + 1) % max_requests_per_hour == 0:
        print(f"Sleeping for 1 hour before {page + 1} requests")
        time.sleep(3600)
print("All Records Fetched")

eia.to_feather('data/eia.feather')