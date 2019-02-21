# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 15:25:57 2019

@author: ALIENWARE
"""
#%%
import requests
import pandas as pd
import json

pd.set_option('display.max_rows', 300) # specifies number of rows to show
pd.options.display.float_format = '{:40,.4f}'.format # specifies default number format to 4 decimal places

#%%
endpoint = 'https://gateway-lon.watsonplatform.net/natural-language-understanding/api/v1/analyze'
data = {'url': 'http://www.presidency.ucsb.edu/ws/index.php?pid=85753', 'features': {"keywords": {'sentiment': 'true', 'emotion': 'true', 'limit': 49} } }
auth = ('apikey', 'pPUXqaNBBKGCV9eZ9u0BFuqjy8ZQi7Og3nRNCgokrQXr');

# This defines that we want our answer as a JSON file
headers = {'Content-Type': 'application/json'};

# We need to select the version of the API we are using
params = (('version', '2017-02-27'),);
r = requests.post(endpoint, headers=headers, params=params, data=json.dumps(data), auth=auth)
r.status_code
r.json()

#%%
from pandas.io.json import json_normalize
test = json_normalize(r.json()['keywords'])
test
#%%
#BING MAP

import urllib
# Get a Bing Map Key - https://docs.microsoft.com/en-us/bingmaps/getting-started/bing-maps-dev-center-help/getting-a-bing-maps-key
# First, Define the query we will send
query = "Warren Street London";
# Then, Enter your API Key
api_key = "Aj1xWoWRCvVsXPbW8-qpgfyN9yZkP0kNxBkC-o6AYtdKXto8e_9ohC-k1jWv8Qb4";
# Now create our endpoint, we URL encode our query to turn spaces into a URL String
endpoint = "http://dev.virtualearth.net/REST/v1/Locations/UK/" + urllib.parse.quote(query) + "?maxResults=10&key=" + api_key
print(endpoint)

r = requests.get(endpoint);
r.status_code   #4**means failed, adn 2** means success
r.json()
from pandas.io.json import json_normalize
test = json_normalize(r.json()['resourceSets'][0]['resources'])
test

#%%
endpoint2="http://dev.virtualearth.net/REST/v1/Elevation/List?points=35.89431,-110.72522,35.89393,-110.72578,35.89374,-110.72560&heights=sealevel&key=Aj1xWoWRCvVsXPbW8-qpgfyN9yZkP0kNxBkC-o6AYtdKXto8e_9ohC-k1jWv8Qb4"
##remember to remove the {} in the url
r2= requests.get(endpoint2)
r2.status_code
r2.json()
#%% pic
endpoint3="https://dev.virtualearth.net/REST/v1/Imagery/Map/imagerySet?mapArea=45.219,-122.325,47.610,-122.107&mapSize=500,1000&pushpin=47.610,-122.107;5;-100.107&mapLayer=TrafficFlow&format=png&mapMetadata=0&key=Aj1xWoWRCvVsXPbW8-qpgfyN9yZkP0kNxBkC-o6AYtdKXto8e_9ohC-k1jWv8Qb4" 
r3= requests.get(endpoint3)
r3.status_code
r3.json()




