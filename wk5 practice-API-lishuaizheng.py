# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 15:25:57 2019

@author: ALIENWARE
"""
#%%
#Watson Dashboard------    https://console.bluemix.net/developer/watson/dashboard    
#%%
#################   Requests and IBM Watson API   ####################
######################################################################
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
######  sentiment analysis   #####
##################################
#https://console.bluemix.net/iam/#/apikeys    get API
#reference----https://www.ibm.com/watson/developercloud/natural-language-understanding/api/v1/?curl#sentiment
#https://stackoverflow.com/questions/47990658/cannot-pip-install-watson-developer-cloud 
# use the link above to install the waston package
# run anonconda prompt as administrator, then run "python -m ensurepip --default-pip" -> "python -m pip install --upgrade"  ->(run in none administrator)"pip install -I watson-developer-cloud==0.26.1"
import json
import pandas as pd
import json
#%%
endpoint = 'https://gateway-lon.watsonplatform.net/natural-language-understanding/api/v1/analyze'
senti_data = {'url': 'http://www.presidency.ucsb.edu/ws/index.php?pid=85753', 'features': {"sentiment": {'document': 'true'} }}
senti_auth = ('apikey', 'pPUXqaNBBKGCV9eZ9u0BFuqjy8ZQi7Og3nRNCgokrQXr');
headers = {'Content-Type': 'application/json'};
params = (('version', '2017-02-27'),);
senti= requests.post(endpoint, headers=headers, params=params, data=json.dumps(senti_data), auth=senti_auth)
#%%
senti.status_code
senti.json()

from pandas.io.json import json_normalize
test = json_normalize(senti.json()['sentiment'])
test

#%%
##############  Language Detection  #################
#####################################################
#reference-https://github.com/IBM-Bluemix-Docs/language-translator/blob/master/identifiable-languages.md
          # --https://cloud.ibm.com/services/language-translator/crn:v1:bluemix:public:language-translator:eu-gb:a%2f609ad303671746a9837a37eaed023972:ccc25c90-a0f8-4f59-b99e-d1ddff660667::?paneId=gettingStarted
           
lang_endpoint = 'https://gateway-lon.watsonplatform.net/language-translator/api/v3/identify'
lang_data = {'url': 'http://www.presidency.ucsb.edu/ws/index.php?pid=85753'}
lang_auth = ('apikey', 'scuU8vZWp3_ys5rCAheQwbYOuzp0r2ftUEbkQETr3V8t');
lang_headers = {'Content-Type': 'text/plain'};
params = (('version', '2018-05-01'),);
language= requests.post(lang_endpoint, headers=lang_headers, params=params, data=json.dumps(lang_data), auth=lang_auth)
#%%
language.status_code
language.json()
#%%
from pandas.io.json import json_normalize
test = json_normalize(language.json()['languages'])
test
#%%
#####################  Face Detection   ##################
##########################################################
#reference ---https://console.bluemix.net/services/watson-vision-combined/crn%3Av1%3Abluemix%3Apublic%3Awatson-vision-combined%3Aus-south%3Aa%2F609ad303671746a9837a37eaed023972%3A0cc0eafb-5813-4c8a-a898-4c3d174ddb87%3A%3A/?paneId=gettingStarted&new=true
visual_endpoint = 'https://gateway.watsonplatform.net/visual-recognition/api/v3/detect_faces'
visual_data = {'url': 'https://watson-developer-cloud.github.io/doc-tutorial-downloads/visual-recognition/Ginni_Rometty_at_the_Fortune_MPW_Summit_in_2011.jpg'}
visual_auth = ('apikey', '8G0qxlw8tCOHQHFoiNJUZJQKQw7Yn7UATgbZiEBAHwqE');
#visual_headers = {'Content-Type': 'text/plain'};
visual_params = (('version', '2018-03-19'),);
visual= requests.post(visual_endpoint, params=visual_params, data=json.dumps(visual_data), auth=visual_auth)
#headers=visual_headers, 
visual.status_code
visual.json()
#%%
#########################   BING MAP    ############################
####################################################################

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




