import requests
from config import Config
import json
import pprint
cookies, headers, params = Config.get_config_product_id()
response = requests.get('https://tiki.vn/api/personalish/v1/blocks/listings', headers=headers, params=params, cookies=cookies)
to_json = response.json().get('data') ## dict
for record in to_json:
    print(record.get('id'))
### export to json file
with open('response.json', 'w') as f:
    json.dump(to_json, f, indent=4)
