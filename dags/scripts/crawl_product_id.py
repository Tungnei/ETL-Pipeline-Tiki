import requests
import time
import random
import pandas as pd
from tqdm import tqdm

cookies = {
    'trackity': 'a68d45a4-6c13-6f1c-ce1d-f684f181274e',
    '_ga': 'GA1.1.1862890771.1724678055',
    '_gcl_au': '1.1.675671625.1724678062',
    '_fbp': 'fb.1.1724678063314.90320298242522262',
    '__uidac': '0166cc7fb111d338285e6447a89c571e',
    'dtdz': '7813ec8d-3bbe-51bc-9f11-678b14f578e0',
    '__R': '1', 
    '_hjSessionUser_522327': 'eyJpZCI6IjA2ZmIxZjc0LWM0NDItNTU2Yy05OGY4LTJhZTBjNDA0M2Q1MSIsImNyZWF0ZWQiOjE3MjQ2NzgwNjIzMDgsImV4aXN0aW5nIjp0cnVlfQ=', '__tb': '0', 'TOKENS': '{%22access_token%22:%22mHkbLRapwO1VxuP5T7IjShFycnU9Wfl4%22}',
    '__iid': '749', '__iid': '749', '__su': '0', '__su': '0',
    '_gcl_aw': 'GCL.1729426109.CjwKCAjw1NK4BhAwEiwAVUHPUEQnxkpj0K9cLY-hpWUhcUAUOMyVGYfQAGF80rHi2EvgqgxRDT-yIRoCh1gQAvD_BwE',
    '_gcl_gs': '2.1.k1$i1729426102$u165506106',
    '__RC': '4', 'delivery_zone': 'Vk4wMzQwMjQwMTM=',
    'SL_G_WPT_TO': 'vi',
    'tiki_client_id': '1862890771.1724678055', 'SL_GWPT_Show_Hide_tmp': '1',
    'SL_wptGlobTipTmp': '1', '_hjSession_522327': 'eyJpZCI6IjIyMGQ5OGY4LWI4MjMtNDBjMS1iYTY1LWU2NmY1OTc5N2I0OSIsImMiOjE3Mjk4NjY3MzE3MzIsInMiOjAsInIiOjAsInNiIjowLCJzciI6MCwic2UiOjAsImZzIjowLCJzcCI6MH0=',
    '__adm_upl': 'eyJ0aW1lIjoxNzI5ODY4NTc4LCJfdXBsIjoiMC0xNDI0Njc4MDY1MTgwOTg3ODQyIn0=', '__uif': '__uid%3A1424678065180987842%7C__ui%3A-1%7C__create%3A1724678093',
    '__IP': '1743668306', '_ga_S9GLR1RQFJ': 'GS1.1.1729866727.30.1.1729866781.6.0.0', 
    'amp_99d374': 'HFFmi9p7fTrCLz5tmzJHGj...1ib21etqv.1ib21glb3.12s.166.292',
}

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36 OPR/114.0.0.0',
    'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
    'Referer': 'https://tiki.vn/dien-thoai-may-tinh-bang/c1789',
    'x-guest-token': 'mHkbLRapwO1VxuP5T7IjShFycnU9Wfl4',
    'Connection': 'keep-alive',
    'TE': 'Trailers',
}

params = {
    'limit': '40',
    'include': 'advertisement',
    'aggregations': '2',
    'trackity_id': 'a68d45a4-6c13-6f1c-ce1d-f684f181274e',
    'category': '1789',
    'page': '1',
    'src': 'c1789',
    'urlKey':  'dien-thoai-may-tinh-bang',
}

product_id = []
def extract_product_id():
    for i in range(1, 11):
        params['page'] = i
        response = requests.get('https://tiki.vn/api/personalish/v1/blocks/listings', headers=headers, params=params, cookies=cookies)
        if response.status_code == 200:
            print(f'request success!!!{i}')
            for record in response.json().get('data'):
                product_id.append({'id': record.get('id')})
        time.sleep(random.randrange(3, 10))

    df = pd.DataFrame(product_id)
    df.drop_duplicates(subset=['id'], inplace=True)
    df.to_csv('product_id.csv', index=False)
extract_product_id()


def parser_product(json):
    d = dict()
    d['id'] = json.get('id')
    d['sku'] = json.get('sku')
    d['short_description'] = json.get('short_description')
    d['price'] = json.get('price')
    d['list_price'] = json.get('list_price')
    d['price_usd'] = json.get('price_usd')
    d['discount'] = json.get('discount')
    d['discount_rate'] = json.get('discount_rate')
    d['review_count'] = json.get('review_count')
    d['order_count'] = json.get('order_count')
    d['inventory_status'] = json.get('inventory_status')
    d['is_visible'] = json.get('is_visible')
    d['stock_item_qty'] = json.get('stock_item').get('qty')
    d['stock_item_max_sale_qty'] = json.get('stock_item').get('max_sale_qty')
    d['product_name'] = json.get('meta_title')
    d['brand_id'] = json.get('brand').get('id')
    d['brand_name'] = json.get('brand').get('name')
    return d


df_id = pd.read_csv('product_id.csv')
p_ids = df_id.id.to_list()
print(p_ids)
result = []
for pid in tqdm(p_ids, total=len(p_ids)):
    response = requests.get('https://tiki.vn/api/v2/products/{}'.format(pid), headers=headers, params=params, cookies=cookies)
    if response.status_code == 200:
        print('Crawl data {} success !!!'.format(pid))
        result.append(parser_product(response.json()))
    # time.sleep(random.randrange(3, 5))
df_product = pd.DataFrame(result)
df_product.to_csv('crawled_data_ncds.csv', index=False)