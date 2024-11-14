import requests
import pandas as pd
import time
import random
from tqdm import tqdm

cookies = {


    '_trackity':'a68d45a4-6c13-6f1c-ce1d-f684f181274e',
    '_ga':'GA1.1.1862890771.1724678055',
    '_gcl_au':'1.1.675671625.1724678062',
    '_fbp':'fb.1.1724678063314.90320298242522262',
    '__uidac':'0166cc7fb111d338285e6447a89c571e',
    'dtdz':'7813ec8d-3bbe-51bc-9f11-678b14f578e0',
    '__R':'1', '_hjSessionUser_522327':'eyJpZCI6IjA2ZmIxZjc0LWM0NDItNTU2Yy05OGY4LTJhZTBjNDA0M2Q1MSIsImNyZWF0ZWQiOjE3MjQ2NzgwNjIzMDgsImV4aXN0aW5nIjp0cnVlfQ=',
    '__tb':'0', 'TOKENS':'{%22access_token%22:%22mHkbLRapwO1VxuP5T7IjShFycnU9Wfl4%22}',
    '__iid':'749', '__iid':'749', '__su':'0', '__su':'0', '_gcl_aw':'GCL.1729426109.CjwKCAjw1NK4BhAwEiwAVUHPUEQnxkpj0K9cLY-hpWUhcUAUOMyVGYfQAGF80rHi2EvgqgxRDT-yIRoCh1gQAvD_BwE',
    '_gcl_gs':'2.1.k1$i1729426102$u165506106', '__RC':'4',
    '__IP':'1906319940', 'delivery_zone':'Vk4wMzQwMjQwMTM:',
    'SL_G_WPT_TO':'vi', 'SL_GWPT_Show_Hide_tmp':'1', 'SL_wptGlobTipTmp':'1',
    'tiki_client_id':'1862890771.1724678055',
    '_ga_S9GLR1RQFJ':'GS1.1.1730190253.34.1.1730190259.54.0.0',
    '_hjSession_522327':'eyJpZCI6IjQ5MzJmZTJiLWRlOGEtNGE1Ny1hMWFlLWE5M2NmMjdmNTQ4OSIsImMiOjE3MzAxOTAyNjU3NTIsInMiOjAsInIiOjAsInNiIjowLCJzciI6MCwic2UiOjAsImZzIjowLCJzcCI6MH0:',
    '__adm_upl':'eyJ0aW1lIjoxNzMwMTkyMTEzLCJfdXBsIjoiMC0xNDI0Njc4MDY1MTgwOTg3ODQyIn0:', '__uif':'__uid%3A1424678065180987842%7C__ui%3A-1%7C__create%3A1724678093',
    'amp_99d374':'HFFmi9p7fTrCLz5tmzJHGj...1ibbm05ie.1ibbm3brg.13q.17p.2bj'
}
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36 OPR/114.0.0.0',
    'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
    'Referer': 'https://tiki.vn/',
    'x-guest-token': 'mHkbLRapwO1VxuP5T7IjShFycnU9Wfl4',
    'Connection': 'keep-alive',
    'TE': 'Trailers',
}
params = {
    'product_id': '273258825',
    'sort': 'score|desc,id|desc,stars|all',
    'page': '1',
    'limit': '5',
    'include': 'comments'
}

def comment_parser(json):
    d = dict()
    d['id'] = json.get('product_id')
    d['title'] = json.get('title')
    d['content'] = json.get('content')
    d['thank_count'] = json.get('thank_count')
    d['customer_id']  = json.get('customer_id')
    d['created_at'] = json.get('created_at')
    d['customer_name'] = json.get('created_by').get('name')
    d['purchased_at'] = json.get('created_by').get('purchased_at')
    return d


df_id = pd.read_csv('product_id.csv')
p_ids = df_id.id.to_list()
result = []
for pid in tqdm(p_ids, total=len(p_ids)):
    params['product_id'] = pid
    print('Crawl comment for product {}'.format(pid))
    for i in range(2):
        params['page'] = i
        response = requests.get('https://tiki.vn/api/v2/reviews', headers=headers, params=params, cookies=cookies)
        if response.status_code == 200:
            print('Crawl comment page {} success!!!'.format(i))
            for comment in response.json().get('data'):
                result.append(comment_parser(comment))
df_comment = pd.DataFrame(result)
df_comment.to_csv('comments_data.csv', index=False)