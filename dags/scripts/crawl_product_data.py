import pandas as pd
import requests
import time
import random
from tqdm import tqdm
from datetime import datetime
from scripts.config import product_list, output_file,output_dir
import os
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

params = (
    ('platform', 'web'),
    ('spid', 273259163)
)

def parser_product(json):
    try:
        d = dict()
        d['id'] = json.get('id')
        d['product_name'] = json.get('name')
        d['brand_name'] = json.get('brand', {}).get('name')
        d['short_description'] = json.get('short_description')
        d['original_price'] = json.get('original_price')
        d['price_after_voucher'] = json.get('price')
        d['discount_rate'] = json.get('discount_rate')
        d['discount_price'] = json.get('discount')
        
        quantity_sold = json.get('quantity_sold')
        d['quantity_sold'] = quantity_sold.get('value') if quantity_sold else 0
        
        d['rating_average'] = json.get('rating_average')
        d['review_count'] = json.get('review_count')

        warranty_info = json.get('warranty_info', [])
        d['warranty_info'] = warranty_info[0].get('value') if warranty_info else "No warranty info"
        
        return_policy = json.get('return_policy')
        d['return_policy'] = return_policy.get('title') if return_policy else "No return policy"

        images = json.get('images', [])
        d['url_img'] = images[0].get('base_url') if images else None
        
        return d
    except Exception as e:
        print(f"Error parsing product data: {str(e)}")
        return None

def crawl_product(pid):
    max_retries = 1
    retry_delay = 3  # seconds
    
    for attempt in range(max_retries):
        try:
            response = requests.get(
                f'https://tiki.vn/api/v2/products/{pid}',
                headers=headers,
                params=params,
                cookies=cookies,
                timeout=10
            )
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    if data:
                        print(f'Crawl data {pid} success!')
                        return parser_product(data)
                except requests.exceptions.JSONDecodeError:
                    print(f"Invalid JSON response for product {pid}")
            elif response.status_code == 429:
                print(f"Rate limited. Waiting {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            else:
                print(f"Failed to fetch product {pid}. Status code: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"Request error for product {pid}: {str(e)}")
            
        if attempt < max_retries - 1:
            sleep_time = random.uniform(retry_delay, retry_delay * 2)
            print(f"Retrying in {sleep_time:.1f} seconds... (Attempt {attempt + 2}/{max_retries})")
            time.sleep(sleep_time)
    
    return None

def crawl():
    try:
        # Đọc danh sách ID sản phẩm
        # df_id = pd.read_csv('product_id.csv')
        # p_ids = df_id.id.to_list()
        p_ids = product_list
        print(f"Total products to crawl: {len(p_ids)}")
        
        # Crawl data
        result = []
        for pid in tqdm(p_ids, total=len(p_ids)):
            product_data = crawl_product(pid)
            if product_data:
                result.append(product_data)
            time.sleep(random.uniform(1, 3))  # Random delay between requests
        
        # Lưu kết quả
        if result:
            df_product = pd.DataFrame(result)
            df_product['date'] = datetime.now().date()
            ## define file name

            full_path = os.path.join(output_dir, output_file)
            os.makedirs(output_file, exist_ok=True)


            df_product.to_csv(full_path, index=False)

            
            print(f"Successfully crawled {len(result)} products")
        else:
            print("No products were successfully crawled")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# if __name__ == "__main__":
#     main()