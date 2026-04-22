import pandas as pd
import requests
import time
import random
from tqdm import tqdm
from datetime import datetime
from scripts.config import product_list, output_file, output_dir
import os
from scripts.config import get_config_product_data

cookies, headers, params = get_config_product_data()


def parser_product(json):
    try:
        d = dict()
        d["id"] = json.get("id")
        d["product_name"] = json.get("name")
        d["brand_name"] = json.get("brand", {}).get("name")
        d["short_description"] = json.get("short_description")
        d["original_price"] = json.get("original_price")
        d["price_after_voucher"] = json.get("price")
        d["discount_rate"] = json.get("discount_rate")
        d["discount_price"] = json.get("discount")

        quantity_sold = json.get("quantity_sold")
        d["quantity_sold"] = quantity_sold.get("value") if quantity_sold else 0

        d["rating_average"] = json.get("rating_average")
        d["review_count"] = json.get("review_count")

        warranty_info = json.get("warranty_info", [])
        d["warranty_info"] = (
            warranty_info[0].get("value") if warranty_info else "No warranty info"
        )

        return_policy = json.get("return_policy")
        d["return_policy"] = (
            return_policy.get("title") if return_policy else "No return policy"
        )

        images = json.get("images", [])
        d["url_img"] = images[0].get("base_url") if images else None

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
                f"https://tiki.vn/api/v2/products/{pid}",
                headers=headers,
                params=params,
                cookies=cookies,
                timeout=10,
            )

            if response.status_code == 200:
                try:
                    data = response.json()
                    if data:
                        print(f"Crawl data {pid} success!")
                        return parser_product(data)
                except requests.exceptions.JSONDecodeError:
                    print(f"Invalid JSON response for product {pid}")
            elif response.status_code == 429:
                print(f"Rate limited. Waiting {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            else:
                print(
                    f"Failed to fetch product {pid}. Status code: {response.status_code}"
                )

        except requests.exceptions.RequestException as e:
            print(f"Request error for product {pid}: {str(e)}")

        if attempt < max_retries - 1:
            sleep_time = random.uniform(retry_delay, retry_delay * 2)
            print(
                f"Retrying in {sleep_time:.1f} seconds... (Attempt {attempt + 2}/{max_retries})"
            )
            time.sleep(sleep_time)
    return None


def crawl():
    try:
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
            df_product["date"] = datetime.now().date()
            ## define file name

            full_path = os.path.join(output_dir, output_file)
            os.makedirs(output_dir, exist_ok=True)
            df_product.to_csv(full_path, index=False)
            print(f"Successfully crawled {len(result)} products")
        else:
            print("No products were successfully crawled")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


# if __name__ == "__main__":
#     main()
