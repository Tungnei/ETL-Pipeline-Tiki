import requests
import pandas as pd
import time
import random
from tqdm import tqdm
from scripts.config import Config
cookies, headers, params = Config.get_config_product_data()

def comment_parser(json):
    d = dict()
    d["id"] = json.get("product_id")
    d["title"] = json.get("title")
    d["content"] = json.get("content")
    d["thank_count"] = json.get("thank_count")
    d["customer_id"] = json.get("customer_id")
    d["created_at"] = json.get("created_at")
    d["customer_name"] = json.get("created_by").get("name")
    d["purchased_at"] = json.get("created_by").get("purchased_at")
    return d


df_id = pd.read_csv("product_id.csv")
p_ids = df_id.id.to_list()
result = []
for pid in tqdm(p_ids, total=len(p_ids)):
    params["product_id"] = pid
    print("Crawl comment for product {}".format(pid))
    for i in range(2):
        params["page"] = i
        response = requests.get(
            "https://tiki.vn/api/v2/reviews",
            headers=headers,
            params=params,
            cookies=cookies,
        )
        if response.status_code == 200:
            print("Crawl comment page {} success!!!".format(i))
            for comment in response.json().get("data"):
                result.append(comment_parser(comment))
df_comment = pd.DataFrame(result)
df_comment.to_csv("comments_data.csv", index=False)
