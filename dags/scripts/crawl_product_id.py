import requests
import time
import random
import pandas as pd
from tqdm import tqdm
from config import Config
cookies, headers, params = Config.get_config_product_id()

class CrawlProductID:
    def __init__(self):
        self.cookies = cookies
        self.headers = headers
        self.params = params

    def extract_product_id(self):
        product_id = []
        for i in range(1, 11):
            self.params["page"] = i
            response = requests.get(
                "https://tiki.vn/api/personalish/v1/blocks/listings",
                headers=self.headers,
                params=self.params,
                cookies=self.cookies,
            )
            if response.status_code == 200:
                print(f"request success!!!{i}")
                for record in response.json().get("data"):
                    product_id.append({"id": record.get("id")})
            time.sleep(random.randrange(3, 10))
        df = pd.DataFrame(product_id)
        df.drop_duplicates(subset=["id"], inplace=True)
        df.to_csv("product_id.csv", index=False)


if __name__ == "__main__":
    crawler = CrawlProductID()
    crawler.extract_product_id()
