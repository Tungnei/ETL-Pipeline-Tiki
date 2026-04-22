import requests
import time
import random
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from config import Config

# Lấy cấu hình từ file config
cookies, headers, params = Config.get_config_product_id()


class CrawlProductID:
    def __init__(self):
        self.cookies = cookies
        self.headers = headers
        self.params = params
        self.base_url = "https://tiki.vn/api/personalish/v1/blocks/listings"

    def fetch_page(self, page):
        """Hàm xử lý việc cào dữ liệu cho duy nhất 1 trang"""
        # Tạo bản sao của params để tránh xung đột dữ liệu giữa các luồng
        current_params = self.params.copy()
        current_params["page"] = page

        ids = []
        try:
            # Thêm timeout để tránh luồng bị treo vĩnh viễn
            response = requests.get(
                self.base_url,
                headers=self.headers,
                params=current_params,
                cookies=self.cookies,
                timeout=10,
            )

            if response.status_code == 200:
                print(f"Trang {page}: Request thành công!")
                data = response.json().get("data", [])
                for record in data:
                    ids.append({"id": record.get("id")})
            else:
                print(f"Trang {page}: Lỗi {response.status_code}")

            # Nghỉ ngắn để tránh bị Tiki block (giảm xuống vì chạy đa luồng)
            time.sleep(random.uniform(1, 3))

        except Exception as e:
            print(f"Trang {page}: Gặp lỗi: {e}")

        return ids

    def extract_product_id_multithread(self, total_pages=10, max_workers=5):
        product_id_all = []

        # Sử dụng ThreadPoolExecutor để quản lý luồng
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Gửi các tác vụ vào các luồng
            pages = range(1, total_pages + 1)
            results = list(executor.map(self.fetch_page, pages))

        # Gộp kết quả từ các luồng lại
        for res in results:
            product_id_all.extend(res)

        # Lưu file
        df = pd.DataFrame(product_id_all)
        df.drop_duplicates(subset=["id"], inplace=True)
        df.to_csv("product_id.csv", index=False)
        print(f"Đã lưu {len(df)} ID vào file product_id.csv")


if __name__ == "__main__":
    crawler = CrawlProductID()
    start_time = time.time()

    # Chạy với 5 luồng song song
    crawler.extract_product_id_multithread(total_pages=10, max_workers=5)

    print(f"Thời gian hoàn thành: {time.time() - start_time:.2f} giây")
