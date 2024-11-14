import os
from datetime import datetime
from scripts.config import output_file

current_time = datetime.now().strftime("%d%m%y")
if os.listdir(output_file) == f'data_product_{current_time}.csv':
    print('Data is already crawled')
else :
    print('Data is not crawled yet')