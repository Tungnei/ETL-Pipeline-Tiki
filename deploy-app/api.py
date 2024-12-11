from fastapi import FastAPI
import os
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/dashboard", response_class=HTMLResponse)
def read_root():
    # Specify the full path of the dashboard.html file
    file_path = r"D:\Source_project\ETL_Pipeline_Tiki\deploy-app\dashboard.html"
    
    # Ensure the file exists
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(content="File not found.", status_code=404)
