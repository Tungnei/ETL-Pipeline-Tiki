from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import aiofiles

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    # Đọc nội dung của file HTML
    async with aiofiles.open("dashboard-tiki.html", mode="r", encoding="utf-8") as file:
        html_content = await file.read()
    return HTMLResponse(content=html_content)
