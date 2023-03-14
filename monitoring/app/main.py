"""Main module."""

from fastapi import FastAPI
import uvicorn
from api.routers import router

app = FastAPI(title='Monitoramento de modelos', version="1.0.0")

@app.get("/")
def read_root():
    """Hello World message."""
    return {"Hello World": "from FastAPI"}


app.include_router(router, prefix="/v1")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)