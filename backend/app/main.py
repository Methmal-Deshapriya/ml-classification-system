from fastapi import FastAPI

app = FastAPI(
    title="ML Classification System",
    description="A simple ML classification system",
    version="1.0.0",
)

@app.get("/")
def home():
    x = 10
    y = 5
    
    return {"message": x+y}

@app.get("/health")
def health():
    return {"status": "ok"}