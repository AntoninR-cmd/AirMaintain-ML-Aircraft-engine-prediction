from fastapi import FastAPI

app = FastAPI(
    title="AeroMaintain API",
    version= "0.1.0",
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/version")
def version():
    return {"version": "0.1.1"}

@app.get("/engine/{engine_id}")
def engine_id(engine_id : int):
    return {
        "IdMoteur": engine_id
    }