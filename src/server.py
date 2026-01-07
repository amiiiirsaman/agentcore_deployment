from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

from main import run_agent  # uses run_agent from main.py

app = FastAPI()

@app.get("/ping")
async def ping():
    # Health check endpoint required by AgentCore HTTP contract
    return {"status": "healthy"}

@app.post("/invocations")
async def invocations(request: Request):
    body = await request.json()
    query = body.get("query") or body.get("prompt") or ""
    if not query:
        return JSONResponse({"error": "query or prompt is required"}, status_code=400)

    result_str = run_agent(query)  # sync function returning JSON string
    return JSONResponse({"result": result_str})
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)