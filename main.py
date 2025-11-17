import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from graph import research_chain
from models.api import EquityResearchRequest

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type"],
)


@app.get("/")
def ping():
    return {"message": "Running"}


@app.post("/research_equity")
async def research_equity(req: EquityResearchRequest):
    """Research provided ticker"""
    res = research_chain.invoke(req["ticker"])
    return res
