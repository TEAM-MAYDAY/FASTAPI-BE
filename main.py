import asyncio
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from routers import langchain

from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    # Should be edited in production env
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images", StaticFiles(directory="images"), name="images")

@app.get("/")
def read_root():
    return FileResponse("./templates/index.html")


class Office(BaseModel):
    name: str
    description: str

class OfficeList(BaseModel):
    offices: List[Office]

@app.post("/filter_office")
async def filter_office(officeData: OfficeList):

    filter_result = await asyncio.create_task(langchain.filter_office(
        officeData
    ))
    
    return filter_result

class Description(BaseModel):
    description: str
    answer1: str
    answer2: str
    answer3 : str
    job : str
    purpose : str

@app.post("/create_proposal")
async def create_proposal(req: Description):

    proposal_result = await asyncio.create_task(langchain.create_proposal(
        req.description,
        req.answer1,
        req.answer2,
        req.answer3,
        req.job,
        req.purpose
    ))
    
    return proposal_result

@app.post("/description_office")
async def description_office(officeData: OfficeList):

    proposal_result = await asyncio.create_task(langchain.description_office(
        officeData
    ))
    
    return proposal_result