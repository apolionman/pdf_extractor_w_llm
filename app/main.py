from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse
import os, json, uuid, shutil, tempfile, httpx, re, asyncio, requests
from app.services.llm_extract import *
from uuid import UUID

# routes
from app.routes.endpoints import router as endpoints_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(endpoints_router, prefix="/api/v1", tags=["Backend Endpoints"])

@app.get("/api/v1/health")
async def health():
    return {"status": "ok"}