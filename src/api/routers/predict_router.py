from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from api.extensions.dependencies import get_strategy
from api.extensions.rate_limiter import limiter
from application.schemas.ner_request import NERRequest
from application.schemas.ner_response import NERResponse
from application.handlers.recognition_handler import RecognitionHandler
from domain.abstractions import ModelStrategy

router = APIRouter()

@router.post("/predict", response_model=NERResponse)
@limiter.limit("5/minute") 
async def predict_text(
    request: Request,
    req: NERRequest,
    strategy: ModelStrategy = Depends(get_strategy)
):
    handler = RecognitionHandler(strategy)
    return await handler.handle_prediction(req)

@router.post("/predict/document", response_model=NERResponse)
@limiter.limit("3/minute")
async def predict_document(
    request: Request,
    file: UploadFile = File(...),
    strategy: ModelStrategy = Depends(get_strategy)
):
    handler = RecognitionHandler(strategy)
    try:
        return await handler.handle_document_prediction(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc