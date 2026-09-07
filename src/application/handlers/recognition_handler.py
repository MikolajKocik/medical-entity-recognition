from domain.abstractions import Context, ModelStrategy
from application.schemas.ner_request import NERRequest
from application.schemas.ner_response import NERResponse
from application.utils.text import extract_text_chunks

from fastapi import UploadFile

class RecognitionHandler():
    def __init__(self, strategy: ModelStrategy):
        self.context = Context(strategy)
        
    async def handle_prediction(self, req: NERRequest) -> NERResponse:
        """
        Handles simple text prediction
        """
        return await self.context.predict_disease(req)

    async def handle_document_prediction(self, file: UploadFile) -> NERResponse:
        """
        Extracts document chunks, predicts each chunk, and restores document offsets.
        """
        entities = []

        for chunk, offset in extract_text_chunks(file):
            response = await self.context.predict_disease(NERRequest(text=chunk))

            for entity in response.entities:
                entity.start += offset
                entity.end += offset
                entities.append(entity)

        return NERResponse(entities=entities)