from fastapi import FastAPI, status, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from pydantic import BaseModel
from vector_db import VectorDB

app = FastAPI(title="VectorDB API", description="API for interacting with the VectorDB service", version="1.0.0")
vector_db = VectorDB()

class QuestionRequest(BaseModel):
    database_id : str = None
    question: str
    chat_history: List[tuple] = []

class DatabaseRequest(BaseModel):
    database_id: str
    urls: List[str]

class DocumentRequest(BaseModel):
    database_id: str
    url: str

class DeleteDocument(BaseModel):
    database_id: str
    url: str

@app.get("/health/", response_model=dict, description="Health check endpoint.")
async def health() -> dict:
    """
    Health check endpoint.
    
    Returns:
    - dict: Status message indicating success.
    """
    return {"message": "OK"}

@app.post("/question/", response_model=dict, status_code=201, description="Submit a question and retrieve an answer.")
async def question(data: QuestionRequest) -> dict:
    """
    Submit a question for a specific database_id and get an answer.
    
    Args:
    - data (QuestionRequest): Input data containing database_id of the database and the question.
    
    Returns:
    - dict: Answer to the question based on the database.
    """
    try:
        answers = vector_db.get_answer(data.database_id, data.question, data.chat_history)
        return {"message": answers}
    except Exception as e:
        # Here, you can also log the error e for debugging purposes.
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/database/", response_model=dict, status_code=201, description="Add a database_id to the VectorDB.")
async def add_database(data: DatabaseRequest) -> dict:
    """
    Add new databases to the VectorDB with initial documents.
    
    Args:
    - data (DatabaseRequest): Input data containing database_id of the database and its URL.
    
    Returns:
    - dict: Status message indicating success or failure.
    """
    try:
        response = vector_db.add_database(data.database_id, data.urls)
        if "error" in response:
            return JSONResponse(status_code=status.HTTP_409_CONFLICT, content={"detail": response["error"]})
        return response
    except Exception as e:
        # Here, you can also log the error e for debugging purposes.
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Update with put a database by adding a document to it
@app.put("/add_document/", response_model=dict, status_code=201, description="Add a document to a database_id in the VectorDB.")
async def add_document(data: DocumentRequest) -> dict:
    """
    Add a document to a database_id in the VectorDB.
    
    Args:
    - data (DatabaseRequest): Input data containing database_id of the database and its URL.
    
    Returns:
    - dict: Status message indicating success or failure.
    """
    try:
        response = vector_db.add_document(data.database_id, data.url)
        if "error" in response:
            return JSONResponse(status_code=status.HTTP_409_CONFLICT, content={"detail": response["error"]})
        return response
    except Exception as e:
        # Here, you can also log the error e for debugging purposes.
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.delete("/delete_document/", response_model=dict, description="Delete a document from the VectorDB.")
async def delete_documents(body: DeleteDocument) -> dict:
    """
    Delete a document from a database_id from the VectorDB.

    Args:
    - database_id (str): ID of the database to delete from.
    - url (str): URL of the document to delete.

    Returns:
    - dict: Status message indicating success or failure.
    """
    try:
        response = vector_db.delete_document(body.database_id, body.url)
        if "error" in response:
            return JSONResponse(status_code=status.HTTP_409_CONFLICT, content={"detail": response["error"]})
        return response
    except Exception as e:
        # Here, you can also log the error e for debugging purposes.
        raise HTTPException(status_code=500, detail="Internal Server Error")
