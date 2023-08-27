# knowledge-chatbot

## Overview

The knowledge-chatbot is a service built using FastAPI and qdrant. The API allows you to interact with a vector database for handling document embeddings and retrieval.

## Table of Contents
- [knowledge-chatbot](#knowledge-chatbot)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Project Structure](#project-structure)
  - [API Endpoints](#api-endpoints)
  - [Setup and Running](#setup-and-running)
    - [Prerequisites](#prerequisites)
    - [Steps](#steps)
  - [Environment Variables](#environment-variables)
  - [Dependencies](#dependencies)

## Project Structure

- **main.py**: Contains the FastAPI application and all its endpoints for handling the VectorDB service.
  
- **vector_db.py**: Defines the VectorDB class and its associated methods for interacting with the Qdrant client and handling document embeddings and retrieval operations.

- **Dockerfile**: Contains the Docker configurations for building the FastAPI application.

- **docker-compose.yaml**: Defines the services for running the FastAPI application and the Qdrant client together in containers.

## API Endpoints

- **Health Check Endpoint**
    - Route: `/health/`
    - Method: GET
    - Description: Endpoint for checking the health status of the service.
```bash
curl -X GET "http://localhost:8000/health/" -H  "accept: application/json"
```

- **Add new Database to the vectorDB**
    - Route: `/database/`
    - Method: POST
    - Description: Add a new database to the VectorDB based on its ID and URLs.
    - Body: 
        - **database_id**: [String] [Requiered] The unique ID of the database to add.
        - **urls**: List[String] [Requiered] The URLs of the documents to add.
```bash 
curl -X POST "http://localhost:8000/database/" -H  "accept: application/json" -H  "Content-Type: application/json" -d '{"database_id":"1", "urls": ["https://en.wikipedia.org/wiki/Sweden", "https://en.wikipedia.org/wiki/Gothenburg"]}'
```

- **Submit a Question and Retrieve an Answer**
    - Route: `/question/`
    - Method: POST
    - Description: Submit a question for a specific document and get an answer.
    - Body: 
        - **question**: [String] [Requiered] The question to ask.
        - **database_id**: [String] [Requiered] The unique ID of the document to retrieve an answer for.
        - **chat_history**: List[List] [Optional] The chat history to use for the question answering.
```bash
curl -X POST "http://localhost:8000/question/" -H  "accept: application/json" -H  "Content-Type: application/json" -d '{"question":"And the second largest city?","database_id":"1", "chat_history": [["What is the capital of Sweden?", "Stockholm"]]}'
```

- **Add a Document to the VectorDB**
    - Route: `/add_document/`
    - Method: POST
    - Description: Add a new document to the VectorDB based on its unique ID and URL.
    - Body:
        - **database_id**: [String] [Requiered] The unique ID of the document to add.
        - **url**: [String] [Requiered] The URL of the document to add.
```bash
curl -X PUT "http://localhost:8000/add_document/" -H  "accept: application/json" -H  "Content-Type: application/json" -d '{"database_id":"1", "url": "https://en.wikipedia.org/wiki/Mahatma_Gandhi"}'
```

- **Delete a Document from the VectorDB**
    - Route: `/delete_document/`
    - Method: DELETE
    - Description: Delete a document from the VectorDB based on its unique ID and URL.
    - Body:
        - **database_id**: [String] [Requiered] The unique ID of the document to delete.
        - **url**: [String] [Requiered] The URL of the document to delete.
```bash
curl -X DELETE "http://localhost:8000/delete_document/" -H  "accept: application/json" -H  "Content-Type: application/json" -d '{"database_id":"1", "url": "https://en.wikipedia.org/wiki/Mahatma_Gandhi"}'
```

## Setup and Running

### Prerequisites

1. Docker
2. Docker Compose

### Steps

1. Build and start the services using Docker Compose:
```bash
docker-compose up --build
```
Once the services are running, the VectorDB API will be accessible at `http://localhost:8000`.

## Environment Variables

Make sure to set up the following environment variables in your `.env` file:

- **OPENAI_API_KEY**: Your OpenAI API key.
- **QDRANT_HOST**: Host for the Qdrant service. Use service name `qdrant` if running with Docker Compose (default is `qdrant`)
- **QDRANT_PORT**: Port for the Qdrant service.
- **COLLECTION_NAME**: Name for the Qdrant collection.

## Dependencies

Ensure that all required Python packages are listed in `requirements.txt`.
