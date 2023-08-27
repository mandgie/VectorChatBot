# Standard library imports
import os
import logging

# Third-party imports
from typing import Optional
from dotenv import load_dotenv
import openai
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain.vectorstores import Qdrant
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredURLLoader, PyPDFLoader
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

# Constants
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 8000))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "default_collection")

PROMPT_TEMPLATE = """
Answer the question in your own words as truthfully as possible from the context given to you.
If you do not know the answer to the question, simply respond with "I don't know. Can you ask another question".
If questions are asked where there is no relevant context available, simply respond with "I don't know. Please ask a question relevant to the documents"
Context: {context}


{chat_history}
Human: {question}
Assistant:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(PROMPT_TEMPLATE)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDB:
    """Vector database manager for handling database embedding and retrieval."""

    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    def __init__(self):
        """Initialize VectorDB with embeddings, splitter, and Qdrant client."""
        self.embedding = OpenAIEmbeddings()
        self.text_splitter = CharacterTextSplitter(
            chunk_size=self.CHUNK_SIZE,
            chunk_overlap=self.CHUNK_OVERLAP,
            length_function=len
        )
        self.qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.vector_db = self._initialize_client()

    def _initialize_client(self) -> Qdrant:
        """Initialize or retrieve the Qdrant client."""
        collections = self.qdrant_client.get_collections()
        if COLLECTION_NAME not in collections:
            test_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
            loader = PyPDFLoader(test_url)
            databases = loader.load_and_split()
            dbs = self.text_splitter.split_documents(databases)
            qdrant = Qdrant.from_documents(dbs, 
                                           self.embedding,
                                           collection_name=COLLECTION_NAME,
                                           host=QDRANT_HOST, 
                                           port=QDRANT_PORT)
            return qdrant
        else:
            logger.info('Collection exists')
            return Qdrant(self.qdrant_client, COLLECTION_NAME, host=QDRANT_HOST, port=QDRANT_PORT)

    def add_database(self, database_id: str, urls: list[str]):
        """
        Add database to the Qdrant collection based on the given ID and URLS.
        
        :param database_id: The database ID.
        :param urls: The URLS of the documents.
        :return: A dictionary containing the result.
        """
        if self._search_by_database_id(COLLECTION_NAME, database_id):
            logger.info(f'ID: {database_id} already exists')
            return {"error": "ID already exists"}
        else:
            self._add_document(database_id, urls)
            logger.info(f'ID: {database_id} added')
            return {"success": f"Documents added for ID: {database_id}"}

    # Add document
    def add_document(self, database_id: str, url: str):
        """
        Add a document to the Qdrant collection based on the given database ID and URL.
        
        :param database_id: The database ID.
        :param url: The URL of the document.
        :return: A dictionary containing the result.
        """
        if self._search_by_database_id(COLLECTION_NAME, database_id):
            self._add_document(database_id, [url])
            logger.info('Document added')
            return {"success": "Document added"}
        else:
            logger.info('Database does not exist')
            return {"error": "Database does not exist"}
        
    def _add_document(self, database_id: str, url: list[str]):
        """Add a document based on its database ID and URL."""
        loader = UnstructuredURLLoader(url)
        documents = loader.load()
        docs = self.text_splitter.split_documents(documents)

        for doc in docs:
            doc.metadata['database_id'] = database_id

        self.vector_db.add_documents(docs)

    def get_answer(self, database_id: Optional[str], query: str, chat_history: list) -> dict:
        """
        Retrieve an answer from a database given a query.
        
        :param database_id: The database ID. (Optional)
        :param query: The query string.
        :return: A dictionary containing the result.
        """
        search_kwargs = {}

        if database_id:
            if not self._search_by_database_id(COLLECTION_NAME, database_id):
                logger.info(f'Database with ID {database_id} does not exist')
                return {"error": "ID does not exist"}
            search_kwargs["filter"] = {"database_id": database_id}
            
            # Convert chat_history from list of lists to list of tuples
        chat_history_tuples = [(x[0], x[1]) for x in chat_history]
        qa_chain = self._get_qa_chain(search_kwargs)
        return qa_chain({"question": query, "chat_history": chat_history_tuples})
    
    def _get_qa_chain(self, search_kwargs: dict) -> RetrievalQA:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        return ConversationalRetrievalChain.from_llm(
            llm=llm, 
            retriever=self.vector_db.as_retriever(search_kwargs=search_kwargs),
            combine_docs_chain_kwargs={'prompt': QA_CHAIN_PROMPT})


    def _search_by_database_id(self, collection_name: str, database_id_value: str) -> bool:
        """
        Search a database by its database ID.
        
        :param collection_name: Name of the collection.
        :param database_id_value: Value of the database ID.
        :return: True if the database exists, False otherwise.
        """
        results = self.qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.database_id",
                    match=models.MatchValue(value=database_id_value),
                )
            ])
        )

        return len(results[0]) > 0
    
    def delete_document(self, database_id: str, url: str):
        """
        Delete a document from the Qdrant collection based on the given database ID and url.
        
        :param database_id: The database ID.
        :param url: The URL of the document.
        :return: A dictionary containing the result.
        """
        if self._search_by_database_id(COLLECTION_NAME, database_id):
            self._delete_document(database_id, url)
            logger.info('Database deleted')
            return {"success": f"Document with url: {url} deleted"}
        else:
            logger.info('Database does not exist')
            return {"error": "Database does not exist"}
        
    def _delete_document(self, database_id, url):
        """Delete a document based on its database ID and url."""
        self.qdrant_client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.database_id",
                            match=models.MatchValue(value=database_id),
                        ),
                        models.FieldCondition(
                            key="metadata.source",
                            match=models.MatchValue(value=url),
                        ),
                    ],
                )
            ),
        )
