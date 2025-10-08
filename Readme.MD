### NEWS BOT
Solution steps:
    1. Load input json file
    2. Create document chunks and store in chromadb
    3. Using prompt and RAG pipeline, query results are retrieved along with the references.

Steps to Run app 
    1. Create a python virtual environment, activate and install all the requirements.txt using pip install -r requirements.txt
    2. RUN python /src/data_ingestion.py
    3. RUN python app.py