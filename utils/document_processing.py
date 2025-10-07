from langchain.schema import Document
import json

def document_processing(data_path):
    # loading the JSON document
    with open(data_path, "r") as f:
        data = json.load(f)

    # flatten the data
    records = [item for _, arr in data.items() for item in arr]

    #documents creation 
    docs = [
        Document(
            page_content=rec["full_text"],
            metadata={"title": rec["title"], "link": rec["link"], "ticker": rec["ticker"]},
        )
        for rec in records
    ]

    return docs