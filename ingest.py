''' Here we use the inbuilt functions in langchain such as
"text_splitter" using RecursiveCharacterTextSplitter ==> splits at the character level(Byte wise),
"document_loader" for loading the pdf,
default "SentenceTransformerEmbeddings" for creating the sentence embeddings,
"Chroms" for vectordb'''

"""  We will load the pdf, give it to text splitter and then gove it to sentence transformer to create embeddings 
and then store it in the data base"""


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import os
from constants import CHROMA_SETTINGS


persist_directory = "db"  #specifying the database


def main():

    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PDFMinerLoader(os.path.join(root, file))
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    #create embeddings here
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    #create vector store
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    db.persist()  # saves
    db = None


if __name__ == "__main__":
    main()
