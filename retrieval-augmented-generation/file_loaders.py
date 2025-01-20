from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PythonLoader

def pdf_loader(file):
    pdf_loader_agent = PyPDFLoader(file.file)
    documents = pdf_loader_agent.load()

def csv_loader(file):
    csv_loader_agent = CSVLoader(file.file)
    documents = csv_loader_agent.load()

def python_loader(file):
    python_loader_agent = PythonLoader(file.file)
    documents = python_loader_agent.load()