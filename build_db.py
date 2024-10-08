# build db
import os
from langchain.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain.document_loaders import DirectoryLoader, MergedDataLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_community.vectorstores import FAISS # Chroma maybe flexible but narrow range
from langchain_openai import OpenAIEmbeddings
import dotenv

dotenv.load_dotenv()
# Create a document loader.

pdf_loader = DirectoryLoader("data_eqsans_exp",
                         glob='*.pdf',
                         loader_cls = PyPDFLoader)
txt_loader = DirectoryLoader("data_eqsans_exp",
                             glob='*.txt',
                             loader_cls = TextLoader)
'''
web_loader = WebBaseLoader(["https://neutrons.ornl.gov/eqsans",
                            "https://neutrons.ornl.gov/eqsans/team",
                            "https://neutrons.ornl.gov/eqsans/documentation",
                            "https://neutrons.ornl.gov/eqsans/publications",
                            "https://neutrons.ornl.gov/eqsans/mail-in",
                            "https://neutrons.ornl.gov/sites/default/files/EQSANS_spec_sheet.pdf",
                            "https://neutrons.ornl.gov/users/shipping-guide",
                            "https://neutrons.ornl.gov/users/onsite",
                            "https://neutrons.ornl.gov/users/after-experiment"
                            ])
'''
loader = MergedDataLoader(loaders=[pdf_loader, txt_loader]) #, web_loader])
docs = loader.load()

# Create a text splitter.
text_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=10000, chunk_overlap=2000)
texts = text_splitter.split_documents(docs)

# Create an embedding object.
#embeddings = SentenceTransformerEmbeddings() # by default, this uses all-MiniML-L6-v2 model
embeddings = OpenAIEmbeddings()
# Create a vector store.
vectorstore = FAISS.from_documents(texts, embeddings)

# Save the vector store to disk.
vectorstore.save_local("db/vectorstore_faiss_eqsans_exp")

print(len(texts))