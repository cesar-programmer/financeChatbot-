from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

print("Ingesting data")
# Define el modelo de embeddings a utilizar (en este caso, un modelo de HuggingFace).
DATA_PATH = "data/"
# Define la ruta donde se almacenará la base de datos de vectores (en este caso, FAISS).
DB_FAISS_PATH = "vectorstore/db_faiss"

# Función que crea una base de datos de vectores.
def create_vector_db():
  print("Creating vector database")
  # Carga documentos de un directorio y extrae el texto de los mismos usando PyPDFLoader.
  loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
  # Carga los documentos especificados.
  documents = loader.load()
  # Crea un objeto que divide el texto en segmentos de un tamaño específico con un solapamiento determinado.
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
  # Divide los documentos en segmentos de texto.
  texts = text_splitter.split_documents(documents)

  # Crea un objeto de embeddings usando un modelo preentrenado de HuggingFace.
  embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
  model_kwargs={"device": "cpu"})

  # Crea una base de datos FAISS con los textos y sus correspondientes embeddings.
  db = FAISS.from_documents(texts, embeddings)
  # Guarda localmente la base de datos de vectores.
  db.save_local(DB_FAISS_PATH)

# Punto de entrada principal que llama a create_vector_db si el script se ejecuta directamente.
if __name__ == "__main__":
  create_vector_db()
