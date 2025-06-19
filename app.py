import os
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeStore
import pinecone

# 📦 Variables de entorno (ya configuradas en Railway)
openai_api_key = os.environ["OPENAI_API_KEY"]
pinecone_api_key = os.environ["PINECONE_API_KEY"]
pinecone_env = os.environ["PINECONE_ENVIRONMENT"]
index_name = os.environ["INDEX_NAME"]

# 🚀 Inicializar Pinecone
pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

# 🧠 Crear el índice si no existe
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536, metric="cosine")

# 📂 Ruta donde Railway verá los archivos (ya están en el repo)
base_dir = "tmp_docs"

# 📄 Lista de documentos
archivos = [
    "Manual de Cumplimiento - VIZUM.docx",
    "Metodología PLD-FT - VIZUM.docx",
    "Matriz de Riesgos Clientes - VIZUM.xlsx",
    "Metodología EBR - VIZUM.xlsx"
]

# 🧾 Cargar documentos
docs = []
for nombre in archivos:
    path = os.path.join(base_dir, nombre)
    print(f"📄 Procesando {nombre}...")
    try:
        if path.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(path)
        elif path.endswith(".xlsx"):
            loader = UnstructuredExcelLoader(path)
        else:
            print(f"❌ Tipo de archivo no soportado: {nombre}")
            continue
        docs.extend(loader.load())
        print(f"✅ Cargado: {nombre}")
    except Exception as e:
        print(f"❌ Error al cargar {nombre}: {e}")

print(f"📄 Total de documentos cargados: {len(docs)}")

# ✂️ Dividir en chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)
print(f"📚 Total de chunks generados: {len(chunks)}")

# 📤 Enviar a Pinecone
print("🔗 Enviando a Pinecone...")
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = PineconeStore.from_documents(chunks, embeddings, index_name=index_name)
print("✅ Documentos cargados exitosamente en Pinecone.")