import os
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

load_dotenv()

class ChatBot:
    """
    ChatBot que utiliza FAISS, embeddings de HuggingFace y Groq LLM
    para responder preguntas basadas en contexto proveniente de CSVs.
    """

    def __init__(self):
        """
        Inicializa el chatbot cargando documentos y creando o recuperando
        un índice FAISS para búsqueda semántica.
        """
        self.csvs = {
            'CISSM': ['event_description'],
            'HACKMAGEDDON': ['Description'],
            'ICSSTRIVE': ['description'],
            'KONBRIEFING': ['description'],
            'TISAFE': ['attack_details', 'id'],
            'WATERFALL': ['incident_summary', 'id']
        }
        self.k = 5
        self.documents = []
        self.model_name = os.getenv('MODEL_NAME')
        self.faiss_index_path = os.getenv("FAISS_INDEX_PATH", "faiss_index")
        self.data_dir = os.getenv("DATA_DIR", "data")
        self._load_documents()
        self._load_faiss()

    def _load_documents(self):
        """
        Carga los datos desde CSVs y los convierte en objetos Document
        compatibles con LangChain.
        """
        for doc_name, columns in self.csvs.items():
            df = pd.read_csv(os.path.join(self.data_dir, f'{doc_name}_cleaned.csv'))
            df = df[columns]
            if 'id' not in columns:
                df['id'] = df.index
            for _, row in df.iterrows():
                self.documents.append(Document(
                    page_content=row.iloc[0],
                    metadata={"source": doc_name, "id": row['id']}
                ))

    def _load_faiss(self):
        """
        Carga un índice FAISS si existe, o lo crea a partir de los documentos.
        """
        embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        try:
            print('Cargando índice FAISS')
            self.faiss_index = FAISS.load_local(
                self.faiss_index_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
        except:
            print('Índice FAISS no encontrado, creando nuevo')
            self.faiss_index = FAISS.from_documents(self.documents, embedding=embeddings)
            self.faiss_index.save_local(self.faiss_index_path)

    def _search_context(self, query: str) -> list[str]:
        """
        Realiza una búsqueda de contexto en el índice FAISS.

        Args:
            query (str): Consulta del usuario.

        Returns:
            list[str]: Lista de textos relevantes.
        """
        results = self.faiss_index.similarity_search(query, k=self.k)
        return [r.page_content for r in results]

    def llama_response(self, query: str) -> str:
        """
        Llama al modelo de lenguaje de Groq para generar una respuesta con contexto.

        Args:
            query (str): Pregunta del usuario.

        Returns:
            str: Respuesta generada por el modelo.
        """
        client = Groq(api_key=os.getenv('API_KEY'))
        context = self._search_context(query)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an assistant designed to answer questions related to cybersecurity. "
                    "Please, only answer the question with the context provided. If the context "
                    "does not provide the answer to the user question, just say 'I don't know'.\n\n"
                    f"Context:\n{context}"
                )
            },
            {"role": "user", "content": query}
        ]

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages
        )

        return response.choices[0].message.content
