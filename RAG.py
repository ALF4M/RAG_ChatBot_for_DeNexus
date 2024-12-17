import os
import pandas as pd
from groq import Groq
from uuid import uuid4
from langchain_core.documents import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document



class ChatBot():

  def __init__(self):
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
    self.model_name = 'sentence-transformers/all-mpnet-base-v2'
    self.carga_documentos()
    self.search_with_langchain_faiss()

  def carga_documentos(self):
    # Paso 2: Divide los csv en Documentos
    for titulo_documento, columns in self.csvs.items():
      df = pd.read_csv(f'data/{titulo_documento}_cleaned.csv')
      df = df[columns]
      if('id' not in columns):
        df['id'] = df.index

      for i, r in df.iterrows():
        self.documents.append(
            Document(
                page_content = r[0],
                metadata={
                    "source":titulo_documento,
                    "id":r['id']
                    }
                )
            )


  def search_with_langchain_faiss(self):
    # Paso 1: Configura el modelo de embeddings
    embeddings = HuggingFaceEmbeddings(model_name= self.model_name)

    try:
      print('Cargando base de datos')
      self.faiss_index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except:
      print('Base de datos no encontrada, generamos base de datos')
      # Paso 4: Carga los documentos en el índice FAISS usando from_documents
      self.faiss_index = FAISS.from_documents(self.documents, embedding=embeddings)
      self.faiss_index.save_local("faiss_index")

    print('Base de datos generada')


  def busca_contexto(self, query):
    # Paso 5: Realiza la búsqueda
    results = self.faiss_index.similarity_search(query, k= self.k)

    # Paso 6: Devuelve los resultados como texto
    return [result.page_content for result in results]


  def llamaResponse(self, query):
    client = Groq(
        # This is the default and can be omitted
        api_key= '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',
    )

    chat_completion = client.chat.completions.create(

        messages=[

            {
                "role": "system",

                "content": f"""

                You are an assitant designed to answer questions related to cibersecurity.
                Please, only answer the question with the context provided, not with your knowledge. If the context
                do not provide the answer to the user question just say 'I don't know'.

                The context is: {self.busca_contexto(query)}
                """
            },

            {

                "role": "user",

                "content": query,

            }

        ],

        model="llama-3.3-70b-versatile",

    )

    return chat_completion.choices[0].message.content
