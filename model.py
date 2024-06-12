from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

DB_FAISS_PATH = "vectorstore/db_faiss"

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
  """
  Prompt template for QA retrieval for each vector stores.
  """
  prompt = PromptTemplate(template=custom_prompt_template,
                          input_variables=["context", "question"])
  return prompt

# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                      chain_type='stuff',
                                      retriever=db.as_retriever(search_kwargs={'k': 2}),
                                      return_source_documents=True,
                                      chain_type_kwargs={'prompt': prompt}
                                      )
    return qa_chain

def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm

def qa_bot():
  embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
                                      model_kwargs={"device": "cpu"})
  # Permitir deserialización peligrosa al cargar FAISS
  db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
  llm = load_llm()
  qa_prompt = set_custom_prompt()
  qa_chain = retrieval_qa_chain(llm, qa_prompt, db)
  return qa_chain

def final_result(query):
  qa_result = qa_bot()
  response = qa_result({'query': query})
  return response

## ChainLit ##
@cl.on_chat_start
async def on_chat_start():
    chain = qa_bot()
    msg = cl.Message(content='starting chat bot ....')
    await msg.send()
    msg.content = 'hi Welcome to the chat bot what is your query?'
    await msg.update()
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer).send()
