import streamlit as st
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage
import time

def criar_vectorstore():
    if "vectorstore" not in st.session_state:
        load_dotenv()
        loader = CSVLoader(file_path=r"C:\Users\gabri\Desktop\Gabriel\UCB\6 Sem\Startups\ia_alfred\pasta_auxiliar\teste_base_dados_tratada.csv", encoding="utf-8")
        loader.load()
        documents = loader.load()

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(documents, embeddings)
        st.session_state.vectorstore = vectorstore
        st.session_state.embeddings = embeddings
        st.session_state.documents = documents

def main():
    def resposta_stream(resposta):
        for word in resposta.split():
            yield word + " "
            time.sleep(0.1)

    st.set_page_config(page_title="Alfred AI", page_icon='imagem_mordomo.png')
    st.header('Alfred AI')

    criar_vectorstore()

    if "retriever" not in st.session_state:
        st.session_state.retriever = st.session_state.vectorstore.as_retriever()

    if "llm" not in st.session_state:
        st.session_state.llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.5)

    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = (
            """"
        Você é um assistente amigável e educado em um restaurante.
        Abaixo está uma pergunta de um cliente sobre o menu.
        Responda de forma detalhada, mencionando ingredientes e preços.
        Caso o que o cliente pediu ou perguntou não esteja no menu, deixe claro para ele, porém sempre de forma educada e sugerindo opções alternativas.
        NÃO fale sobre pratos ou bebidas que não estão no menu.
        {context}
        """
        )

    if "contextualize_q_system_prompt" not in st.session_state:
        st.session_state.contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", st.session_state.contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    if "rag_chain" not in st.session_state:
        history_aware_retriever = create_history_aware_retriever(
            st.session_state.llm, st.session_state.retriever, contextualize_q_prompt
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", st.session_state.system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )


        question_answer_chain = create_stuff_documents_chain(st.session_state.llm, qa_prompt)

        st.session_state.rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question := st.chat_input("Olá! Como posso ajudá-lo hoje?"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
    
        with st.chat_message("assistant"):
            response = st.session_state.rag_chain.invoke({"input": question, "chat_history": st.session_state.chat_history})

            st.session_state.chat_history.extend(
            [
                HumanMessage(content=question),
                AIMessage(content=response["answer"]),
            ]
        )
            st.write_stream(resposta_stream(response["answer"]))

            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
if __name__ == "__main__":

    main()

