import streamlit as st
import langchain
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

# 1. Vectorise the sales response csv data
loader = TextLoader(file_path="article.txt")
article = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(article, embeddings)

# 2. Function for similarity search


def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)

    page_contents_array = [doc.page_content for doc in similar_response]

    # print(page_contents_array)

    return page_contents_array


# 3. Setup LLMChain & prompts
llm = ChatOpenAI(temperature=1, model="gpt-3.5-turbo-16k-0613")

template = """
You are a world class response giver. 
I will share a article with you and you will give me the best answer that 
I should send to this article based on the article chunks, 

Below is the question I'll ask:
{question}

Here is the best possible approaches answer from ths only:
{article_chunks}

Please write the best response for me:
"""

prompt = PromptTemplate(
    input_variables=["question", "article_chunks"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)


# 4. Retrieval augmented generation
def generate_response(question):
    article_chunks = retrieve_info(question)
    response = chain.run(question=question, article_chunks=article_chunks)
    return response


# 5. Build an app with streamlit
def main():
    st.set_page_config(
        page_title="Q&A page", page_icon=":bird:")

    st.header("Q&A Box :bird:")
    question = st.text_area("question")

    if question:
        st.write("Generating best answer...")

        result = generate_response(question)

        st.info(result)


if __name__ == '__main__':
    main()
