from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(temperature=0)

def summarize_doc(text):
    prompt = PromptTemplate.from_template("""
    Summarize the following document in under 150 words maintaining key insights and conclusions.
    Document:
    {doc}
    """)
    return llm(prompt.format(doc=text))
