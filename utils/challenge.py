from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(temperature=0)

def generate_questions(doc_text):
    prompt = PromptTemplate.from_template("""
    Based on the following document, generate three comprehension or logic-based questions to test user understanding.
    Document:
    {doc}
    """)
    return llm(prompt.format(doc=doc_text))

def evaluate_answer(question, user_answer, doc_text):
    prompt = PromptTemplate.from_template("""
    Evaluate the user's answer based on the document.

    Question: {q}
    User Answer: {ua}

    Refer to the document below and provide:
    - Correctness (Correct/Incorrect)
    - Brief feedback with reference to document

    Document:
    {doc}
    """)
    return llm(prompt.format(q=question, ua=user_answer, doc=doc_text))
