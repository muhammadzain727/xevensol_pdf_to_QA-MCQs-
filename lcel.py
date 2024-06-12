import os
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from data_creation import text_loader
from dotenv import load_dotenv
from operator import itemgetter
from langchain_mistralai.chat_models import ChatMistralAI
import random
load_dotenv()
def qa_ret(file_path):
    try:
        context=[]
        #OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")
        MISTRAL_API_KEY=os.environ.get("MISTRAL_API_KEY")
        text=text_loader(file_path)
        filter_text=[item for item in text if item != ""]
        text_len=len(filter_text)
        if text_len > 50:
            random_context= random.sample(range(len(filter_text)), 5)
            for index in random_context:
                context.append(filter_text[index])
        else:
            random_context= random.sample(range(len(filter_text)), text_len)
            for index in random_context:
                context.append(filter_text[index])


        template = """Must generate 10 multiple-choice questions (MCQs) directly from the provided {context}. Do not include any introductory statements or additional text. Each question should be followed by four options and the correct answer.
        Format each question as follows:
        {{
            "question": "Your question here",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "answer": "Correct Option"
        }}
        Strictly ensure the entire response is a valid JSON object. Start generating the MCQs directly below:
        If the provided {context} is insufficient than return a JSON response error with:
        {{
            "error": "Sorry! the provided context is insufficient"
        }}
        """
        prompt = ChatPromptTemplate.from_template(template)
        setup_and_retrieval = {"context": itemgetter("context")}
                
        #llm = ChatOpenAI(model = "gpt-3.5-turbo", openai_api_key = OPENAI_API_KEY, temperature=0.3)
        llm = ChatMistralAI(api_key=MISTRAL_API_KEY,model_name="mistral-large-latest")
        output_parser= StrOutputParser()
        rag_chain = (
        setup_and_retrieval
        | prompt
        | llm
        | output_parser
        )
        respone=rag_chain.invoke({"context":context})
        return respone
    except Exception as ex:
        return ex
    


# Response in JSON :
#         instructions:
#         1. Do not make any single change in the format use "Evaluation", "question", "options" and "answer" key words as it is.
#         2. You must have to generate 4 options a) , b) , c) and d)
#         3.You must have to generate 10 questions if the {context} is less than rephrase the above generated question that is very tricky to answer.
#         4.Do not go out of the context.
#         5.If you find anything empty in the {context} than also rephrase the above generated question that is very tricky to answer.
#         6. You are strickly instructed to not to include any introductory statements or additional text.