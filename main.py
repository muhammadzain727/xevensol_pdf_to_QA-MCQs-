from fastapi import FastAPI,HTTPException,File, UploadFile
import tempfile
from data_creation import text_loader
from lcel import qa_ret
import json
from pydantic import BaseModel
from typing import List
import uvicorn
app = FastAPI()

class InputText(BaseModel):
    texts: List[str]

final_results = []

@app.post("/generate-mcqs")
async def generate_mcqs(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(await file.read())
            file = temp_file.name
        results = qa_ret(file)
        results_json = json.loads(results)
        final_results.append(results_json)
        return results_json
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))
    


@app.post("/scoring_mcqs")
async def score_evaluation(input_text: InputText):
    try:
        answers=[]
        correct_answers = []
        total_score=0
        # data = json.load(final_results)
        for mcq_data in final_results:
            for mcq in mcq_data:
                answers.append(mcq["answer"])
        for text in input_text.texts:
            correct_answers.append(text)
        for ans, correct_ans in zip(answers, correct_answers):
            if ans == correct_ans:
                total_score += 1
        return {"Total_score": total_score}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))

    
# @app.post("/process_texts/")
# async def process_texts(text_data: InputText):
#     processed_texts = []
#     for text in text_data.texts:
#         # Process each text here, for example, you can just append it to a list
#         processed_texts.append({"original_text": text,})
#     return {"processed_texts": processed_texts}


if __name__== "__main__":
    uvicorn.run("main:app",host="127.0.0.1",port=1080)
    