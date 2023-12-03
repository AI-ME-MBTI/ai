from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi import status
from pydantic import BaseModel
import uvicorn

from common_mbti_prediction import common_train_model, extra_train_model, get_common_mbti
from specific_mbti_prediction import get_specific_mbti, specific_train_model


app = FastAPI()

# @app.exception_handler()
def bad_request_exception(answer: str):
    error_list = []

    if not answer:
        error_list.append('답변은 필수로 입력해야 합니다.')
    if not isinstance(answer, str):
        error_list.append('답변은 문자열 형식이어야 합니다.')
        
    if len(error_list) != 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST, 
            content={"message": error_list}
        )
    else:
        return

class Answer(BaseModel):
    answer: str

class MbtiAnswer(BaseModel):
    mbti_type: str
    answer: str
    
class DetailAnswer(BaseModel):
    detail_mbti: str
    answer: str

class Feedback(BaseModel):
    mbti: str
    common_answer: str
    detail_answer: list(DetailAnswer)
    
@app.post("/answer/common")
def get_common_answer(user_answer: Answer):
    bad_request_exception(user_answer.answer)
    try:
        mbti = get_common_mbti(user_answer.answer)
            
        return JSONResponse(
            status_code=status.HTTP_201_CREATED, 
            content={
                "statusCode": 201,
                "data": {
                    "message": ['MBTI 결과가 정상적으로 나왔습니다.'], 
                    "mbti": mbti
                }
            }
        )
    
    except:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            content={
                "statusCode": 500,
                "data": {
                    "message": ['MBTI 받아오는 데 실패했습니다.'], 
                }
            }
        )
   
@app.post('/answer/specific')
def get_specific_answer(user_answer: MbtiAnswer):
    bad_request_exception(user_answer.answer)
    
    try:
        mbti = get_specific_mbti(user_answer.mbti_type, user_answer.answer)
        
        return JSONResponse(
            status_code=status.HTTP_201_CREATED, 
            content={
                "statusCode": 201,
                "data": {
                    "message": ['MBTI 질문에 대한 결과가 정상적으로 나왔습니다.'], 
                    "mbti": mbti
                }
            }
        )
    
    except:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            content={
                "statusCode": 500,
                "data": {
                    "message": [ 'MBTI 받아오는 데 실패했습니다.']
                }
            }
        )

@app.post('/feedback')
def get_feedback(feedback: Feedback):
    common_is_success = extra_train_model(feedback.mbti, [feedback.common_answer])
    
    specific_is_success = specific_train_model(feedback.detail_answer)
    
    if common_is_success & specific_is_success:
        return JSONResponse(
            status_code=status.HTTP_201_CREATED, 
            content={
                "statusCode": 201,
                "data": {
                    "message": ['정상적으로 데이터가 훈련됐습니다.'], 
                    "mbti": feedback.mbti,
                    "answer": feedback.answer
                }
            }
        )
        
    else:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            content={
                "statusCode": 500,
                "data": {
                    "message": [ '모델을 훈련시키는데 실패했습니다.']
                }
            }
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    