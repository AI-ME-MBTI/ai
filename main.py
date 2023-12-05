from typing import List
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi import status
from pydantic import BaseModel
import uvicorn

from common_mbti_prediction import extra_train_model, get_common_mbti, make_feedback_df
from specific_mbti_prediction import extra_train_specific_model, get_feedbackf, get_specific_mbti


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
    detail_answer: List[DetailAnswer]

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
    try:
        make_feedback_df(feedback.mbti, feedback.common_answer)
        get_feedbackf(feedback.detail_answer)
        
        
        return JSONResponse(
            status_code=status.HTTP_201_CREATED, 
            content={
                "statusCode": 201,
                "data": {
                    "message": ['피드백 데이터를 정상적으로 저장했습니다.'], 
                    "mbti": feedback.mbti,
                    "answer": feedback.answer
                }
            }
        )
        
    except:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            content={
                "statusCode": 500,
                "data": {
                    "message": [ '피드백 데이터를 저장하는 데 실패했습니다.']
                }
            }
        )
        
@app.post('/train')
def extra_train():
    try:
        common_is_successed = extra_train_model()
        detail_is_successed = extra_train_specific_model()

        if not common_is_successed:
            return JSONResponse(
                content={
                    "data": {
                        "message": [ '일반 질문 피드백 데이터가 적어 아직 훈련할 수 없습니다.']
                    }
                }
            )
        
        if not detail_is_successed:
            return JSONResponse(
                content={
                    "data": {
                        "message": [ '세부 질문 피드백 데이터가 적어 아직 훈련할 수 없습니다.']
                    }
                }
            )
            
        return JSONResponse(
                status_code=status.HTTP_201_CREATED, 
                content={
                    "statusCode": 201,
                    "data": {
                        "message": ['피드백 데이터를 정상적으로 훈련시켰습니다.']
                    }
                }
            )
    except:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            content={
                "statusCode": 500,
                "data": {
                    "message": [ '피드백 데이터를 훈련하는 데 실패했습니다.']
                }
            }
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    