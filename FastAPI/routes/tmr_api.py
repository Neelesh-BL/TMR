from fastapi import FastAPI, Path, Depends
from prediction_tmr import tmr_main
from fastapi import APIRouter
from models.features import Features
# from typing import Optional
# from pydantic import BaseModel

prediction = APIRouter()



@prediction.get('/')
async def welcome():
    return {"detail":"Welcome to Bridgelabz TMR"}


@prediction.post("/test_model")
async def test_model(params: Features = Depends()):
    last_internal_rating = ['Unsatisfactory','Improvement needed','Meets expectations','Exceeds expectations','Exceptional']
    practice_head = ['Ashish','Sunil','Gunjan','Dilip','Nagendra']

    if str(params.RFP_Last_Internal_Rating) not in last_internal_rating:
        return {"Error" : "RFP_Last_Internal_Rating should be [Unsatisfactory,Improvement needed,Meets expectations,Exceeds expectations,Exceptional] one of the following"}
    if str(params.Trial_Last_Internal_Rating) not in last_internal_rating:
        return {"Error" : "Trial_Last_Internal_Rating should be [Unsatisfactory,Improvement needed,Meets expectations,Exceeds expectations,Exceptional] one of the following"}
    if str(params.Practice_Head) not in practice_head:
        return {"Error" : "Practice_Head should be ['Ashish','Sunil','Gunjan','Dilip','Nagendra'] one of the following"}


    y_pred = tmr_main(str(params.RFP_Last_Internal_Rating), str(params.Trial_Last_Internal_Rating), float(params.Trial_percent_Present), float(params.RFP_Avg_TRACK_Score),
            float(params.Trial_Avg_TRACK_Score), float(params.RFP_Tech_Ability), float(params.RFP_Last_TRACK_Score), float(params.Trial_Techability_Score), str(params.Practice_Head))

    if y_pred == 3:
        return {"Predicted" : "FullStack_DeepTech"}
    elif y_pred == 2:
        return {"Predicted" : "AdvTech_1_2"}
    elif y_pred == 1:
        return {"Predicted" : "BasicTech"}
    else:
        return {"Predicted" : "StdTech"}