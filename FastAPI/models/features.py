from pydantic import BaseModel
from fastapi import Query

class Features(BaseModel):

    RFP_Avg_TRACK_Score : float = Query(description="The RFP Avg TRACK Score should be within the range 0 to 5", ge=0,le=5)
    RFP_Tech_Ability : float = Query(description="The RFP Tech Ability should be within the range 0 to 2", ge=0,le=2)
    RFP_Learnability : float = Query(description="The RFP Learnability should be within the range 0 to 2", ge=0,le=2)
    RFP_Communicability : float = Query(description="The RFP Communicability should be within the range 0 to 1", ge=0,le=1)
    Trial_Communicability_Score : float = Query(description="The Trial Communicability Score should be within the range 0 to 1", ge=0,le=1)
    Trial_Learnability_Score : float = Query(description="The Trial Learnability Score should be within the range 0 to 2", ge=0,le=2)
    Trial_Techability_Score : float = Query(description="The Trial Techability Score should be within the range 0 to 2", ge=0,le=2)
    Trial_Last_Internal_Rating: str = Query(description="The Trial Last_Internal Rating should be [Unsatisfactory,Improvement needed,Meets expectations,Exceeds expectations,Exceptional]")
    Practice_Head : str = Query(description="The Practice_Head should be [Ashish,Sunil,Gunjan,Dilip,Nagendra]")    