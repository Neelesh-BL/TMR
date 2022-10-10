from pydantic import BaseModel
from fastapi import Query

class Features(BaseModel):

    RFP_Last_Internal_Rating: str = Query(description="The RFP_Last_Internal_Rating should be [Unsatisfactory,Improvement needed,Meets expectations,Exceeds expectations,Exceptional]")
    Trial_Last_Internal_Rating: str = Query(description="The Trial_Last_Internal_Rating should be [Unsatisfactory,Improvement needed,Meets expectations,Exceeds expectations,Exceptional]")
    Trial_percent_Present : float = Query(description="The Trial_percent_Present should be within the range 0 to 100", ge=0,le=100)
    RFP_Avg_TRACK_Score : float = Query(description="The RFP_Avg_TRACK_Score should be within the range 0 to 5", ge=0,le=5)
    Trial_Avg_TRACK_Score : float = Query(description="The Trial_Avg_TRACK_Score should be within the range 0 to 5", ge=0,le=5)
    RFP_Tech_Ability : float = Query(description="The RFP_Tech_Ability should be within the range 0 to 2", ge=0,le=2)
    RFP_Last_TRACK_Score : float = Query(description="The RFP_Last_TRACK_Score should be within the range 0 to 5", ge=0,le=5)
    Trial_Techability_Score : float = Query(description="The Trial_Techability_Score should be within the range 0 to 2", ge=0,le=2)
    Practice_Head : str = Query(description="The Practice_Head should be [Ashish,Sunil,Gunjan,Dilip,Nagendra]")    