from dataclasses import dataclass
from pydantic import BaseModel, Field
from fastapi import Query
# from dataclasses import dataclass

# @dataclass
class Features(BaseModel):

    # def __init__(
    #     self,
    #     RFP_Last_Internal_Rating: str = Query(description="The RFP_Last_Internal_Rating should be [Unsatisfactory,Improvement needed,Meets expectations,Exceeds expectations,Exceptional]"),
    #     Trial_Last_Internal_Rating: str = Query(description="The Trial_Last_Internal_Rating should be [Unsatisfactory,Improvement needed,Meets expectations,Exceeds expectations,Exceptional]"),
    #     Trial_percent_Present : float = Query(description="The Trial_percent_Present should be within the range 0 to 100", gte=0,lte=100),
    #     RFP_Avg_TRACK_Score : float = Query(description="The RFP_Avg_TRACK_Score should be within the range 0 to 5", gte=0,lte=5),
    #     Trial_Avg_TRACK_Score : float = Query(description="The Trial_Avg_TRACK_Score should be within the range 0 to 5", gte=0,lte=5),
    #     RFP_Tech_Ability : float = Query(description="The RFP_Tech_Ability should be within the range 0 to 2", gte=0,lte=2),
    #     RFP_Last_TRACK_Score : float = Query(description="The RFP_Last_TRACK_Score should be within the range 0 to 5", gte=0,lte=5),
    #     Trial_Techability_Score : float = Query(description="The Trial_Techability_Score should be within the range 0 to 2", gte=0,lte=2),
    #     Practice_Head : str = Query(description="The Practice_Head should be [Ashish,Sunil,Gunjan,Dilip,Nagendra]")
    # ):
    #     self.RFP_Last_Internal_Rating = RFP_Last_Internal_Rating
    #     self.Trial_Last_Internal_Rating = Trial_Last_Internal_Rating
    #     self.Trial_percent_Present = Trial_percent_Present
    #     self.RFP_Avg_TRACK_Score = RFP_Avg_TRACK_Score
    #     self.Trial_Avg_TRACK_Score = Trial_Avg_TRACK_Score
    #     self.RFP_Tech_Ability = RFP_Tech_Ability
    #     self.RFP_Last_TRACK_Score = RFP_Last_TRACK_Score
    #     self.Trial_Techability_Score = Trial_Techability_Score
    #     self.Practice_Head = Practice_Head

    RFP_Last_Internal_Rating: str = Query(description="The RFP_Last_Internal_Rating should be [Unsatisfactory,Improvement needed,Meets expectations,Exceeds expectations,Exceptional]")
    Trial_Last_Internal_Rating: str = Query(description="The Trial_Last_Internal_Rating should be [Unsatisfactory,Improvement needed,Meets expectations,Exceeds expectations,Exceptional]")
    Trial_percent_Present : float = Query(description="The Trial_percent_Present should be within the range 0 to 100", ge=0,le=100)
    RFP_Avg_TRACK_Score : float = Query(description="The RFP_Avg_TRACK_Score should be within the range 0 to 5", ge=0,le=5)
    Trial_Avg_TRACK_Score : float = Query(description="The Trial_Avg_TRACK_Score should be within the range 0 to 5", ge=0,le=5)
    RFP_Tech_Ability : float = Query(description="The RFP_Tech_Ability should be within the range 0 to 2", ge=0,le=2)
    RFP_Last_TRACK_Score : float = Query(description="The RFP_Last_TRACK_Score should be within the range 0 to 5", ge=0,le=5)
    Trial_Techability_Score : float = Query(description="The Trial_Techability_Score should be within the range 0 to 2", ge=0,le=2)
    Practice_Head : str = Query(description="The Practice_Head should be [Ashish,Sunil,Gunjan,Dilip,Nagendra]")    