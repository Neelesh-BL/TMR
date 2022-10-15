from fastapi import Depends
from prediction_tmr import tmr_main
from fastapi import APIRouter, File, UploadFile
from models.features import Features
import pandas as pd
import json
from config.database import db
import numpy as np
import pydantic
from bson.objectid import ObjectId
pydantic.json.ENCODERS_BY_TYPE[ObjectId]=str
import sys
from logger import logger


log = logger.logger_init('tmr_api')


prediction = APIRouter()

@prediction.get('/')
async def welcome():
    return {"detail":"Welcome to Bridgelabz TMR"}


@prediction.post("/test_model")
async def test_model(params: Features = Depends()):
    try:
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
    
    except:
        exception_type, _, exception_traceback = sys.exc_info()       
        line_number = exception_traceback.tb_lineno
        log.exception(f"Exception type : {exception_type} \nError on line number : {line_number}")


def convertStringToDataframe(content):
    """
        Description:
            This function is used to convert the string data into a dataframe
        Parameters:
            content: The string data that needs to be converted into a dataframe
        Return:
            Returns a dataframe
    """
    try :

        # List to store the data rows for the dataframe
        data_list = []
        data = content.decode('utf-8').splitlines()

        # iterating the data
        for index,row in enumerate(data):

            # The first line contains the column names therefore checking for index=0
            if index ==0:

                # storing the column names
                columns = row.split(',')

            else:

                # List to store the data for each row
                row_list = []

                # storing the row data
                row_data = row.split(',')

                for value in row_data:

                    # Checking if any entry is empty as replacing it with nan
                    if value == '':
                        row_list.append(np.nan)
                    else:
                        row_list.append(value)

                # Appending the all the row lists
                data_list.append(row_list)
            
        df = pd.DataFrame(data_list)
        df.columns = columns
        return df
    
    except:
        
        exception_type, _, exception_traceback = sys.exc_info()       
        line_number = exception_traceback.tb_lineno
        log.exception(f"Exception type : {exception_type} \nError on line number : {line_number}")
    

def techcategory_predition(df):
    """
        Description:
            This function is used to predict the TechCategory for each datapoint in the dataframe
        Parameters:
            df: The dataframe containing the datapoints
        Return:
            It returns a dataframe containing an additional column with the predicted result
    """
    try :

        # List to store the prediction result for each datapoint
        prediction_list = []

        # List to store the admit_id for each datapoint
        admit_id_list = []

        # Dictionary to store the admit_id and predictions
        final_dict = {}

        # Dropping the rows wherein the mandatory columns are null/empty
        df.dropna(subset=['RFP Last Internal Rating','Trial Last Internal Rating','Trial % Present','RFP Avg TRACK Score','Trial Avg TRACK Score','RFP Tech Ability','RFP Last TRACK Score','Trial Techability Score','Practice Head'],inplace=True)
        df.reset_index(inplace = True, drop=True)

        # Checking if the column names are present in the dataframe
        if (('RFP Last Internal Rating' in df.columns)  and ('Trial Last Internal Rating' in df.columns) and 
            ('Trial % Present' in df.columns) and ('RFP Avg TRACK Score' in df.columns) and ('Trial Avg TRACK Score' in df.columns)
            and ('RFP Tech Ability'  in df.columns) and ('RFP Last TRACK Score' in df.columns) and ('Trial Techability Score' in df.columns)
            and ('Practice Head' in df.columns)) :

            # Iterating through the dataframe
            for index in df.index:
                RFP_Last_Internal_Rating = df['RFP Last Internal Rating'][index]
                Trial_Last_Internal_Rating = df['Trial Last Internal Rating'][index]
                Trial_percent_Present = df['Trial % Present'][index]
                RFP_Avg_TRACK_Score = df['RFP Avg TRACK Score'][index]
                Trial_Avg_TRACK_Score = df['Trial Avg TRACK Score'][index]
                RFP_Tech_Ability = df['RFP Tech Ability'][index]
                RFP_Last_TRACK_Score = df['RFP Last TRACK Score'][index]
                Trial_Techability_Score = df['Trial Techability Score'][index]
                Practice_Head = df['Practice Head'][index]
        
                # Calling the tmr_main()
                y_pred = tmr_main(RFP_Last_Internal_Rating, Trial_Last_Internal_Rating, Trial_percent_Present, RFP_Avg_TRACK_Score,
                                Trial_Avg_TRACK_Score, RFP_Tech_Ability, RFP_Last_TRACK_Score, Trial_Techability_Score, Practice_Head)

                if y_pred == 3:
                    admit_id_list.append(df['Admit ID'][index])
                    prediction_list.append("FullStack_DeepTech")
                elif y_pred == 2:
                    admit_id_list.append(df['Admit ID'][index])
                    prediction_list.append("AdvTech_1_2")
                elif y_pred == 1:
                    admit_id_list.append(df['Admit ID'][index])
                    prediction_list.append("BasicTech")
                else:
                    admit_id_list.append(df['Admit ID'][index])
                    prediction_list.append("StdTech")

            final_dict = {
                'Admit ID' : admit_id_list,
                'Predicted' : prediction_list
            }
            
            # Adding the Prediction column to the existing dataframe
            df['Prediction'] = prediction_list

            new_df = pd.DataFrame(final_dict)

            return parse_csv(new_df), df

        else:
            return {'Error': 'Please check if all the mandatory columns are present in the dataset or not'}

    except:
        exception_type, _, exception_traceback = sys.exc_info()       
        line_number = exception_traceback.tb_lineno
        log.exception(f"Exception type : {exception_type} \nError on line number : {line_number}")


def parse_csv(df):
    """ 
        Description:
            This function is used to convert the dataframe into json format
        Parameters:
            df: The dataframe that is to be converted
        Return:
            It returns the json formatted output 
    """
    try:
        result = df.to_json(orient="records")
        parsed = json.loads(result)
        return parsed

    except:
        exception_type, _, exception_traceback = sys.exc_info()       
        line_number = exception_traceback.tb_lineno
        log.exception(f"Exception type : {exception_type} \nError on line number : {line_number}")


@prediction.post("/csv")
async def parsecsv(file: UploadFile = File(...)):
    try:
        # Reading the content of the file
        contents = await file.read()

        # Converting the csv data into a dataframe
        df = convertStringToDataframe(contents)
        
        # Calling the techcategory_prediction function
        json_string, df2 = techcategory_predition(df)

        # Converting the dataframe to dictionary
        data_dict = df2.to_dict("records")

        # Storing the data_dict into mongodb collection
        db.tmr_prediction_results.insert_many(data_dict)

        return {"file_contents": json_string}

    except:
        exception_type, _, exception_traceback = sys.exc_info()       
        line_number = exception_traceback.tb_lineno
        log.exception(f"Exception type : {exception_type} \nError on line number : {line_number}")