from fastapi import APIRouter, File, UploadFile, BackgroundTasks, Depends, status
from prediction_tmr import tmr_main
from models.features import Features
import pandas as pd
import numpy as np
from config.database import db
from bson.objectid import ObjectId
import sys, csv, json, os, codecs, pydantic
pydantic.json.ENCODERS_BY_TYPE[ObjectId]=str
from pathlib import Path
from logger import logger
from routes.auth import get_api_key
from fastapi.security.api_key import APIKey
from datetime import datetime
 


log = logger.logger_init('tmr_api')


prediction = APIRouter()

@prediction.get('/')
async def welcome():
    return {"detail":"Welcome to Bridgelabz TMR"}


@prediction.post("/predict_individual_data")
async def test_model(params: Features = Depends(), api_key: APIKey = Depends(get_api_key)):
    try:

        last_internal_rating = ['UNSATISFACTORY','IMPROVEMENT NEEDED','MEETS EXPECTATIONS','EXCEEDS EXPECTATIONS','EXCEPTIONAL']
        practice_head = ['ASHISH','SUNIL','GUNJAN','DILIP','NAGENDRA']

        if str(params.Trial_Last_Internal_Rating).upper() not in last_internal_rating:
            return {"status_code":status.HTTP_400_BAD_REQUEST, "message": "Trial_Last_Internal_Rating should be [Unsatisfactory,Improvement needed,Meets expectations,Exceeds expectations,Exceptional] one of the following"}
        if str(params.Practice_Head).upper() not in practice_head:
            return {"status_code":status.HTTP_400_BAD_REQUEST, "message": "Practice_Head should be ['Ashish','Sunil','Gunjan','Dilip','Nagendra'] one of the following"}

        y_pred = tmr_main(float(params.RFP_Avg_TRACK_Score), float(params.RFP_Tech_Ability), float(params.RFP_Learnability), float(params.RFP_Communicability),
                        str(params.Trial_Last_Internal_Rating), float(params.Trial_Communicability_Score), float(params.Trial_Learnability_Score), float(params.Trial_Techability_Score), str(params.Practice_Head))

        if y_pred == 2:
            return {"status_code":status.HTTP_200_OK,"Predicted" : "FullStack_DeepTech"}
        elif y_pred == 0:
            return {"status_code":status.HTTP_200_OK,"Predicted" : "AdvTech_1_2"}
        elif y_pred == 1:
            return {"status_code":status.HTTP_200_OK,"Predicted" : "BasicTech"}
        else:
            return {"status_code":status.HTTP_200_OK,"Predicted" : "StdTech"}
    
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


def techcategory_predition(df):
    """
        Description:
            This function is used to predict the TechCategory for each datapoint in the dataframe
        Parameters:
            df: The dataframe containing the datapoints
        Return:
            It returns a dataframe containing an additional column with the predicted results
    """
    try :

        # List to store the prediction result for each datapoint
        prediction_list = []

        # List to store the admit_id for each datapoint
        admit_id_list = []

        # Dictionary to store the admit_id and predictions
        final_dict = {}
                    
        # Renaming few column names
        df = df.rename(columns={ 
                    'ADMIT ID':'Admit ID',
                    'RFP AVG TRACK SCORE':'RFP Avg TRACK Score',
                    'RFP TECH ABILITY':'RFP Tech Ability',
                    'RFP LEARNABILITY':'RFP Learnability',
                    'RFP COMMUNICABILITY':'RFP Communicability',
                    'TRIAL COMMUNICABILITY SCORE':'Trial Communicability Score',
                    'TRIAL LAST INTERNAL RATING':'Trial Last Internal Rating',
                    'TRIAL LEARNABILITY SCORE':'Trial Learnability Score',
                    'TRIAL TECHABILITY SCORE':'Trial Techability Score',
                    'PRACTICE HEAD':'Practice Head',
                    'TECHCATEGORY':'TechCategory',
                    'CONSIDER':'Consider'
                    })

        print(df.columns)
        
        # Dropping the rows wherein the column(Consider) is 'Ignore'
        df.drop(df[df['Consider']=='Ignore'].index , inplace=True)

        # Replacing the empty string with nan in the dataframe
        df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

        # Dropping the rows wherein the mandatory columns are null/empty
        df.dropna(subset=['RFP Avg TRACK Score','RFP Tech Ability','RFP Learnability','RFP Communicability','Trial Communicability Score',
                'Trial Last Internal Rating','Trial Learnability Score','Trial Techability Score','Practice Head','TechCategory'],inplace=True)
        df.reset_index(inplace = True, drop=True)

        # Changing the type of certain columns from object to float
        df['RFP Avg TRACK Score'] = df['RFP Avg TRACK Score'].astype('float64')
        df['RFP Tech Ability'] = df['RFP Tech Ability'].astype('float64')
        df['RFP Learnability'] = df['RFP Learnability'].astype('float64')
        df['RFP Communicability'] = df['RFP Communicability'].astype('float64')
        df['Trial Communicability Score'] = df['Trial Communicability Score'].astype('float64')
        df['Trial Learnability Score'] = df['Trial Learnability Score'].astype('float64')
        df['Trial Techability Score'] = df['Trial Techability Score'].astype('float64')

        # Iterating through the dataframe
        for index in df.index:
            RFP_Avg_TRACK_Score = df['RFP Avg TRACK Score'][index]
            RFP_Tech_Ability = df['RFP Tech Ability'][index]
            RFP_Learnability = df['RFP Learnability'][index]
            RFP_Communicability = df['RFP Communicability'][index]
            Trial_Communicability_Score = df['Trial Communicability Score'][index]
            Trial_Last_Internal_Rating = df['Trial Last Internal Rating'][index]              
            Trial_Learnability_Score = df['Trial Learnability Score'][index]
            Trial_Techability_Score = df['Trial Techability Score'][index]     
            Practice_Head = df['Practice Head'][index]
                
            # Calling the tmr_main()
            y_pred = tmr_main(RFP_Avg_TRACK_Score, RFP_Tech_Ability, RFP_Learnability, RFP_Communicability,
                            Trial_Last_Internal_Rating,Trial_Communicability_Score, Trial_Learnability_Score, Trial_Techability_Score, Practice_Head)

            if y_pred == 2:
                admit_id_list.append(df['Admit ID'][index])
                prediction_list.append("FullStack_DeepTech")
            elif y_pred == 0:
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

    except: 
        exception_type, _, exception_traceback = sys.exc_info()       
        line_number = exception_traceback.tb_lineno
        log.exception(f"Exception type : {exception_type} \nError on line number : {line_number}")


@prediction.post("/predict_bulk_data")
async def upload(background_tasks: BackgroundTasks,file: UploadFile = File(...), api_key: APIKey = Depends(get_api_key)):
    try:

        # Checking if the file extension is .csv or not
        if file.filename.endswith('.csv'):
            csvReader = csv.DictReader(codecs.iterdecode(file.file, 'utf-8'))
            background_tasks.add_task(file.file.close)
            json_data = list(csvReader)

            df = pd.json_normalize(json_data)
            df.columns = [col.upper() for col in df.columns]
            input_col_list = list(df.columns)
            expected_col_list = ['RFP AVG TRACK SCORE','RFP TECH ABILITY','RFP LEARNABILITY','RFP COMMUNICABILITY','TRIAL COMMUNICABILITY SCORE',
                        'TRIAL LAST INTERNAL RATING','TRIAL LEARNABILITY SCORE','TRIAL TECHABILITY SCORE','PRACTICE HEAD','TECHCATEGORY']
            check =  all(item in input_col_list for item in expected_col_list)

            if check is True:
                json_string, df2 = techcategory_predition(df)
                date_string = datetime.now().strftime("%d_%m_%Y_%H_%M")
                path = os.path.join(os.getcwd(), 'results')
                isExist = os.path.exists(path)

                if not isExist:          
                    os.mkdir(path)            
                path = os.getcwd() + '/results/' + 'result_' + date_string + '.csv'
                
                # Storing the dataframe into a csv file
                df2.to_csv(Path(path), index=False)

                # Converting the dataframe to dictionary
                data_dict = df2.to_dict("records")

                # Storing the data_dict into mongodb collection
                db.tmr_prediction_results.insert_many(data_dict)

                return {"status_code":status.HTTP_200_OK, "file_contents": json_string}

            else:
                return {"status_code":status.HTTP_400_BAD_REQUEST, "message": "The file doesn't contains the all the columns required to make predictions"}
        else:
            return {"status_code":status.HTTP_400_BAD_REQUEST, "message": "The file extension should be .csv"}
    except:
        exception_type, _, exception_traceback = sys.exc_info()       
        line_number = exception_traceback.tb_lineno
        log.exception(f"Exception type : {exception_type} \nError on line number : {line_number}")