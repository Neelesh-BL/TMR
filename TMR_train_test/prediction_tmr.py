import pandas as pd
import sys
import joblib
# from sklearn.metrics import accuracy_score
from logger import logger


log = logger.logger_init('prediction_tmr')


class TestingModel:
    def create_df(self, RFP_Last_Internal_Rating, Trial_Last_Internal_Rating, Trial_percent_Present, RFP_Avg_TRACK_Score,
                 Trial_Avg_TRACK_Score, RFP_Tech_Ability, RFP_Last_TRACK_Score, Trial_Techability_Score, Practice_Head):
        try:     
            df = pd.DataFrame({'RFP Last Internal Rating': RFP_Last_Internal_Rating, 'Trial Last Internal Rating': Trial_Last_Internal_Rating,
            'Trial % Present': Trial_percent_Present, 'RFP Avg TRACK Score': RFP_Avg_TRACK_Score, 'Trial Avg TRACK Score': Trial_Avg_TRACK_Score,
            'RFP Tech Ability': RFP_Tech_Ability,'RFP Last TRACK Score': RFP_Last_TRACK_Score,'Trial Techability Score': Trial_Techability_Score,
            'Practice Head': Practice_Head}, index=[0])

            return df

        except:
            exception_type, _, exception_traceback = sys.exc_info()       
            line_number = exception_traceback.tb_lineno
            log.exception(f"Exception type : {exception_type} \nError on line number : {line_number}")


    def preprocessing(self, df):
        """
            Description:
                This function is used perform all the preprocessing steps in the data.
            Parameters:
                None.
            Return:
                Returns the dataframe with preprocessed data.
        """
        try:
            # Dropping the rows wherein column values does not lies within the defined range
            df.drop(self.testing_data_df.loc[
                                    (df['RFP Avg TRACK Score']>5.0) | 
                                    (df['Trial Avg TRACK Score']>5.0) | 
                                    (df['RFP Tech Ability']>2.0) |
                                    (df['RFP Last TRACK Score']>5.0) |
                                    (df['Trial Techability Score']>2.0)
                                    ].index, inplace = True)

            # Removing the % symbol from the 'Trial % Present'
            df.replace('\%', '', regex=True, inplace=True)

            # Changing the type of 'Trial % Present' column from object to float
            df['Trial % Present'] = df['Trial % Present'].astype('float64')

           # Dropping all those rows which have any value as null
            df = df.dropna(subset=['RFP Last Internal Rating',
            'Trial Last Internal Rating','Trial % Present','RFP Avg TRACK Score','Trial Avg TRACK Score',
            'RFP Tech Ability','RFP Last TRACK Score','Trial Techability Score','Practice Head'])

            # Replacing the practice head names to Sunil and Ashish if Sunil Patil, Ashish Vishwakarma are present
            df.loc[df["Practice Head"] == "Sunil Patil", "Practice Head"] = "Sunil"
            df.loc[df["Practice Head"] == "Ashish Vishwakarma", "Practice Head"] = "Ashish"

            return df

        except:
            exception_type, _, exception_traceback = sys.exc_info()       
            line_number = exception_traceback.tb_lineno
            log.exception(f"Exception type : {exception_type} \nError on line number : {line_number}")


    def encoding(self, df):
        """
            Description:
                This function is used to encode all the categorical features in the data.
            Parameters:
                df : The dataframe which contains the categorical features which needs to be encoded.
            Return:
                Returns the dataframe with encoded categorical features.
        """
        try:
            # Encoding the feature 'Trial Last Internal Rating'
            df['Trial Last Internal Rating'] = df['Trial Last Internal Rating'].replace(
                                                {'Unsatisfactory': 0, 'Improvement needed': 1, 'Meets expectations':2,
                                                 'Exceeds expectations':3,'Exceptional':4})
            
            # Encoding the feature 'RFP Last Internal Rating'
            df['RFP Last Internal Rating'] = df['RFP Last Internal Rating'].replace(
                                                {'Unsatisfactory': 0, 'Improvement needed': 1, 'Meets expectations':2,
                                                 'Exceeds expectations':3,'Exceptional':4})

             # Encoding the feature 'Practice Head'
            df['Practice Head'] = df['Practice Head'].replace({'Nagendra':0, 'Dilip':1,'Gunjan':3, 'Sunil':4,
                                                                'Ashish':5})

            return df   

        except:
            exception_type, _, exception_traceback = sys.exc_info()       
            line_number = exception_traceback.tb_lineno
            log.exception(f"Exception type : {exception_type} \nError on line number : {line_number}")


    def segregate(self, df):
        """
            Description:
                This function is used to seperate out the class label from rest of the features in the data.
            Parameters:
                df: Dataframe that contains all the features including the class label.
            Return:
                Returns the 2 dataframes one containg the class label and the other containing rest of the features.
        """
        try:
            # Dataframe containing only the class label
            y = df['TechCategory']

            # Dataframe containing all the features other than the class label
            X = df.drop(['TechCategory'], axis=1)

            return X, y

        except:
            exception_type, _, exception_traceback = sys.exc_info()       
            line_number = exception_traceback.tb_lineno
            log.exception(f"Exception type : {exception_type} \nError on line number : {line_number}")


    def load_model(self, X, y):
        """
            Description:
                This function is used to load the saved model and make prediction on the unseen data.
            Parameters:
                X: Dataframe that contains all the features excluding the class label.
                y: Dataframe containing only the class label.
            Return:
                Returns the accuracy of the model on unseen data.
        """
        try:
            # Load the model from the file
            svc_clf = joblib.load('/home/hp/Desktop/TMR-EDA/Data/svc.pkl')

            # Use the loaded model to make predictions
            y_pred = svc_clf.predict(X)

            return y_pred

        except:
            exception_type, _, exception_traceback = sys.exc_info()       
            line_number = exception_traceback.tb_lineno
            log.exception(f"Exception type : {exception_type} \nError on line number : {line_number}")


def tmr_main(RFP_Last_Internal_Rating, Trial_Last_Internal_Rating, Trial_percent_Present, RFP_Avg_TRACK_Score,
            Trial_Avg_TRACK_Score, RFP_Tech_Ability, RFP_Last_TRACK_Score, Trial_Techability_Score, Practice_Head):
    """
        Description:
            This function is used to call the other functions.
        Parameters:
            None.
        Return:
            None.
    """
    try:
        # Instantiating the class
        testing_model_obj = TestingModel()

        df = testing_model_obj.create_df(RFP_Last_Internal_Rating, Trial_Last_Internal_Rating, Trial_percent_Present, RFP_Avg_TRACK_Score,
                                        Trial_Avg_TRACK_Score, RFP_Tech_Ability, RFP_Last_TRACK_Score, Trial_Techability_Score, Practice_Head)

        # Calling the preprocessing function
        preprocessed_df = testing_model_obj.preprocessing(df)

        # Calling the encoding function
        encoded_df = testing_model_obj.encoding(preprocessed_df)
        
        # Calling the segregate function
        X, y = testing_model_obj.segregate(encoded_df)

        # Calling the load_model function
        y_pred = testing_model_obj.load_model(X, y)

        return y_pred

    except:
        exception_type, _, exception_traceback = sys.exc_info()       
        line_number = exception_traceback.tb_lineno
        log.exception(f"Exception type : {exception_type} \nError on line number : {line_number}")


if __name__ == "__main__":
    tmr_main()