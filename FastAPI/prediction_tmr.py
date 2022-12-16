import pandas as pd
import sys, joblib, glob, os, pickle
from pathlib import Path
from logger import logger


log = logger.logger_init('prediction_tmr')


class TestingModel:
    

    def create_df(self, RFP_Avg_TRACK_Score, RFP_Tech_Ability, RFP_Learnability, RFP_Communicability,
                Trial_Last_Internal_Rating,Trial_Communicability_Score, Trial_Learnability_Score, Trial_Techability_Score, Practice_Head):
        """
            Description:
                This function is used create a dataframe for a single datapoint.
            Parameters:
                List of parameters which are feature values
            Return:
                Returns a dataframe for a datapoint
        """
        try:

            df = pd.DataFrame({'RFP Avg TRACK Score': float(RFP_Avg_TRACK_Score), 'RFP Tech Ability': float(RFP_Tech_Ability),
            'RFP Learnability': float(RFP_Learnability), 'RFP Communicability': float(RFP_Communicability),  'Trial Communicability Score': float(Trial_Communicability_Score),
            'Trial Last Internal Rating': str(Trial_Last_Internal_Rating),'Trial Learnability Score': float(Trial_Learnability_Score), 'Trial Techability Score': float(Trial_Techability_Score),
            'Practice Head': str(Practice_Head)}, index=[0])

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
                df: The dataframe on which the preprocessing is to done.
            Return:
                Returns the dataframe with preprocessed data.
        """
        try:
            # Dropping the rows wherein column values does not lies within the defined range
            df.drop(df.loc[
                        (df['RFP Avg TRACK Score']>5.0) | 
                        (df['RFP Tech Ability']>2.0) |
                        (df['RFP Learnability']>2.0) |                   
                        (df['RFP Communicability']>1.0) |
                        (df['Trial Communicability Score']>1.0) |
                        (df['Trial Learnability Score']>2.0) |
                        (df['Trial Techability Score']>2.0)
                        ].index, inplace = True)

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
            path = os.path.join(os.getcwd(), 'encoder') 
            isExist = os.path.exists(path)

            if isExist:
                path = os.getcwd() + '/encoder/' + 'encoder_objects.pickle'      
                with open(Path(path), 'rb') as handle:
                    encoder_dict = pickle.load(handle)

            for col in list(df.columns):
                # checking if the column is categorical or not
                if df[col].dtype == 'object':
                    df[col] = encoder_dict[col].fit_transform(df[col])

            else:
                log.exception(f"No such directory exists")
           
            return df   

        except:
            exception_type, _, exception_traceback = sys.exc_info()       
            line_number = exception_traceback.tb_lineno
            log.exception(f"Exception type : {exception_type} \nError on line number : {line_number}")


    def load_model(self, X):
        """
            Description:
                This function is used to load the saved model and make prediction on the unseen data.
            Parameters:
                X: Dataframe that contains all the features excluding the class label.
            Return:
                Returns the predicted tech category.
        """
        try:
            path = os.getcwd() + '/ML_models/'
            isExist = os.path.exists(path)

            if isExist:
                list_of_files = glob.glob(path + '*.pkl')
                latest_file = max(list_of_files, key=os.path.getctime)
                
                # Load the model from the file
                rf_clf = joblib.load(latest_file)

                # Use the loaded model to make predictions
                y_pred = rf_clf.predict(X)

                return y_pred

        except:
            exception_type, _, exception_traceback = sys.exc_info()       
            line_number = exception_traceback.tb_lineno
            log.exception(f"Exception type : {exception_type} \nError on line number : {line_number}")


def tmr_main(RFP_Avg_TRACK_Score, RFP_Tech_Ability, RFP_Learnability, RFP_Communicability,
            Trial_Last_Internal_Rating,Trial_Communicability_Score, Trial_Learnability_Score, Trial_Techability_Score, Practice_Head):
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
    
        df = testing_model_obj.create_df(RFP_Avg_TRACK_Score, RFP_Tech_Ability, RFP_Learnability, RFP_Communicability,
            Trial_Last_Internal_Rating,Trial_Communicability_Score, Trial_Learnability_Score, Trial_Techability_Score, Practice_Head)

        # Calling the preprocessing function
        preprocessed_df = testing_model_obj.preprocessing(df)

        # Calling the encoding function
        encoded_df = testing_model_obj.encoding(preprocessed_df)

        # Calling the load_model function
        y_pred = testing_model_obj.load_model(encoded_df)

        return y_pred

    except:
        exception_type, _, exception_traceback = sys.exc_info()       
        line_number = exception_traceback.tb_lineno
        log.exception(f"Exception type : {exception_type} \nError on line number : {line_number}")


if __name__ == "__main__":
    tmr_main()