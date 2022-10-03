import pandas as pd
import sys
import joblib
from sklearn.metrics import accuracy_score
from logger import logger


log = logger.logger_init('testing_model')


class TestingModel:
    def __init__(self) -> None:
        try:
            self.testing_data_df = pd.read_csv('/home/hp/Desktop/TMR-EDA/Data/part_20.csv')
          
        except:
            exception_type, _, exception_traceback = sys.exc_info()       
            line_number = exception_traceback.tb_lineno
            log.exception(f"Exception type : {exception_type} \nError on line number : {line_number}")


    def preprocessing(self):
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
            self.testing_data_df.drop(self.testing_data_df.loc[
                                    (self.testing_data_df['RFP Avg TRACK Score']>5.0) | 
                                    (self.testing_data_df['Trial Avg TRACK Score']>5.0) | 
                                    (self.testing_data_df['RFP Tech Ability']>2.0) |
                                    (self.testing_data_df['RFP Last TRACK Score']>5.0) |
                                    (self.testing_data_df['Trial Techability Score']>2.0)
                                    ].index, inplace = True)

            # Removing the % symbol from the 'Trial % Present'
            self.testing_data_df.replace('\%', '', regex=True, inplace=True)

            # Changing the type of 'Trial % Present' column from object to float
            self.testing_data_df['Trial % Present'] = self.testing_data_df['Trial % Present'].astype('float64')

            # Dropping all those rows which have any value as null
            self.testing_data_df = self.testing_data_df.dropna(subset=['RFP Last Internal Rating',
            'Trial Last Internal Rating','Trial % Present','RFP Avg TRACK Score','Trial Avg TRACK Score',
            'RFP Tech Ability','RFP Last TRACK Score','Trial Techability Score','TechCategory'])

            return self.testing_data_df

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

            # Encoding the feature 'TechCategory'
            df['TechCategory'] = df['TechCategory'].replace({'FullStack': 3, 'AdvTech-1': 2,'AdvTech-2':2,
                                                             'BasicTech':1, 'StdTech':0, 'DeepTech':3,})

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

            # Finding the Accuracy on unseen data
            accuracy = accuracy_score(y, y_pred)

            return accuracy

        except:
            exception_type, _, exception_traceback = sys.exc_info()       
            line_number = exception_traceback.tb_lineno
            log.exception(f"Exception type : {exception_type} \nError on line number : {line_number}")


def main():
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

        # Calling the preprocessing function
        preprocessed_df = testing_model_obj.preprocessing()

        # Calling the encoding function
        encoded_df = testing_model_obj.encoding(preprocessed_df)
        
        # Calling the segregate function
        X, y = testing_model_obj.segregate(encoded_df)

        # Calling the load_model function
        accuracy = testing_model_obj.load_model(X, y)

        # Printing the accuracy for the model
        log.info(f"Model Accuracy : {accuracy}")

    except:
        exception_type, _, exception_traceback = sys.exc_info()       
        line_number = exception_traceback.tb_lineno
        log.exception(f"Exception type : {exception_type} \nError on line number : {line_number}")


if __name__ == "__main__":
    main()