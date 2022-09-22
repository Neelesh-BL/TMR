import pandas as pd
import sys
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import joblib
from logger import logger


log = logger.logger_init('training_model')


class TrainingModel:

    def __init__(self) -> None:
        try:
            self.training_data_df = pd.read_csv('/home/hp/Desktop/TMR-EDA/Data/part_80.csv')
          
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
            self.training_data_df.drop(self.training_data_df.loc[
                                    (self.training_data_df['RFP Avg TRACK Score']>5.0) | 
                                    (self.training_data_df['Trial Avg TRACK Score']>5.0) | 
                                    (self.training_data_df['RFP Tech Ability']>2.0) |
                                    (self.training_data_df['RFP Last TRACK Score']>5.0) |
                                    (self.training_data_df['Trial Techability Score']>2.0)
                                    ].index, inplace = True)

            # Removing the % symbol from the 'Trial % Present'
            self.training_data_df.replace('\%', '', regex=True, inplace=True)

            # Changing the type of 'Trial % Present' column from object to float
            self.training_data_df['Trial % Present'] = self.training_data_df['Trial % Present'].astype('float64')

            # Dropping the the rows witReturns the 2 dataframes one containg the class label and the other containing rest of the featuresh null values
            self.training_data_df = self.training_data_df.dropna(subset=['RFP Last Internal Rating',
            'Trial Last Internal Rating','Trial % Present','RFP Avg TRACK Score','Trial Avg TRACK Score',
            'RFP Tech Ability','RFP Last TRACK Score','Trial Techability Score','TechCategory'])

            return self.training_data_df

        except:
            exception_type, _, exception_traceback = sys.exc_info()       
            line_number = exception_traceback.tb_lineno
            log.exception(f"Exception type : {exception_type} \nError on line number : {line_number}")


    def encoding(self, df):
        """
            Description:
                This function is used to encode all the categorical features in the data.
            Parameters:
                df : The dataframe which contains the categorical features which needs to be encoded
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

    
    def sampling(self,category_list, category_to_be_sampled_to):
        """
            Description:
                This function is used to perform the sampling of data to the class label which we want.
            Parameters:
                category_list :  List of categories that need to be sampled
                category_to_be_sampled_to : The category to which the category_list is to be sampled upto.
            Return:
                Returns the balanced dataframe with equal rows for each category.
        """
        try:
            # The list to store the sampled data for each category
            resultant_list = []

            # Iterating through each category in category_list
            for category in category_list:

                # Sampling each category in category_list to the size of category_to_be_sampled_to
                category  = resample(category, replace=True, n_samples=len(category_to_be_sampled_to),random_state=42)
                resultant_list.append(category)
            resultant_list.append(category_to_be_sampled_to)

            # Concatinating the all the categories and storing it in a dataframe
            sampled_df = pd.concat(resultant_list)

            return sampled_df

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
                Returns the 2 dataframes one containg the class label and the other containing rest of the features
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


    def model_buidling(self, X, y , classifier, kfold, param_grid):
        """
            Description:
                This function is used to split the data into train and test and apply the hyperparameter-tuned classification model to it.
            Parameters:
                X: Dataframe that contains all the features other than the class label.
                y: Dataframe containing only the class label
                classifier: The object of the model to be applied
                kfold: The number of cross-validation needs to be applied
                param_grid: The dict of all the parameters for the specified classification model
            Return:
                Returns the model, best parameters for the model and the train and test accuracy for the model
        """
        try:
            # Spliting the data into train and test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=100)

            # Hyperparameter tuning using GridSearchCV
            clf_Grid = GridSearchCV(estimator = classifier, param_grid=param_grid, cv=kfold, verbose=2, n_jobs=4)

            # Fitting the model
            clf_Grid.fit(X_train, y_train)

            # Finding the train and test accuracy for the model
            train_accuracy = clf_Grid.score(X_train, y_train)
            test_accuracy = clf_Grid.score(X_test, y_test)
     
            return clf_Grid, clf_Grid.best_params_, train_accuracy, test_accuracy
        
        except:
            exception_type, _, exception_traceback = sys.exc_info()       
            line_number = exception_traceback.tb_lineno
            log.exception(f"Exception type : {exception_type} \nError on line number : {line_number}")

        
    def save_model(self, model):
        """
            Description:
                This function is used to save the model.
            Parameters:
                model: The model that needs to be saved
            Return:
                None
        """
        try:
            # Save the model as a pickle in a file
            joblib.dump(model, '/home/hp/Desktop/TMR-EDA/Data/svc.pkl')
            log.info('The model has been successfully saved')

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
        training_model_obj = TrainingModel()

        # Calling the preprocessing function
        preprocessed_df = training_model_obj.preprocessing()

        # Calling the encoding function
        encoded_df = training_model_obj.encoding(preprocessed_df)

        StdTech = encoded_df[encoded_df["TechCategory"] == 0]
        BasicTech = encoded_df[encoded_df["TechCategory"] == 1]
        AdvTech_1_2 = encoded_df[encoded_df["TechCategory"] == 2]
        FullStack_DeepTech = encoded_df[encoded_df["TechCategory"] == 3]

        # Creating the category list
        category_list = [StdTech,BasicTech,AdvTech_1_2]

        # Calling the sampling function
        sampled_df = training_model_obj.sampling(category_list, FullStack_DeepTech)

        # Calling the segregate function
        X, y = training_model_obj.segregate(sampled_df)

        # Intialing the classifier and parameters for the SVC model
        svc_clf = SVC(gamma='auto')
        param_Grid = {'C':[0.1,1,10,100],
                    'gamma':[1,0.1,0.01,0.001],
                    'kernel':['rbf']}
        kfold=10

        # Calling the model_building function
        model, model_best_params_, train_accuracy, test_accuracy = training_model_obj.model_buidling(X, y, svc_clf, kfold, param_Grid)

        # Printing the train and test accuracy for the model
        log.info(f"Model Best Params : {model_best_params_}")
        log.info(f"Train_Accuracy : {train_accuracy}")
        log.info(f"Test Accuracy : {test_accuracy}")  

        # Calling the save_model function
        training_model_obj.save_model(model)

    except:
        exception_type, _, exception_traceback = sys.exc_info()       
        line_number = exception_traceback.tb_lineno
        log.exception(f"Exception type : {exception_type} \nError on line number : {line_number}")


if __name__ == "__main__":
    main()