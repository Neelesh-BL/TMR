import pandas as pd
import sys, joblib, os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from datetime import datetime
from pathlib import Path
from logger import logger


log = logger.logger_init('training_model')


class TrainingModel:

    def load_file(self):
        try:
            path = os.getcwd() + '/data/' + 'processed_data.csv' 
            isExist = os.path.exists(path)

            if isExist:                      
                df = pd.read_csv(Path(path))
                return df
            else:
                log.exception(f"No such directory/file exists")         
            
        except:
            exception_type, _, exception_traceback = sys.exc_info()       
            line_number = exception_traceback.tb_lineno
            log.exception(f"Exception type : {exception_type} \nError on line number : {line_number}")


    def feature_selection(self, df):
        """
            Description:
                This function is used to select the features based on the results of feature importance.
            Parameters:
                df : The original encoded df.
            Return:
                Returns the dataframe with selected features data.
        """
        try:

            columns = ['RFP Avg TRACK Score','RFP Tech Ability','RFP Learnability','RFP Communicability','Trial Communicability Score',
                        'Trial Last Internal Rating','Trial Learnability Score','Trial Techability Score','Practice Head','TechCategory']

            df = df[columns]

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
            date_string = datetime.now().strftime("%d_%m_%Y_%H_%M")
            path = os.path.join(os.getcwd(), 'ML_models')
            isExist = os.path.exists(path)

            if not isExist:          
                os.mkdir(path)            
            path = os.getcwd() + '/ML_models/' + 'model' + date_string + '.pkl'
            
            # Save the model as a pickle in a file
            joblib.dump(model, Path(path))
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

        # Calling the file load function
        df = training_model_obj.load_file()
        if df is not None:

            # Calling the feature_selection function
            feature_selected_df = training_model_obj.feature_selection(df)
            
            # Calling the segregate function
            X, y = training_model_obj.segregate(feature_selected_df)

            # Intialing the classifier and parameters for the Random Forest Classifier model
            rf_clf = RandomForestClassifier(random_state=42)

            # The number of trees in the forest.
            n_estimators = [ int(x) for x in np.linspace(start=10, stop=80, num=10) ]
            # Number of features to consider at every split
            max_features = ['auto', 'sqrt']
            # Maximum number of levels in tree
            max_depth = [2,3,4,5,10]
            # Minimum number of samples required to split a node
            min_samples_split = [2,3,5]
            # Minimum number of samples required at each leaf node
            min_samples_leaf = [1,2]
            # Method of selecting samples for training each tree
            bootstrap = [True, False]

            # Creating the param grid
            param_grid = {'n_estimators': n_estimators,
                        'max_features': max_features,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf,
                        'bootstrap': bootstrap}
            kfold=10

            # Calling the model_building function
            model, model_best_params_, train_accuracy, test_accuracy = training_model_obj.model_buidling(X, y, rf_clf, kfold, param_grid)

            # Printing the train and test accuracy for the model
            log.info(f"Model Best Params : {model_best_params_}")
            log.info(f"Train Accuracy : {train_accuracy}")
            log.info(f"Validation Accuracy : {test_accuracy}")  

            # Calling the save_model function
            training_model_obj.save_model(model)

    except:
        exception_type, _, exception_traceback = sys.exc_info()       
        line_number = exception_traceback.tb_lineno
        log.exception(f"Exception type : {exception_type} \nError on line number : {line_number}")


if __name__ == "__main__":
    main()