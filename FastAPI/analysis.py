import pandas as pd
import sys, os, pickle
from logger import logger
from pathlib import Path
from sklearn.preprocessing import LabelEncoder 


log = logger.logger_init('analysis')


class Analysis:

    def load_file(self):
        try:
            path = os.getcwd() + '/data/' + 'original_data1_data2_data3.csv' 
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


    def preprocessing(self,df):
        """
            Description:
                This function is used to perform all the preprocessing steps in the data.
            Parameters:
                None.
            Return:
                Returns the dataframe with preprocessed data.
        """
        try:

            # Dropping all those rows which have any value as null
            df.dropna(inplace=True)
            df.reset_index(inplace = True, drop=True)

            # Dropping duplicate rows
            df.drop_duplicates(keep='last', inplace=True)
            df.reset_index(inplace = True, drop=True)
            
            # Dropping the rows wherein column values does not lies within the defined range
            df.drop(df.loc[
                                    (df['RFP Avg TRACK Score']>5.0) | 
                                    (df['Trial Avg TRACK Score']>5.0) | 
                                    (df['RFP Tech Ability']>2.0) |
                                    (df['RFP- Learnability']>2.0) |
                                    (df['RFP-Communicability']>1.0) |
                                    (df['RFP Last TRACK Score']>5.0) |
                                    (df['Trial Techability Score']>2.0)|
                                    (df['Trial Communicability Score']>1.0)|
                                    (df['Trial Learnability Score']>2.0)|
                                    (df['Year']>2022)
                                    ].index, inplace = True)

            # Removing the % symbol from the 'Trial % Present' and 'RFP % Present columns
            df.replace('\%', '', regex=True, inplace=True)

            # Changing the type of 'Trial % Present' and 'RFP % Present' columns from object to float
            df['Trial % Present'] = df['Trial % Present'].astype('float64')
            df['RFP % Present'] = df['RFP % Present'].astype('float64')

            # Replacing the 'Practice Head' values for certain rows
            df.loc[df["Practice Head"] == "Sunil Patil", "Practice Head"] = "Sunil"
            df.loc[df["Practice Head"] == "Sunil patil", "Practice Head"] = "Sunil"
            df.loc[df["Practice Head"] == "Ashish Vishwakarma", "Practice Head"] = "Ashish"

            # Replacing the row value of 'RFP Last TRACK Score' with its mean value wherein it is zero
            df.loc[df["RFP Last TRACK Score"] == 0, "RFP Last TRACK Score"] = round(df['RFP Last TRACK Score'].mean(),2)
            
            # Replacing the 'Tech Category' values for certain rows
            df.loc[df["TechCategory"] == "FullStack", "TechCategory"] = "FullStack_DeepTech"
            df.loc[df["TechCategory"] == "DeepTech", "TechCategory"] = "FullStack_DeepTech"
            df.loc[df["TechCategory"] == "AdvTech-1", "TechCategory"] = "AdvTech_1_2"
            df.loc[df["TechCategory"] == "AdvTech-2", "TechCategory"] = "AdvTech_1_2"

            df = df.rename(columns={ 'RFP- Learnability':'RFP Learnability',
            'RFP-Communicability':'RFP Communicability'})

            return df

        except:
            exception_type, _, exception_traceback = sys.exc_info()       
            line_number = exception_traceback.tb_lineno
            log.exception(f"Exception type : {exception_type} \nError on line number : {line_number}")


    def bucketing_graduation(self, row):
        """
            Description:
                This function is used to perform the bucketing of 'Last Graduation' into smaller buckets.
            Parameters:
                None.
            Return:
                Returns the dataframe with preprocessed data.
        """
        if row['Last Graduation'] in ['B.E. / B.TECH CS', 'B.E. / B.TECH IT', 'B.Tech Information Technology', 'Computer Science', 'B.E. Computer Engineering', 'B.Tech Computer Science','B.E. Information Technology']:
            return 'BE_BTECH_CS_IT'
        if row['Last Graduation'] in ['B.E. / B.TECH Non IT','B.E. Mechanical Engineering','Mechatronics Engineering','B.Tech Chemical engineering','B.Tech Electrical Engineering','B.Tech Electronics Engineering','B.E. Electrical Engineering','B.E. Electronics & Communication Engineering','B.E. CIVIL ENGINEERING','B.E. Electronics & Telecommunications']:
            return 'BE_BTECH_NON_IT'
        if row['Last Graduation'] in ['B.E. / B.TECH', 'B-tech']:
            return 'BE_BTECH'
        if row['Last Graduation'] in ['MCA', 'MSc CS/IT', 'MCA IT']:
            return 'MCA_MSc_CS_IT'
        if row['Last Graduation'] in ['MSc Non IT']:
            return 'MSc_Non_IT'
        if row['Last Graduation'] in ['M.E. / M.Tech CS/IT','PG CS/IT']:
            return 'ME_MTECH_PG_CS_IT'
        if row['Last Graduation'] in ['M.E. / M.Tech Non IT','PG Non IT']:
            return 'ME_MTECH_PG_Non_IT'
        if row['Last Graduation'] in ['BSc CS','BSc IT','BCA IT','BCA','BCA CS','Computer Application','Other IT']:
            return 'BCA_BSc_CS_Other_IT'
        if row['Last Graduation'] in ['BSc Non IT','BSc','Other Non IT','Diploma Non IT']:
            return 'BCA_BSc_Other_Diploma_Non_IT'


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

            # Encoding all the categorical features using label encoder
            le = LabelEncoder()

            # Dictionary to store encoder objects
            encoder_dict = dict()

            for col in list(df.columns):

                # checking if the column is categorical or not
                if df[col].dtype == 'object' and col != 'Admit ID':
                    encode = le.fit(df[col])
                    df[col] = le.transform(df[col])
                    encoder_dict[col] = encode
            
            return df , encoder_dict

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
                Returns the class label and the rest of the features.
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
    

    def correlation(self, dataset, threshold):
        """
            Description:
                This function is find the set of features that are important for model building
            Parameters:
                dataset: The encoded data without class label
                threshold: The value above which the features are selected
            Return:
                Returns features which important for model building.
        """
        col_corr = set() 
        corr_matrix = dataset.corr()
        for i in range (len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i,j]) > threshold:
                    colname = corr_matrix.columns[i] 
                    col_corr.add(colname)
        return col_corr


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
        analysis_obj = Analysis()

        # Calling the file load function
        df = analysis_obj.load_file()
        if df is not None:

            # Calling the preprocessing function
            preprocessed_df = analysis_obj.preprocessing(df)

            # Calling the bucketing function
            preprocessed_df['Last Graduation'] = preprocessed_df.apply(lambda row: analysis_obj.bucketing_graduation(row), axis=1)

            # Calling the encoding function
            encoded_df , encoder_dict = analysis_obj.encoding(preprocessed_df)

            # Saving the encoded data into csv file
            path = os.path.join(os.getcwd(), 'data') 
            isExist = os.path.exists(path)

            if isExist:
                path = os.getcwd() + '/data/' + 'processed_data.csv'      
                # Storing the dataframe into a csv file
                encoded_df.to_csv(Path(path), index=False)
                log.info(f"Processed Data saved successfully")
            
            else:
                log.exception(f"No such directory exists")

            # Saving the label encoder object into a pickle file
            path = os.path.join(os.getcwd(), 'encoder') 
            isExist = os.path.exists(path)

            if not isExist:          
                os.mkdir(path)            
            path = os.getcwd() + '/encoder/' + 'encoder_objects' + '.pickle'
            
            with open(Path(path), 'wb') as handle:
                pickle.dump(encoder_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # Calling the segregate function
            X, y = analysis_obj.segregate(encoded_df)

            # Feature importance
            threshold = 0.6
            correlated_cols = analysis_obj.correlation(X,threshold)
            log.info(f"Features that are important are : {list(correlated_cols)}")

    except:
        exception_type, _, exception_traceback = sys.exc_info()       
        line_number = exception_traceback.tb_lineno
        log.exception(f"Exception type : {exception_type} \nError on line number : {line_number}")


if __name__ == "__main__":
    main()