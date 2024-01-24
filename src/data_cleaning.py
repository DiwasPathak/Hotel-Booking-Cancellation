import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split

# Define the subsystem classes
class RemoveUselessColumns:
    def process(self, data: pd.DataFrame):
        """
        Removes the columns that are not required

        Args:
            data: Raw data.
        
        Returns:
            data: Data after removing useless cols.
        """
        try:
            data = data.drop(columns =  ['name', 'email', 'credit_card', 'adr', 'company', 'arrival_date_year', 'stays_in_weekend_nights', 'babies', 'required_car_parking_spaces', 'reservation_status', 'meal', 'arrival_date_month', 'arrival_date_week_number', 'arrival_date_day_of_month', 'is_repeated_guest', 'reserved_room_type', 'reservation_status_date', 'phone-number', 'agent','children', 'previous_bookings_not_canceled', 'days_in_waiting_list', 'adults', 'market_segment'], axis = 1)
            return data
        except Exception as e:
            logging.error(f'Error in  RemoveUselessColumns{e}')

class FindColsWithOutliers:
    def process(self, data: pd.DataFrame):
          """
          Uses IQR method to find features having outliers

          Returns:
              columns_with_outliers: List having names of featueres with outliers.
          """

          try:
              columns_with_outliers = []
              numeric_features = ['lead_time', 'stays_in_week_nights','previous_cancellations', 'booking_changes','total_of_special_requests']
              for feature in numeric_features:
                  Q1 = data[feature].quantile(0.25)
                  Q3 = data[feature].quantile(0.75)
                  IQR = Q3 - Q1
                  
                  # Identify columns with outliers
                  lower_bound = Q1 - 1.5 * IQR
                  upper_bound = Q3 + 1.5 * IQR
    
                  outliers = data[(data[feature] < lower_bound) | (data[feature] > upper_bound)][feature]
    
                  if not outliers.empty:
                      columns_with_outliers.append(feature)
              return columns_with_outliers  
          except Exception as e:
            logging.error(f'Error in FindColsWithOutliers {e}')

class HandleOutliers:
    def process(self, data: pd.DataFrame, cols_with_outliers: list):
        """
        Handles outliers in the numeric data

        Args:
            data: Original DataFrame.
            cols_with_outliers: List of columns having outliers
        
        Returns: 
            None
        """
        try:
            for feature in cols_with_outliers:
                Q1 = data[feature].quantile(0.25)
                Q3 = data[feature].quantile(0.75)
                IQR = Q3 - Q1
        
                # Calculate upper and lower bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
        
                # Replace outliers with bounds
                data[feature] = data[feature].apply(lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x))
        except Exception as e:
            logging.error(f'Error in HandleOutliers {e}')

class HandleMissingValues:
    """ Imputes the missig values in the data"""
    def process(self, data:pd.DataFrame):
        """ 
        Handles missing values in variables
        """
        try:
            data['distribution_channel'].replace('Undefined', data['distribution_channel'].mode()[0], inplace=True)
            data['country'].fillna(data['country'].mode()[0], inplace=True)
        except Exception as e:
            logging.error(f'Error in HandleMissingValues {e}')

class EncodeCategoricalColumns:
    """ Encode categorial variables with appropriate encoding techniques"""
    def process(self, data: pd.DataFrame):
        """
        Encodes categorial variables with relevant enoding techniques

        Returns:
            data: DataFrame with encoded variables.
        """
        try:
            top_countries = data['country'].value_counts().nlargest(10).index.tolist()
            top_room_type = data['assigned_room_type'].value_counts().nlargest(8).index.tolist()
            
            # Encode country variable
            country_encoded = pd.get_dummies(data['country'].apply(lambda x: x if x in top_countries else 'Other'), prefix='country', drop_first=True)
            other_vars_encoded = pd.get_dummies(data[['hotel', 'distribution_channel', 'deposit_type', 'customer_type']], drop_first = True)
            room_type_encoded = pd.get_dummies(data['assigned_room_type'].apply(lambda x: x if x in top_room_type else 'Other'), prefix='assigned_room_type', drop_first=True)
            
            # Remove vars from dataframe
            data = data.drop(columns = ['hotel', 'distribution_channel', 'deposit_type', 'customer_type','country','assigned_room_type'])
    
            # Concatenate encoded  with dataframe
            data = pd.concat([data, country_encoded, other_vars_encoded, room_type_encoded], axis=1)
    
            return data
        except Exception as e:
            logging.error(f'Error in EncodeCategoricalColumns {e}')

class DataSplitter:
    def process(self, data: pd.DataFrame):
        """ 
        Divids the proceseed data into training and testing sets

        Args:
            data: DataFrame
        
        Returns:
            X_train: Training fetures
            X_test: Testing features
            y_train: Training labels
            y_test: Testing labales
        """
        try:
            X = data.drop(columns = ['is_canceled'])
            y = data['is_canceled']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f'Error in DataSplitter {e}')

# Create a facade class
class DataPreprocessingFacade:
    def __init__(self):
        """ Initiate preprocessing steps"""
        self.remover = RemoveUselessColumns()
        self.outlier_finder = FindColsWithOutliers()
        self.outlier_handler = HandleOutliers()
        self.missing_handler = HandleMissingValues()
        self.encoder = EncodeCategoricalColumns()
        self.data_splitter = DataSplitter()

    def preprocess_data(self, data):
        """ Implements preprocessing steps"""
        logging.info("Preprocessing data......")
        try:
            # Remove useless cols
            data = self.remover.process(data = data)
    
            # Find column with outliers
            columns_with_outliers = self.outlier_finder.process(data = data) 
    
            # Handle outlier cols
            self.outlier_handler.process(data = data, cols_with_outliers= columns_with_outliers)
    
            # Handle missing values
            self.missing_handler.process(data = data) 
    
            # Encode categorical columns
            preprocessed_data = self.encoder.process(data = data) 
    
            # Divide the data 
            X_train, X_test, y_train, y_test = self.data_splitter.process(data = preprocessed_data)
    
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f'Error in proprocessing data {e}')

# Step 3: Client code

if __name__ == "__main__":
    # Sample data
    data= pd.read_csv('/home/diwas/Documents/DevStuff/Hotel Booking Cancellation/data/raw/hotel_booking.csv')

    # Using the facade to preprocess the data
    facade = DataPreprocessingFacade()
    X_train, X_test, y_train, y_test = facade.preprocess_data(data)

    # Display the processed data
    print("Processed Data:")
    print('')





# Apply the preprocessing steps
# processed_data = remover.process(data)
# columns_with_outliers = outlier_finder.process(processed_data)
# outlier_handler.process(processed_data, columns_with_outliers)
# missing_handler.process(processed_data)
# preprocessed_data = encoder.process(processed_data)
# preprocessed_data.to_csv("processed_data.csv")
