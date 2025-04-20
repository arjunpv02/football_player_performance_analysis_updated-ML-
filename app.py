from flask import Flask, render_template, request, redirect, url_for, jsonify
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
import numpy as np
from flask_sqlalchemy import SQLAlchemy

import matplotlib.pyplot as plt
import io
import base64
import pandas as pd

app = Flask(__name__)

# database connection
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'  # Change to MySQL/PostgreSQL if needed
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define global variables to store the dataset and trained model
uploaded_dataset = None
player_data_gk = {}
knn_model = None
rd = pd.DataFrame()

# Define global variables to store the dataset and trained model
scaler = StandardScaler()
knn_fw = KNeighborsRegressor(n_neighbors=5)  # Assuming k=5
fw_mse = None
scaler_fw = MinMaxScaler()

 
@app.route('/')
def index():
    return render_template('index11.html', title='FOOTBALL PLAYER PERFORMANCE ANALYSIS')
@app.route('/upload', methods=['POST'])
def upload():
    global uploaded_dataset
    
    if request.method == 'POST':
        file = request.files['file']
        if not file:
            return render_template('index11.html', message='No file uploaded', title='FOOTBALL PLAYER PERFORMANCE ANALYSIS')

        # Debugging: Check if file is being uploaded
        print("Uploaded File:", file)

        uploaded_dataset = pd.read_csv(file)

        # Debugging: Print the first few rows of the uploaded dataset
        #print("First Few Rows of Uploaded Dataset:")
        #print(uploaded_dataset.head())

        if uploaded_dataset.empty:
            return render_template('index11.html', message='Uploaded dataset is empty', title='FOOTBALL PLAYER PERFORMANCE ANALYSIS')

        # Get the first 10 rows of the dataset
        df_first_10 = uploaded_dataset.head(10)
        # Convert DataFrame to HTML format
        table_html = df_first_10.to_html(classes='table table-striped')
        return render_template('dataset.html', table_html=table_html)


@app.route('/cb', methods=['GET', 'POST'])
def cb_analysis():
    global uploaded_dataset, knn_model,scaler_cb
    scaler = StandardScaler()
    
    if uploaded_dataset is not None:
        rd = uploaded_dataset.copy()
        
        # Preprocess the uploaded dataset 
        # relevant_features_cb = [
        #     'Overall', 'Potential', 'Age', 'Height(in cm)', 'Weight(in kg)',
        #     'TotalStats', 'BaseStats', 'International Reputation',
        #     'Crossing', 'Finishing', 'Heading Accuracy', 'Short Passing', 'Volleys',
        #     'Dribbling', 'Curve', 'Freekick Accuracy', 'LongPassing', 'BallControl',
        #     'Acceleration', 'Sprint Speed', 'Agility', 'Reactions', 'Balance',
        #     'Shot Power', 'Jumping', 'Stamina', 'Strength', 'Long Shots', 'Aggression',
        #     'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking',
        #     'Standing Tackle', 'Sliding Tackle'
        # ]
        
        # Define weights for each feature (example weights, adjust as needed) 
        relevant_features_cb = [
        'Overall', 'Heading Accuracy', 'Reactions', 'Strength', 'Aggression',
        'Interceptions', 'Composure', 'Marking', 'Standing Tackle', 'Sliding Tackle'
        ]

        feature_weights = {
        'Overall': 0.1,
        'Heading Accuracy': 0.1,
        'Reactions': 0.1,
        'Strength': 0.1,
        'Aggression': 0.1,
        'Interceptions': 0.1,
        'Composure': 0.1,
        'Marking': 0.1,
        'Standing Tackle': 0.1,
        'Sliding Tackle': 0.1
        }

        
        
        # Extract centerbacks from the dataset
        centerback = rd[rd['Positions Played'].isin(['CB','LB','RB'])].copy()
        
        # Extract relevant features for centerbacks
        centerback_features = centerback[relevant_features_cb].copy()
        
        # Standardize the features
        scaler_cb = StandardScaler()
        cb_scaled = scaler_cb.fit_transform(centerback_features)
        
        # Apply weights to the standardized features
        cb_weighted = cb_scaled * np.array([feature_weights[col] for col in relevant_features_cb])
        
        # Calculate centerback performance as the sum of standardized features multiplied by their weights
        cb_performance = np.sum(cb_weighted, axis=1)
        
        # Add centerback performance to the dataframe
        centerback['cb performance'] = cb_performance
        
        cb_avg_per = centerback['cb performance'].mean()

        # Split the data into features (X) and target variable (y)
        X = cb_weighted  # Features with standardized and weighted values
        y = cb_performance  # Target variable 'mid_performance'

# Split the data into training and testing sets (e.g., 80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Now 'X_train', 'X_test', 'y_train', and 'y_test' contain the training and testing sets

# Create KNN model
        knn_model = KNeighborsRegressor(n_neighbors=5)  # You can adjust the number of neighbors as needed

# Train KNN model
        knn_model.fit(X_train, y_train)

# Predict on the training set
        y_train_pred_knn = knn_model.predict(X_train)

# Predict on the testing set
        y_test_pred_knn = knn_model.predict(X_test)

# Calculate mean squared error manually for KNN
        train_mse_knn = mean_squared_error(y_train, y_train_pred_knn)
        test_mse_knn = mean_squared_error(y_test, y_test_pred_knn)

# Calculate root mean squared error (RMSE) for KNN
        train_rmse_knn = np.sqrt(train_mse_knn)
        test_rmse_knn = np.sqrt(test_mse_knn)

        #print("KNN Train RMSE:", train_rmse_knn)
        #print("KNN Test RMSE:", test_rmse_knn)
        
        # Define thresholds for performance categories
        threshold_low = 1.5
        threshold_medium = 2
        threshold_high = 2.5

# Create a new column to store the performance category
        centerback['cb category'] = ''

# Categorize centerbacks based on 'cb performance'
        for index, row in centerback.iterrows():
                performance = row['cb performance']
                if performance < threshold_low:
                       category = 'Low'
                elif threshold_low <= performance < threshold_medium:
                       category = 'Medium'
                elif threshold_medium <= performance < threshold_high:
                       category = 'High'
                else:
                       category = 'Very High'
                centerback.at[index, 'cb category'] = category


        # Select top 4 centerbacks from Real Madrid CF
        real_centerback = centerback[centerback['Club Name'] == 'Real Madrid CF']
        real_centerback_sorted = real_centerback.sort_values(by='cb performance', ascending=False)
        real_centerback_sorted=real_centerback_sorted.head(10)
        selected_centerback = real_centerback_sorted[['Known As', 'cb performance', 'cb category', 'Positions Played']].head(4)
        print(selected_centerback)
        
        # Create a horizontal bar plot of Performance_max values
        plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
        plt.barh(selected_centerback['Known As'], selected_centerback['cb performance'], color='skyblue')
        plt.xlabel('cb performance')
        plt.ylabel('Player')
        plt.title('Performance_max for CB')

        # Save the plot to a BytesIO object
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)

        # Encode the plot image to base64
        plot_url = base64.b64encode(img.getvalue()).decode()

        # Pass the predictions and any other relevant data to cb.html for display
        return render_template('cb11.html',cb_avg_per = cb_avg_per, plot_url=plot_url, selected_centerback=selected_centerback,real_centerback_sorted=real_centerback_sorted, train_rmse_knn=train_rmse_knn, test_rmse_knn=test_rmse_knn)

    else:
        return render_template('cb11.html', message='No dataset uploaded')

@app.route('/cb_pred', methods=['POST'])
def predict_centerback():
    global knn_cb, scaler  # Ensure the model and scaler are loaded globally
    if request.method == 'POST':
        # Extract player details
        player_data = {
            'Player Name': request.form['player_name'],
            'Club Name': request.form['club_name'],
            'foot' : "left"
        }

        # Extract center-back attributes
        centerback_data_input = {
            'Overall': float(request.form['overall']),
            'Heading Accuracy': float(request.form['heading_accuracy']),
            'Reactions': float(request.form['reactions']),
            'Strength': float(request.form['strength']),
            'Aggression': float(request.form['aggression']),
            'Interceptions': float(request.form['interceptions']),
            'Composure': float(request.form['composure']),
            'Marking': float(request.form['marking']),
            'Standing Tackle': float(request.form['standing_tackle']),
            'Sliding Tackle': float(request.form['sliding_tackle'])
        }

        # Convert input to DataFrame
        input_df = pd.DataFrame([centerback_data_input])

        # Normalize input data (same as training process)
        input_scaled = scaler_cb.transform(input_df)

        # Predict Center-Back Performance
        predicted_performance = knn_model.predict(input_scaled)

        # Define thresholds for performance categories
        threshold_low = 0.6
        threshold_medium = 1.0
        threshold_high = 1.7

        # Determine performance category
        if predicted_performance < threshold_low:
            performance_category = 'Very Low'
        elif predicted_performance < threshold_medium:
            performance_category = 'Low'
        elif predicted_performance < threshold_high:
            performance_category = 'Medium'
        else:
            performance_category = 'High'

        # Prepare response
        response = {
            'Player Name': player_data['Player Name'],
            'Club Name': player_data['Club Name'],
            'Performance': predicted_performance,
            'max_performance':2,
            'Preferred Foot': player_data['foot'],
            'Performance Category': performance_category
        }

        return render_template('prediction_results.html', response=response)



@app.route('/goalkeeper', methods=['GET', 'POST'])
def goalkeeper_analysis():
    global uploaded_dataset, scaler, player_data_gk, knn_gk
      
    if uploaded_dataset is not None:
        rd = uploaded_dataset.copy()
        
         
        goalkeeper_columns = ['Known As', 'Overall', 'Potential', 'GK Rating', 'Goalkeeper Diving', 'Goalkeeper Handling', ' GoalkeeperKicking', 
                      'Goalkeeper Positioning', 'Goalkeeper Reflexes', 'Club Name', 'Preferred Foot']
 
        
        
        # extracting gk from dataset
        gk = rd[rd['Positions Played'] == 'GK'].copy()

        gk=gk[goalkeeper_columns]
        from sklearn.preprocessing import MinMaxScaler


        # Normalize the selected attributes
        scaler = MinMaxScaler()
        normalized_attributes = scaler.fit_transform(gk.drop(columns=['Known As', 'Club Name', 'Preferred Foot']))
        # Assign weights to the normalized attributes
        weights = {
    'Overall': 0.3,
    'Potential': 0.2,
    'GK Rating': 0.1,
    'GK Diving': 0.1,
    'GK Handling': 0.1,
    'GK Kicking': 0.1,
    'GK Positioning': 0.1,
    'GK Reflexes': 0.1
        }

# Calculate the weighted sum to derive 'GK Performance'
        gk['GK Performance'] = (normalized_attributes * np.array(list(weights.values()))).sum(axis=1)
        gk_per_max = gk['GK Performance'].max()
        gk_per_min = gk['GK Performance'].min()

# Sort the goalkeeper data based on 'GK Performance' column in descending order
        sorted_goalkeeper_data = gk.sort_values(by='GK Performance', ascending=False)

# Print players with descending order of GK performance
        #print(sorted_goalkeeper_data[['Known As', 'GK Performance','Performance Category']])


# Assume 'goalkeeper_data' contains relevant features and 'GK Performance' contains target variable


# Split the data into features and target variable

        X = gk.drop(columns=['GK Performance','Known As', 'Club Name', 'Preferred Foot'])
        y = gk['GK Performance']
        print(X.info())

# Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (important for KNN)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

# Train the KNN model
        knn_gk = KNeighborsRegressor(n_neighbors=5)  # Assuming k=5
        knn_gk.fit(X_train_scaled, y_train)

# Predict goalkeeper performance on the testing set
        y_pred = knn_gk.predict(X_test_scaled)

# Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        #print("Mean Squared Error:", mse)


        # MAX AND min of the values::
        print("max value  :",gk['GK Performance'].max())
        print("min value : ",gk['GK Performance'].min())
        
        # Define thresholds for performance categories
        threshold_low = 0.25
        threshold_medium = 0.5
        threshold_high = 0.75

        # Create a new column to store the performance category
        gk['Performance Category'] = ''

        # Categorize players based on 'GK Performance'
                # Categorize players based on 'GK Performance'
# Assuming gk is a DataFrame with 'GK Performance' column
        for index, row in gk.iterrows():
            performance = row['GK Performance']
            if performance < threshold_low:
                category = 'Very Low'
            elif threshold_low <= performance < threshold_medium:
                category = 'Low'
            elif threshold_medium <= performance < threshold_high:
                category = 'Medium'
            else:
                category = 'High'
            gk.at[index, 'Performance Category'] = category
        gk_pass=gk[['Known As', 'GK Performance', 'Performance Category','GK Rating']]
            
        real_gk = gk[gk['Club Name'] == 'Real Madrid CF']
        real_gk = real_gk.sort_values(by='GK Performance',ascending=False)
        #print(real_gk)

        real_gk_pass=real_gk[['Known As', 'GK Performance', 'Performance Category','GK Rating']]
        print(real_gk_pass)
        
        # Create a horizontal bar plot of Performance_max values
        plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
        plt.barh(real_gk_pass['Known As'], real_gk_pass['GK Performance'], color='red')
        plt.xlabel('Performance_max')
        plt.ylabel('Player')
        plt.title('Performance_max for GOALKEEPERS')

    # Save the plot to a BytesIO object
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)

    # Encode the plot image to base64
        plot_url = base64.b64encode(img.getvalue()).decode()

        #print(gk_pass)
        # Print the DataFrame with the added performance category
        return render_template('gk11.html',gk_per_max = gk_per_max, gk_per_min = gk_per_min, plot_url=plot_url, gk_pass=gk_pass , mse=mse, real_gk_pass=real_gk_pass )
        # Pass the predictions and any other relevant data to cb.html for display
    else:
        return render_template('gk11.html', message =' NO DATSET UPLOADED !..')
    return render_template('gk11.html')


@app.route('/gk_predict', methods=['POST'])
def predict_goalkeeper():
    global knn_gk,scaler
    if request.method == 'POST':
        # Extract data from form
        
        player_data = {
                'Player Name': request.form['player_name'],
                'Club Name': request.form['club_name'],
                'Preferred Foot': request.form['preffered_foot']
        }
        
        goalkeeper_data_input = {
            'Overall': float(request.form['overall']),
            'Potential': float(request.form['potential']),
            'GK Rating': float(request.form['gk_rating']),
            'GK Diving': float(request.form['gk_diving']),
            'GK Handling': float(request.form['gk_handling']),
            'GK Kicking': float(request.form['gk_kicking']),
            'GK Positioning': float(request.form['gk_positioning']),
            'GK Reflexes': float(request.form['gk_reflexes']),
        }
        
               # Convert input to DataFrame
        input_df = pd.DataFrame([goalkeeper_data_input])

        # Normalize input data (same as training process)
        input_scaled = scaler.fit_transform(input_df)

        # Predict GK Performance
        predicted_performance = knn_gk.predict(input_scaled)[0]
        
        threshold_low = 0.25
        threshold_medium = 0.5
        threshold_high = 0.75
        
        # Determine performance category
        if predicted_performance < threshold_low:
            performance_category = 'Very Low'
        elif predicted_performance < threshold_medium:
            performance_category = 'Low'
        elif predicted_performance < threshold_high:
            performance_category = 'Medium'
        else:
            performance_category = 'High'

        # Return response
        response = {
            'Player Name': player_data['Player Name'],
            'Club Name': player_data['Club Name'],
            'Preferred Foot': player_data['Preferred Foot'],
            'Performance': round(predicted_performance, 3),
            'max_performance' : threshold_high + .25,
            'Performance Category': performance_category
        }
        
        return render_template('prediction_results.html', response = response)




@app.route('/midfielder', methods=['GET', 'POST'])
def midfielder_analysis():
    global uploaded_dataset, scaler, knn_mid

    if uploaded_dataset is not None:
            
        rd = uploaded_dataset.copy()
        
        # EXTRACTING MIDFIELDERS FROM DATASET
        midfielders = rd[
        rd['Positions Played'].isin(['CAM', 'CM', 'CDM', 'RM', 'LM'])].copy()

        # EXTRACTING RELEVANT FEATURES OF MIDFIELDER FROM DATASET (LIMITED TO 10)
        midfielder_columns = [
            'Known As', 'Overall', 'Potential', 'TotalStats', 'Short Passing',
            'BallControl', 'LongPassing', 'Dribbling', 'Vision', 'Stamina',
            'Club Name', 'Preferred Foot'
        ]

        # Extract relevant features for midfielders
        midfielders = midfielders[midfielder_columns]

        from sklearn.preprocessing import MinMaxScaler

        # Normalize the selected attributes
        scaler = MinMaxScaler()
        normalized_attributes = scaler.fit_transform(midfielders.drop(columns=['Known As', 'Club Name', 'Preferred Foot']))

        # Define weights for each feature
        feature_weights = {
            'Overall': 1,
            'Potential': 1,
            'TotalStats': 1,
            'Short Passing': 1,
            'BallControl': 1,
            'LongPassing': 1,
            'Dribbling': 1,
            'Vision': 1,
            'Stamina': 1
        }
    
        # Calculate the weighted sum to derive 'mid Performance'
        midfielders['mid performance'] = (normalized_attributes * np.array(list(feature_weights.values()))).sum(axis=1)

        max_mid_per = midfielders["mid performance"].max()
        min_mid_per = midfielders["mid performance"].min()

        # Sort the midfielder data based on 'mid Performance' column in descending order
        sorted_midfielders = midfielders.sort_values(by='mid performance', ascending=False)

        # Split the data into features (X) and target variable (y)
        X = midfielders.drop(columns=['Known As', 'Club Name', 'Preferred Foot', 'mid performance'])
        y = midfielders['mid performance']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize features for KNN
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        from sklearn.neighbors import KNeighborsRegressor

        # Create KNN model
        knn_mid = KNeighborsRegressor(n_neighbors=5)
        knn_mid.fit(X_train_scaled, y_train)
        
        # Predict mid performance on the testing set
        y_pred = knn_mid.predict(X_test_scaled)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)

        # Predict on the training set
        y_train_pred_knn = knn_mid.predict(X_train)

        # Predict on the testing set
        y_test_pred_knn = knn_mid.predict(X_test)

        # Calculate mean squared error manually for KNN
        cm_train_mse_knn = mean_squared_error(y_train, y_train_pred_knn)
        cm_test_mse_knn = mean_squared_error(y_test, y_test_pred_knn)

        # Calculate root mean squared error (RMSE) for KNN
        cm_train_rmse_knn = np.sqrt(cm_train_mse_knn)
        cm_test_rmse_knn = np.sqrt(cm_test_mse_knn)

        # Define thresholds for performance categories
        threshold_low = 2
        threshold_medium = 4
        threshold_high = 6

        # Create a new column to store the performance category
        midfielders['Performance Category'] = ''
        for index, row in midfielders.iterrows():
            performance = row['mid performance']
            if performance < threshold_low:
                category = 'Very Low'
            elif threshold_low <= performance < threshold_medium:
                category = 'Low'
            elif threshold_medium <= performance < threshold_high:
                category = 'Medium'
            else:
                category = 'High'
            midfielders.at[index, 'Performance Category'] = category
        
        # Extract midfielders from Real Madrid CF
        real_midfielders = midfielders[midfielders['Club Name'] == 'Real Madrid CF']

        # Sort real midfielders based on performance
        real_midfielders_sorted = real_midfielders.sort_values(by='mid performance', ascending=False)

        # Select top midfielders
        selected_midfielders = real_midfielders_sorted[['Known As', 'mid performance', 'Performance Category', 'Club Name', 'Preferred Foot']]
        print(selected_midfielders)

        # Create a horizontal bar plot of Performance_max values
        plt.figure(figsize=(8, 6))
        plt.barh(selected_midfielders['Known As'], selected_midfielders['mid performance'], color='skyblue')
        plt.xlabel('mid performance')
        plt.ylabel('Player')
        plt.title('Performance_max for midfielders')

        # Save the plot to a BytesIO object
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)

        # Encode the plot image to base64
        plot_url = base64.b64encode(img.getvalue()).decode()

        return render_template('m11.html', max_mid_per=max_mid_per, min_mid_per=min_mid_per, plot_url=plot_url, mse=mse, cm_train_mse_knn=cm_train_mse_knn, cm_test_mse_knn=cm_test_mse_knn, selected_midfielders=selected_midfielders)
    else:
        return render_template('m11.html', message='NO DATASET UPLOADED!')


@app.route('/midfielder_predict', methods=['POST'])
def predict_midfielder():
    global knn_mid, scaler
    if request.method == 'POST':
        # Extract data from form
        
        player_data = {
            'Player Name': request.form['player_name'],
            'Club Name': request.form['club_name'],
            'Preferred Foot': request.form['preferred_foot']  # Fixed spelling
        }

        midfielder_data_input = {
            'Overall': float(request.form['overall_rating']),  # Fixed name
            'Potential': float(request.form['potential_rating']),  # Fixed name
            'TotalStats': float(request.form['total_stats']),  # Fixed name
            'Short Passing': float(request.form['short_passing']),  # Ensure this field exists in HTML
            'BallControl': float(request.form['ballcontrol']),  # Ensure this field exists in HTML
            'LongPassing': float(request.form['longpassing']),  # Ensure this field exists in HTML
            'Dribbling': float(request.form['dribbling']),  # Ensure this field exists in HTML
            'Vision': float(request.form['vision']),  # Ensure this field exists in HTML
            'Stamina': float(request.form['stamina'])  # Ensure this field exists in HTML
        }

        
        # Convert input to DataFrame
        input_df = pd.DataFrame([midfielder_data_input])

        # Normalize input data (same as training process)
        input_scaled = scaler.transform(input_df)

        # Predict Midfielder Performance
        predicted_performance = knn_mid.predict(input_scaled)[0]
        
        threshold_low = 2
        threshold_medium = 4
        threshold_high = 6
        
        # Determine performance category
        if predicted_performance < threshold_low:
            performance_category = 'Very Low'
        elif predicted_performance < threshold_medium:
            performance_category = 'Low'
        elif predicted_performance < threshold_high:
            performance_category = 'Medium'
        else:
            performance_category = 'High'

        # Return response
        response = {
            'Player Name': player_data['Player Name'],
            'Club Name': player_data['Club Name'],
            'Preferred Foot': player_data['Preferred Foot'],
            'Performance': round(predicted_performance, 3),
            'max_performance': threshold_high +2,
            'Performance Category': performance_category
        }
        
        return render_template('prediction_results.html', response=response)



@app.route('/forward', methods=['GET', 'POST'])
def forward_analysis():
    global uploaded_dataset, scaler, fw_mse, scaler_fw , knn_fw, fw_mse
    
    if uploaded_dataset is not None:
        rd = uploaded_dataset.copy()
        
        forward_columns = ['Known As', 'Overall', 'Pace Total', 'Shooting Total',
                           'Passing Total', 'Finishing', 'Crossing', 'Heading Accuracy', 'Potential',
                           'ST Rating', 'Club Name', 'Preferred Foot']

        # Extracting forwards from dataset
        fw = rd[rd['Positions Played'].isin(['LW', 'RW', 'CF', 'ST'])].copy()
        fw = fw[forward_columns]
        
        # Normalize the selected attributes
        
        normalized_attributes = scaler_fw.fit_transform(fw.drop(columns=['Known As', 'Club Name', 'Preferred Foot']))

        # Assign weights to the features
        fw_weights = {
            'Overall': 0.1,
            'Pace Total': 0.1,
            'Shooting Total': 0.1,
            'Passing Total': 0.1,
            'Finishing': 0.1,
            'Crossing': 0.75,
            'Heading Accuracy': 0.75,
            'Potential': 0.1,
            'ST Rating': 0.15
        }

        # Calculate the weighted sum to derive 'Forward Performance'
        fw['Forward Performance'] = (normalized_attributes * np.array(list(fw_weights.values()))).sum(axis=1)

        # Sort the forward data based on 'Forward Performance' column in descending order
        sorted_fw = fw.sort_values(by='Forward Performance', ascending=False)

        # Split the data into features and target variable
        X = fw.drop(columns=['Forward Performance', 'Known As', 'Club Name', 'Preferred Foot'])
        y = fw['Forward Performance']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize features (important for KNN)
        X_train_scaled = scaler_fw.fit_transform(X_train)
        X_test_scaled = scaler_fw.transform(X_test)

        # Train the KNN model
        knn_fw.fit(X_train_scaled, y_train)

        # Predict forward performance on the testing set
        y_pred = knn_fw.predict(X_test_scaled)

        # Evaluate the model
        fw_mse = round(mean_squared_error(y_test, y_pred),2)

        # MAX AND min of the values
        max_value = round(fw['Forward Performance'].max(),2)
        min_value = round(fw['Forward Performance'].min(),2)
        
        # Define thresholds for performance categories
        threshold_low = 0.9
        threshold_medium = 1.4
        threshold_high = 1.75



        # Create a new column to store the performance category
        fw['Performance Category'] = ''

# Categorize players based on 'Forward Performance'
# Assuming fw is a DataFrame with 'Forward Performance' column
        for index, row in fw.iterrows():
              performance = row['Forward Performance']
              if performance < threshold_low:
                    category = 'Very Low'
              elif threshold_low <= performance < threshold_medium:
                    category = 'Low'
              elif threshold_medium <= performance < threshold_high:
                    category = 'Medium'
              else:
                    category = 'High'
              fw.at[index, 'Performance Category'] = category

# Print the updated DataFrame with performance categories
        fw_pass=fw[['Known As', 'Forward Performance', 'Performance Category']].head()
        print(fw_pass)
        
        # Extract forwards from Real Madrid CF
        real_forwards = fw[fw['Club Name'] == 'Real Madrid CF']

# Sort real forwards based on performance
        real_forwards_sorted = real_forwards.sort_values(by='Forward Performance', ascending=False)
        real_forwards_sorted = real_forwards_sorted[['Known As', 'Forward Performance', 'Performance Category','Club Name', 'Preferred Foot']]
        
        # Create a horizontal bar plot of Performance_max values
        plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
        plt.barh(real_forwards_sorted['Known As'], real_forwards_sorted['Forward Performance'], color='skyblue')
        plt.xlabel('Performance_max')
        plt.ylabel('Player')
        plt.title('Performance_max for Forwards')

        # Save the plot to a BytesIO object
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        
        # Encode the plot image to base64
        plot_url = base64.b64encode(img.getvalue()).decode()
        
                # Create a horizontal bar plot of Performance_max values
        plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
        plt.barh(real_forwards['Known As'], real_forwards['Overall'], color='blue')
        plt.xlabel('Overall')
        plt.ylabel('Player')
        plt.title('Overall rating for Forwards')

        # Save the plot to a BytesIO object
        img2 = io.BytesIO()
        plt.savefig(img2, format='png')
        img2.seek(0)
        
        # Encode the plot image to base64
        plot_url_2 = base64.b64encode(img2.getvalue()).decode()
        
        
    # Pass the image file path to the HTML template
        return render_template('f11.html', plot_url_2 = plot_url_2, plot_url = plot_url, max_value=max_value,real_forwards_sorted = real_forwards_sorted , min_value=min_value, fw_mse=fw_mse, fw_pass=fw_pass)

    else:
        return render_template('f11.html', message='NO DATASET UPLOADED')

# Define route to handle form submission for random player data
@app.route('/predict_forward', methods=['POST'])
def predict_forward():
    global knn_fw, scaler_fw
    if request.method == 'POST':
        # Extract data from form
        player_data_input = {
            'Overall': float(request.form['overall']),
            'Pace Total': float(request.form['pace']),
            'Shooting Total': float(request.form['shooting']),
            'Passing Total': float(request.form['passing']),
            'Finishing': float(request.form['finishing']),
            'Crossing': float(request.form['crossing']),
            'Heading Accuracy': float(request.form['heading']),
            'Potential': float(request.form['potential']),
            'ST Rating': float(request.form['st_rating'])
        }
        
        # Preprocess the input data
        X_input = np.array(list(player_data_input.values())).reshape(1, -1)  # Reshape to match expected input shape
        
        # Scale the input data
        X_input_scaled = scaler_fw.transform(X_input)
        
        # Predict forward performance
        forward_performance = knn_fw.predict(X_input_scaled)
        
                # Define thresholds for performance categories
        threshold_low = 0.9
        threshold_medium = 1.4
        threshold_high = 1.7

        Per = ''

        # Categorize players based on 'Forward Performance'
        # Assuming fw is a DataFrame with 'Forward Performance' column
        if forward_performance < threshold_low:
                category = 'Very Low'
        elif threshold_low <= forward_performance < threshold_medium:
                category = 'Low'
        elif threshold_medium <= forward_performance < threshold_high:
                category = 'Medium'
        else:
                category = 'High'
        per = category



        
        # Format the result 
        result = {
            'forward_performance': round(forward_performance[0],2),
            'performance_category' : per
        }
        
        # Render template with result
        return render_template('prediction_result.html', result=result)


@app.route('/home')
def home():
    return redirect(url_for('index'))

if __name__ == '__main__':   
        app.run(debug=True)
    

