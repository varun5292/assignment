import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def load_data(file_path, sheet_name=None):
    """Load data from an Excel file and return a DataFrame."""
    return pd.read_excel(file_path, sheet_name=sheet_name)

def calculate_statistics(data):
    """Calculate and print mean and variance of a numerical column in a DataFrame."""
    mean_D = data['Price'].mean()
    variance_D = data['Price'].var()
    print('Mean:', mean_D)
    print('Variance:', variance_D)

def calculate_probability(data, column_name, condition):
    """Calculate and print the probability of a condition in a boolean column."""
    condition_list = list(map(lambda v: v == condition, data[column_name]))
    condition_true = [value for value in condition_list if value is True]
    probability = (len(condition_true) / len(condition_list)) * 100
    print(f'Probability: {probability}%')

if __name__ == "__main__":
    # Load data and set column names
    df1 = pd.read_excel("C:\\Users\\mvy48\\OneDrive\\Desktop\\vscodeprograms\\ml_labsessions\\Lab_Session1_Data.xlsx", usecols=range(5))

    df1.columns = ['col1', 'col2', 'col3', 'col4', 'col5']

    # Extract matrices A and C
    A = df1[['col2', 'col3', 'col4']]
    C = df1[['col5']]

    # Display matrices A and C
    print("Matrix A:")
    print(A)
    print("\nMatrix C:")
    print(C)

    # Get dimensionality and number of vectors in the vector space
    dimensionality = A.shape[1]
    num_vectors = A.shape[0]
    print("\nDimensionality of the vector space:", dimensionality)
    print("Number of vectors in the vector space:", num_vectors)

    # Calculate rank of Matrix A
    A_rank = np.linalg.matrix_rank(A)
    print("\nRank of Matrix A:", A_rank)

    # Perform linear regression to estimate product costs
    C = C.to_numpy()
    X = np.linalg.pinv(A).dot(C.reshape(-1, 1))
    print('\nCost of each product available for sale:')
    print(X.flatten())

    # Create a new column 'Customer Type' based on a condition
    df1['Customer Type'] = df1['col5'].apply(lambda x: 'RICH' if x > 200 else 'POOR')

    # Extract features (X) and target variable (y) for training a logistic regression model
    X = df1[['col2', 'col3', 'col4']]
    y = df1['Customer Type']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the feature matrices
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train a logistic regression model
    log_reg_model = LogisticRegression(random_state=42)
    log_reg_model.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_pred = log_reg_model.predict(X_test_scaled)

    # Add predicted classes to the DataFrame
    df1['Predicted Customer Type'] = log_reg_model.predict(scaler.transform(X))

    # Display the DataFrame with predicted classes
    print("\nDataFrame with Predicted Classes:")
    print(df1)

    # Load additional data from another sheet
    df1 = load_data("C:\\Users\\mvy48\\OneDrive\\Desktop\\vscodeprograms\\ml_labsessions\\Lab_Session1_Data.xlsx", sheet_name=1)
    
    # Calculate and print mean and variance for the 'Price' column in df1
    calculate_statistics(df1)

    # Analyze data based on specific conditions
    wednesday_df = df1[df1['Day'] == 'Wed']
    calculate_statistics(wednesday_df)
    calculate_probability(df1, 'Chg%', True)
    calculate_probability(wednesday_df, 'Chg%', True)

    # Calculate mean for April and overall population
    April_df = df1[df1['Month'] == 'Apr']
    calculate_statistics(April_df)
    calculate_statistics(df1)

    # Calculate and print the mean for Wednesday and overall population
    wednesday_mean = wednesday_df['Price'].mean()
    population_mean = df1['Price'].mean()
    print('Wednesday Mean:', wednesday_mean)
    print('Population Mean:', population_mean)

    # Calculate and print conditional probability for profits on Wednesday
    l3 = list(map(lambda v: v > 0, wednesday_df['Chg%']))
    l3_True = [value for value in l3 if value is True]
    probability_wed = (len(l3_True) / len(l3)) * 100
    conditional_prob = probability_wed / wednesday_df.shape[0]
    print(f'Profits on Wednesday: {probability_wed}%')
    print(f'Conditional Probability: {conditional_prob}%')
    
    # Create a scatter plot to visualize the relationship between 'Day' and 'Chg%'
    sns.scatterplot(x='Day', y='Chg%', data=df1)

    # Display the plot
    plt.show()
