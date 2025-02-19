""" #genxcode - LEVEL : House value prediction (linear regression) """

# Importing necessary modules for data processing and visualization

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd

# Loading the dataset (sourced from Kaggle: housing.csv)

house = pd.read_csv('../data/housing.csv')

# Showing the data

print(house.head())

# Checking for missing values

print(house.isnull().sum())

# Selecting features and target for prediction

# Features: 
    
# - RM: Average number of rooms per dwelling
# - LSTAT: Percentage of lower status population
# - PTRATIO: Pupil-teacher ratio by town

# Target:
    
# - MEDV: Median value of owner-occupied homes in $1000s

data = house[['RM', 'LSTAT', 'PTRATIO']]
target = house[['MEDV']]

# Assigning features to X and target to y

X = data
y = target

# Displaying the shape of the datasets (features and target)

print(X.shape) # Shape of the feature matrix
print(y.shape) # Shape of the target vector

# 3D data modeling

fig = plt.figure() #Creation of the figure

ax = plt.axes(projection='3d') # Making the 3D object for the visualization

# Visualizing training data

sc = ax.scatter(X['RM'], X['PTRATIO'], X['LSTAT'], c=y.values.flatten(), cmap='plasma')
plt.colorbar(sc, ax=ax, label='House Price ($1000s)')  # Adding a colorbar

# Titling the different areas

ax.set_xlabel('Average Rooms (RM)') # X-axis title
ax.set_ylabel('Pupil-Teacher Ratio (PTRATIO)') # Y-axis title
ax.set_zlabel('Population Poverty (%) (LSTAT)') # Z-axis title

plt.show() # Displaying the data in 3D

# Splitting the data into two parts :

# - The train part : About 80% of the data
# - The test part : About 20% of the data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 5)

print('Train set:', X_train.shape) # Shape of the data for the training part
print('Test set:', X_test.shape) # Shape of the data for the testing part

# Using SKlearn's LinearRegression model

model = LinearRegression() #no hyperparameter, as least-squares method is used

# Train / Test / Use method

# Training Part

model.fit(X_train, y_train)

print('Train score: ', model.score(X_train, y_train))

# Testing Part :

print('Test score: ', model.score(X_test, y_test))

# 3D Visualization of the using part

y_pred = model.predict(X_test)  # Model's predictions

#  Root mean square error

mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False) # root mean squared error

# Calculation of the coefficient of determination R²

r2 = r2_score(y_test, y_pred)

# Displaying metrics

print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'R² Score: {r2:.2f}')  # Adding the R²

# Asking the user werever they want a 3D or 2D visualization :

a = ""

while a not in ["3D", "2D"]:
    
    print("3D visualization or 2D ? : ")
    a = input("3D or 2D ? : ")
    
    if a == "3D":
        
        # Visualization of actual and predicted values in 3D :
    
        fig = plt.figure() #Creation of the figure
    
    
        # Graph of actual values :
            
        ax1 = fig.add_subplot(121, projection='3d') # Making the 3D object for the visualization
        ax1.scatter(X_test['RM'], X_test['PTRATIO'], y_test, c=y_test, cmap='Blues')
    
        # Price prediction based on Average Rooms and Pupil-Teacher Ratio
    
        ax1.set_xlabel('Average Rooms (RM)')
        ax1.set_ylabel('Pupil-Teacher Ratio (PTRATIO)')
        ax1.set_zlabel('Actual Price ($)')
        ax1.set_title('Actual House Prices')
    
        # Graph of predictions :
            
        ax2 = fig.add_subplot(122, projection='3d') # Making the 3D object for the visualization
        ax2.scatter(X_test['RM'], X_test['PTRATIO'], y_pred, c=y_pred, cmap='PuRd')
    
        # Price prediction based on Average Rooms and Pupil-Teacher Ratio
    
        ax2.set_xlabel('Average Rooms (RM)')
        ax2.set_ylabel('Pupil-Teacher Ratio (PTRATIO)')
        ax2.set_zlabel('Predicted Price ($)')
        ax2.set_title('Predicted House Prices')
        
    
        plt.show()
    
    elif a == "2D" :
        
        # Visualization of tested and predicted values in 3D :
    
        fig = plt.figure() #Creation of the figure
        
        # Graph
        
        plt.scatter(y_test, y_pred, alpha=0.6)
        
        plt.plot(y_test, y_test, alpha=0.6, color='red', linestyle='--')
        
        plt.show()
    
    else:
        
        print("Try again. Type : 2D or 3D.")
    
