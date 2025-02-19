""" #genxcode - LEVEL : Prédiction de la valeur des maisons (régression linéaire) """

# Modules nécessaires au traitement et à la visualisation des données

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd

# Chargement du jeu de données (source Kaggle : housing.csv)

house = pd.read_csv('../data/housing.csv')

# Afficher les données

print(house.head())

# Vérifier les valeurs manquantes

print(house.isnull().sum())

# Sélection des caractéristiques et de la cible pour la prédiction

# Caractéristiques : 
    
# - RM : Nombre moyen de pièces par logement
# - LSTAT : Pourcentage de la population de statut inférieur
# - PTRATIO : Taux d'encadrement par ville

# Cible :
    
# - MEDV : Valeur médiane des logements occupés par leur propriétaire, 
# en milliers de dollars

data = house[['RM', 'LSTAT', 'PTRATIO']]
target = house[['MEDV']]

# Attribution des caractéristiques à X et de la cible à y

X = data
y = target

# Affichage de la forme des ensembles de données (caractéristiques et cibles)

print(X.shape) # Forme de la matrice des caractéristiques
print(y.shape) # Forme du vecteur cible

# Modélisation 3D des données actuelles

fig = plt.figure() #Création de la figure

ax = plt.axes(projection='3d') # Créer l'objet 3D pour la visualisation

# Visualisation des données d'entraînement

sc = ax.scatter(X['RM'], X['PTRATIO'], X['LSTAT'], c=y.values.flatten(), cmap='plasma')
plt.colorbar(sc, ax=ax, label='House Price ($1000s)')  # Ajout d'une barre de couleur

# Titrer les différents axes

ax.set_xlabel("Chambres moyennes (RM)") # Titre de l'axe X
ax.set_ylabel("Taux d'encadrement (PTRATIO)") # Titre de l'axe des Y
ax.set_zlabel("Pauvreté de la population (%) (LSTAT)") # Titre de l'axe Z

plt.show() # Affichage des données en 3D

# Diviser les données en deux parties :

# - La partie train : Environ 80% des données
# - La partie test : Environ 20% des données

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 5)

print('Train set:', X_train.shape) # Forme des données d'entraînement
print('Test set:', X_test.shape) # Forme des données de testes

# Utilisation du modèle de régression linéaire de SKlearn

model = LinearRegression() #pas d'hyperparamètre, car la méthode des moindres carrés est utilisée

# Train / Test / Use méthode

# Partie Training : Entraînement

model.fit(X_train, y_train)

print('Train score: ', model.score(X_train, y_train))

# Partie Testing : Test

print('Test score: ', model.score(X_test, y_test))

# Visualisation en 3D de la partie Use : Utilisée/Prédite

y_pred = model.predict(X_test)  # Prédictions du modèle

# Erreur quadratique moyenne

mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False) # Erreur quadratique moyenne

# Calcul du coefficient de détermination R²

r2 = r2_score(y_test, y_pred)

# Affichage des métriques

print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'R² Score: {r2:.2f}')  # Adding the R²

# Demander à l'utilisateur s'il souhaite une visualisation en 3D ou en 2D :

a = ""

while a not in ["3D", "2D"]:
    
    print("3D visualization or 2D ? : ")
    a = input("3D or 2D ? : ")
    
    if a == "3D":
        
        # Visualisation des valeurs réelles et prédites en 3D :
    
        fig = plt.figure() #Création de la figure
    
    
        # Graphique des valeurs réelles :
            
        ax1 = fig.add_subplot(121, projection='3d') # Making the 3D object for the visualization
        ax1.scatter(X_test['RM'], X_test['PTRATIO'], y_test, c=y_test, cmap='Blues')
    
        # Prévision de prix basée sur le nombre moyen de pièces et 
        # le nombre d'élèves par enseignant
    
        ax1.set_xlabel('Average Rooms (RM)')
        ax1.set_ylabel('Pupil-Teacher Ratio (PTRATIO)')
        ax1.set_zlabel('Actual Price ($)')
        ax1.set_title('Actual House Prices')
    
        # # Graphique des prédictions :
            
        ax2 = fig.add_subplot(122, projection='3d') # Création de l'objet 3D pour la visualisation
        ax2.scatter(X_test['RM'], X_test['PTRATIO'], y_pred, c=y_pred, cmap='PuRd')
    
        # Prévision de prix basée sur le nombre moyen de pièces 
        # et le nombre d'élèves par enseignant

    
        ax2.set_xlabel('Average Rooms (RM)')
        ax2.set_ylabel('Pupil-Teacher Ratio (PTRATIO)')
        ax2.set_zlabel('Predicted Price ($)')
        ax2.set_title('Predicted House Prices')
        
    
        plt.show()
    
    elif a == "2D" :
        
        # Visualisation des valeurs testées et prédites en 3D :
    
        fig = plt.figure() #Création de la figure
        
        # Graphique
        
        plt.scatter(y_test, y_pred, alpha=0.6)
        
        plt.plot(y_test, y_test, alpha=0.6, color='red', linestyle='--')
        
        plt.show()
    
    else:
        
        print("Try again. Type : 2D or 3D.")
    
