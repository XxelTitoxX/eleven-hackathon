import csv
import matplotlib.pyplot as plt
fichier_csv = 'waiting_times_train.csv'

# Lecture du fichier CSV
data_train = []
titre = []
with open(fichier_csv, mode='r') as fichier:
    lecteur_csv = csv.reader(fichier, delimiter=',')
    titre = next(lecteur_csv)
    
    for ligne in lecteur_csv:
        for i in range(2, 9):
            if ligne[i]=='':
                ligne[i]=None
            else:
                ligne[i]=float(ligne[i])
        data_train.append(ligne)

data_train.pop(0)
ind_premiere_colonne = 5
ind_deuxieme_colonne = 8

premiere_colonne = [ligne[ind_premiere_colonne] for ligne in data_train if ((ligne[ind_premiere_colonne] is not None) and (ligne[ind_deuxieme_colonne] is not None))]
deuxieme_colonne = [ligne[ind_deuxieme_colonne] for ligne in data_train if ((ligne[ind_premiere_colonne] is not None) and (ligne[ind_deuxieme_colonne] is not None))]

# Tracer la courbe
plt.scatter(premiere_colonne, deuxieme_colonne)
plt.title('Courbe de y en fonction de x')
plt.xlabel(titre[ind_premiere_colonne])
plt.ylabel(titre[ind_deuxieme_colonne])
plt.show()



