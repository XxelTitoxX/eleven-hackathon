import csv
from datetime import datetime
import numpy as np

csv_train = 'waiting_times_train.csv'

csv_val = 'waiting_times_X_test_val.csv'

# Lecture du fichier CSV d'entrainement
data_train_wr = []
data_train_ps = []
data_train_fc = []
titre_train = []
with open(csv_train, mode='r') as fichier:
    lecteur_csv = csv.reader(fichier, delimiter=',')
    titre_train = next(lecteur_csv)
    
    for ligne in lecteur_csv:
        for i in range(2, 9):
            if ligne[i]=='':
                ligne[i]=0.
            else:
                ligne[i]=float(ligne[i])
        date_object = datetime.strptime(ligne[0], '%Y-%m-%d %H:%M:%S')

        # Extraire l'heure et les minutes de l'objet datetime
        heure = date_object.hour
        minutes = date_object.minute

        # Calculer le nombre total de minutes
        total_minutes = heure * 60 + minutes
        ligne[0]=total_minutes
        if(ligne[1]=="Water Ride"):
            data_train_wr.append(ligne)
        elif(ligne[1]=="Pirate Ship"):
            data_train_ps.append(ligne)
        else:
            data_train_fc.append(ligne)

data_train_fc.pop(0)

def get_mu_std(dataset):

    mu, std = np.array([0., 0., 0., 0., 0., 0., 0., 0.]), np.array([0., 0., 0., 0., 0., 0., 0., 0.]) # Attention le nombre de valeurs change selon val ou train
    dataset_len = len(dataset)
    for k in range(dataset_len):
      vect = dataset[k][2:9]+[dataset[k][0]] # Attention l'indice de fin change selon val ou train
      vect = np.asarray(vect)
      mu += vect
      std += np.square(vect)
    mu = mu/(dataset_len)
    std = std/(dataset_len)-np.square(mu)

    return mu, np.sqrt(std)

mu_wr, std_wr = get_mu_std(data_train_wr)
mu_ps, std_ps = get_mu_std(data_train_ps)
mu_fc, std_fc = get_mu_std(data_train_fc)

print(mu_wr)
print(std_wr)
print(mu_ps)
print(std_ps)
print(mu_fc)
print(std_fc)