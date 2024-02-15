import csv
fichier_csv = 'waiting_times_X_test_val.csv'
fichier_sortie = 'prompt.csv'

# Lecture du fichier CSV
data_train = []
titre = []
with open(fichier_csv, mode='r') as fichier:
    lecteur_csv = csv.reader(fichier, delimiter=',')
    titre = next(lecteur_csv)[0:2]+["y_pred", "KEY"]
    
    for ligne in lecteur_csv:
        ligne = ligne[0:2]+[0.0, "Validation"]
        data_train.append(ligne)
data_train[0] = titre

with open(fichier_sortie, mode='w', newline='') as fichier:
    # Créer un écrivain CSV
    ecrivain_csv = csv.writer(fichier)

    # Écrire les données dans le fichier CSV
    for ligne in data_train:
        ecrivain_csv.writerow(ligne)