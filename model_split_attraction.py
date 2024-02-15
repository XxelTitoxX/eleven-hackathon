import torch
import torch.nn as nn
import torch.optim as optim
import pathlib
from torch.utils.data import sampler
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime
import math

csv_train = 'waiting_times_train.csv'
csv_val = 'waiting_times_X_test_final.csv'

mu_train = [ 413.742856,  0.268768404,  23.3896588,  106.796337, -22.9071778,  264.336251,  23.6055056]
std_train = [252.38810116,   1.5942279,   14.11456101, 174.30027737,  91.05014758, 261.33024836,  14.28219315]

mu_val = [ 406.148527,  0.206713058,  25.1412198,  96.3917315, -17.2943103,  240.808432]
std_val = [248.4956487,    1.37794598,  14.03282917, 158.16599843,  78.13698885, 255.38941424]

mu_train_wr = [232.81422358,   0.4352907,   22.79669014, 133.58024243, -17.37715032,
 304.91689047,  23.99216085]
std_train_wr = [ 39.17962398,   2.01530332,  11.61582147, 178.06880184,  79.96086195,
 265.40005149,  11.34118271]
mu_train_ps = [ 2.16919943e+02,  4.98119229e-02,  2.92636498e+01,  6.27801208e+01,
 -1.36036704e+01,  1.77501425e+02,  2.86686424e+01]
std_train_ps = [ 78.24230265,   0.67635247,  13.7878653,  152.52535946,  72.06548291,
 253.05613626,  14.19161723]
mu_train_fc = [ 7.05403311e+02,  2.42966752e-01,  2.03933089e+01,  1.07947743e+02,
 -3.38152347e+01,  2.78291975e+02,  2.01683141e+01]
std_train_fc = [134.46892111,   1.51718735,  15.35306489, 177.65871679, 108.50509725,
 249.96112156,  15.80497185]

mu_time = 850
std_time = 190

# Lecture du fichier CSV
data_train_wr = []
data_train_ps = []
data_train_fc = []
titre_train = []
with open(csv_train, mode='r') as fichier:
    lecteur_csv = csv.reader(fichier, delimiter=',')
    titre_train = next(lecteur_csv)
    
    for ligne in lecteur_csv:
        date_object = datetime.strptime(ligne[0], '%Y-%m-%d %H:%M:%S')

        # Extraire l'heure et les minutes de l'objet datetime
        heure = date_object.hour
        minutes = date_object.minute

        # Calculer le nombre total de minutes
        total_minutes = heure * 60 + minutes
        ligne[0]=(total_minutes-mu_time)/std_time
        if(ligne[1]=="Water Ride"):
            for i in range(2, 9):
                if ligne[i]=='':
                    ligne[i]=0.
                else:
                    ligne[i]=(float(ligne[i])-mu_train_wr[i-2])/std_train_wr[i-2]
            data_train_wr.append(ligne)
        elif(ligne[1]=="Pirate Ship"):
            for i in range(2, 9):
                if ligne[i]=='':
                    ligne[i]=0.
                else:
                    ligne[i]=(float(ligne[i])-mu_train_ps[i-2])/std_train_ps[i-2]
            data_train_ps.append(ligne)
        else:
            for i in range(2, 9):
                if ligne[i]=='':
                    ligne[i]=0.
                else:
                    ligne[i]=(float(ligne[i])-mu_train_fc[i-2])/std_train_fc[i-2]
            data_train_fc.append(ligne)

data_train_fc.pop(0)

taille_trois_quarts_wr = len(data_train_wr) * 9 // 10
taille_trois_quarts_ps = len(data_train_ps) * 9 // 10
taille_trois_quarts_fc = len(data_train_fc) * 9 // 10
data_accuracy_wr = data_train_wr[taille_trois_quarts_wr:]
data_accuracy_ps = data_train_ps[taille_trois_quarts_ps:]
data_accuracy_fc = data_train_fc[taille_trois_quarts_fc:]

data_val_wr = []
data_val_ps = []
data_val_fc = []
data_val = []
titre_val = []

with open(csv_val, mode='r') as fichier:
    lecteur_csv = csv.reader(fichier, delimiter=',')
    titre_val = next(lecteur_csv)
    
    for ligne in lecteur_csv:
        if(ligne[1]=="Water Ride"):
            for i in range(2, 8):
                if ligne[i]=='':
                    ligne[i]=0.
                else:
                    ligne[i]=(float(ligne[i])-mu_train_wr[i-2])/std_train_wr[i-2]
            data_val_wr.append(ligne)
        elif(ligne[1]=="Pirate Ship"):
            for i in range(2, 8):
                if ligne[i]=='':
                    ligne[i]=0.
                else:
                    ligne[i]=(float(ligne[i])-mu_train_ps[i-2])/std_train_ps[i-2]
            data_val_ps.append(ligne)
        else:
            for i in range(2, 8):
                if ligne[i]=='':
                    ligne[i]=0.
                else:
                    ligne[i]=(float(ligne[i])-mu_train_fc[i-2])/std_train_fc[i-2]
            data_val_fc.append(ligne)
        data_val.append(ligne)

data_val_fc.pop(0)


USE_GPU = True
dtype = torch.float32 # we will be using float throughout this tutorial
device = torch.device('cuda') if (USE_GPU and torch.cuda.is_available()) else torch.device('cpu')
print('using device:', device)

def run_val(loader, model_wr, model_ps, model_fc):
    model_wr.eval()
    model_ps.eval()
    model_fc.eval()
    scores = []
    with torch.no_grad():
        for vect in loader:
            date_object = datetime.strptime(vect[0], '%Y-%m-%d %H:%M:%S')

            # Extraire l'heure et les minutes de l'objet datetime
            heure = date_object.hour
            minutes = date_object.minute

            # Calculer le nombre total de minutes
            total_minutes = heure * 60 + minutes
            x = torch.tensor(vect[2:8]+[(total_minutes-mu_time)/std_time])
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            if(vect[1]=="Water Ride"):
                scores.append(float(model_wr(x).squeeze())*std_train_wr[6]+mu_train_wr[6])
            elif(vect[1]=="Pirate Ship"):
                scores.append(float(model_ps(x).squeeze())*std_train_ps[6]+mu_train_ps[6])
            else:
                scores.append(float(model_fc(x).squeeze())*std_train_fc[6]+mu_train_fc[6])

    return scores

def create_prompt(csv_prompt, data_val, titre_val, scores):
    with open(csv_prompt, mode='w', newline='') as fichier:
    # Créer un écrivain CSV
        ecrivain_csv = csv.writer(fichier)
        ecrivain_csv.writerow(titre_val[0:2]+["y_pred", "KEY"])
        for i in range(len(data_val)):
            ecrivain_csv.writerow(data_val[i][0:2]+[scores[i], "c57d53a31f68e864e929524b80c3dfe31190a5e431187fa12f"])

def check_accuracy(data_accuracy, model):
    model.eval()
    criterion = nn.MSELoss()
    accuracy=0.
    with torch.no_grad():
        for vect in data_accuracy:
            x = torch.tensor(vect[2:8]+[vect[0]])
            y = torch.tensor(vect[8])
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=dtype)
            scores = model(x).squeeze()
            loss = criterion(scores, y)
            accuracy+=loss.item()
    accuracy/=len(data_accuracy)
    return accuracy



def train_module(model, optimizer, dataset, epochs=1, loss_every=100):
    losses = {}
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    criterion = nn.MSELoss()
    current_loss=0.
    for e in range(epochs):
        for t, vect in enumerate(dataset):
            x = torch.tensor(vect[2:8]+[vect[0]])
            y = torch.tensor(vect[8])
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=dtype)

            scores = model(x).squeeze()
            loss = criterion(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()
            current_loss+=loss.item()
            if t % loss_every == 0:
                losses[e * len(dataset) + t] = current_loss/loss_every
                current_loss = 0.

    return losses


learning_rate_list = [1e-4]
input_size = 7
first_hid_layer_list = [100]
second_hid_layer_list = [100]
for first_hid_layer in first_hid_layer_list:
    for second_hid_layer in second_hid_layer_list:
        for learning_rate in learning_rate_list:

            model_wr = nn.Sequential(
                nn.Linear(input_size, first_hid_layer),
                nn.ReLU(),
                nn.Linear(first_hid_layer, second_hid_layer),
                nn.ReLU(),
                nn.Linear(second_hid_layer, 1),
            )

            model_fc = nn.Sequential(
                nn.Linear(input_size, first_hid_layer),
                nn.ReLU(),
                nn.Linear(first_hid_layer, second_hid_layer),
                nn.ReLU(),
                nn.Linear(second_hid_layer, 1),
            )

            model_ps = nn.Sequential(
                nn.Linear(input_size, first_hid_layer),
                nn.ReLU(),
                nn.Linear(first_hid_layer, second_hid_layer),
                nn.ReLU(),
                nn.Linear(second_hid_layer, 1),
            )

            with torch.no_grad():
                nn.init.kaiming_normal_(model_wr[0].weight, mode='fan_in', nonlinearity='relu')
                nn.init.kaiming_normal_(model_wr[2].weight, mode='fan_in', nonlinearity='relu')
                nn.init.kaiming_normal_(model_wr[4].weight, mode='fan_in', nonlinearity='relu')
                nn.init.kaiming_normal_(model_ps[0].weight, mode='fan_in', nonlinearity='relu')
                nn.init.kaiming_normal_(model_ps[2].weight, mode='fan_in', nonlinearity='relu')
                nn.init.kaiming_normal_(model_ps[4].weight, mode='fan_in', nonlinearity='relu')
                nn.init.kaiming_normal_(model_fc[0].weight, mode='fan_in', nonlinearity='relu')
                nn.init.kaiming_normal_(model_fc[2].weight, mode='fan_in', nonlinearity='relu')
                nn.init.kaiming_normal_(model_fc[4].weight, mode='fan_in', nonlinearity='relu')

            # you can use Nesterov momentum in optim.SGD
            optimizer_wr = optim.Adam(model_wr.parameters(), lr=learning_rate)
            optimizer_ps = optim.Adam(model_ps.parameters(), lr=learning_rate)
            optimizer_fc = optim.Adam(model_fc.parameters(), lr=learning_rate)

            loss_train_wr = train_module(model_wr, optimizer_wr, data_train_wr, epochs=2)
            loss_train_ps = train_module(model_ps, optimizer_ps, data_train_ps, epochs=2)
            loss_train_fc = train_module(model_fc, optimizer_fc, data_train_fc, epochs=2)

            approx_accuracy_wr = check_accuracy(data_accuracy_wr, model_wr)
            approx_accuracy_ps = check_accuracy(data_accuracy_ps, model_ps)
            approx_accuracy_fc = check_accuracy(data_accuracy_fc, model_fc)

            approx_accuracy = (approx_accuracy_wr+approx_accuracy_ps+approx_accuracy_fc)/3

            # fig, ax = plt.subplots(figsize=(11, 6))
            # x = sorted(loss_train.keys())
            # y = [loss_train[e] for e in x]
            # ax.plot(x, y, label='loss_train')
            # ax.legend()
            # plt.show()

            csv_prompt = 'prompt_'+str(first_hid_layer)+'_'+str(second_hid_layer)+'_'+str(learning_rate)+'_'+str(approx_accuracy)+'.csv'
            create_prompt(csv_prompt, data_val, titre_val, run_val(data_val, model_wr, model_ps, model_fc))
