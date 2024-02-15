import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pathlib
from torch.utils.data import sampler
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import csv

csv_train = 'waiting_times_train.csv'
csv_val = 'waiting_times_X_test_val.csv'

mu_train = [ 413.742856,  0.268768404,  23.3896588,  106.796337, -22.9071778,  264.336251,  23.6055056]
std_train = [252.38810116,   1.5942279,   14.11456101, 174.30027737,  91.05014758, 261.33024836,  14.28219315]

mu_val = [ 406.148527,  0.206713058,  25.1412198,  96.3917315, -17.2943103,  240.808432]
std_val = [248.4956487,    1.37794598,  14.03282917, 158.16599843,  78.13698885, 255.38941424]

# Lecture du fichier CSV
data_train = []
titre_train = []
with open(csv_train, mode='r') as fichier:
    lecteur_csv = csv.reader(fichier, delimiter=',')
    titre_train = next(lecteur_csv)
    
    for ligne in lecteur_csv:
        for i in range(2, 9):
            if ligne[i]=='':
                ligne[i]=0.
            else:
                ligne[i]=(float(ligne[i])-mu_train[i-2])/std_train[i-2]
        data_train.append(ligne)

data_train.pop(0)

data_val = []
titre_val = []

with open(csv_val, mode='r') as fichier:
    lecteur_csv = csv.reader(fichier, delimiter=',')
    titre_val = next(lecteur_csv)
    
    for ligne in lecteur_csv:
        for i in range(2, 8):
            if ligne[i]=='':
                ligne[i]=0.
            else:
                ligne[i]=(float(ligne[i])-mu_val[i-2])/std_val[i-2]
        data_val.append(ligne)

data_val.pop(0)


USE_GPU = True
dtype = torch.float32 # we will be using float throughout this tutorial
device = torch.device('cuda') if (USE_GPU and torch.cuda.is_available()) else torch.device('cpu')
print('using device:', device)

def run_val(loader, model):
    model.eval()
    scores = []
    with torch.no_grad():
        for vect in loader:
            x = torch.tensor(vect[2:8])
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            scores.append(float(model(x).squeeze())*std_train[6]+mu_train[6])

    return scores

def create_prompt(csv_prompt, data_val, titre_val, scores):
    with open(csv_prompt, mode='w', newline='') as fichier:
    # Créer un écrivain CSV
        ecrivain_csv = csv.writer(fichier)
        ecrivain_csv.writerow(titre_val[0:2]+["y_pred", "KEY"])
        for i in range(len(data_val)):
            ecrivain_csv.writerow(data_val[i][0:2]+[scores[i], "Validation"])


def train_module(model, optimizer, epochs=1, loss_every=100):
    losses = {}
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    criterion = nn.MSELoss()
    current_loss=0.
    for e in range(epochs):
        for t, vect in enumerate(data_train):
            x = torch.tensor(vect[2:8])
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
                losses[e * len(data_train) + t] = current_loss/loss_every
                current_loss = 0.

    return losses


learning_rate_list = [1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5]
input_size = 6
first_hid_layer_list = [50, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280]
second_hid_layer_list = [50, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280]
for first_hid_layer in first_hid_layer_list:
    for second_hid_layer in second_hid_layer_list:
        for learning_rate in learning_rate_list:
            csv_prompt = 'prompt_'+str(first_hid_layer)+'_'+str(second_hid_layer)+'_'+str(learning_rate)+'.csv'

            model = nn.Sequential(
                nn.Linear(6, first_hid_layer),
                nn.ReLU(),
                nn.Linear(first_hid_layer, second_hid_layer),
                nn.ReLU(),
                nn.Linear(second_hid_layer, 1),
            )

            with torch.no_grad():
                nn.init.kaiming_normal_(model[0].weight, mode='fan_in', nonlinearity='relu')
                nn.init.kaiming_normal_(model[2].weight, mode='fan_in', nonlinearity='relu')
                nn.init.kaiming_normal_(model[4].weight, mode='fan_in', nonlinearity='relu')

            # you can use Nesterov momentum in optim.SGD
            optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                            momentum=0.9, nesterov=True)

            loss_train = train_module(model, optimizer)

            # fig, ax = plt.subplots(figsize=(11, 6))
            # x = sorted(loss_train.keys())
            # y = [loss_train[e] for e in x]
            # ax.plot(x, y, label='loss_train')
            # ax.legend()
            # plt.show()

            create_prompt(csv_prompt, data_val, titre_val, run_val(data_val, model))
