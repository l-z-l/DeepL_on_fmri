from datetime import datetime

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np

from utils.data import *
from models.LSTM import LSTM
from utils.config import args
from utils.helper import train_loader, train_loader_graph, plot_train_result, num_correct
from utils.datasets import DatasetFactory
from torch.utils.data import DataLoader
from sklearn.linear_model import Lasso
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
# plot
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import plotly.express as px
import plotly.graph_objects as go
import plotly.offline
import seaborn as sns

##########################################################
# %% Meta
###############train_test_split###########################
SAVE = False
MODEL_NANE = f'LSTMCV_{datetime.now().strftime("%Y-%m-%d-%H:%M")}'
datadir = './data'
outdir = './outputs'
dataset_name = '273_MSDL'
device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

if SAVE:
    save_path = os.path.join(outdir, f'{MODEL_NANE}_{dataset_name}/') if SAVE else ''
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
else:
    save_path = ''
##########################################################
# %% Load Data
##########################################################
device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
## augmented
train_dataset, test_dataset = DatasetFactory.create_train_test_roi_signal_datasets_from_path(
    train_path="data/augmented/2088_train_273_MSDL_org_100_window_5_stride",
    test_path="data/augmented/369_test_273_MSDL_org_100_window_5_stride")
## not augmented
# train_dataset, test_dataset = DatasetFactory.create_train_test_roi_signal_datasets_from_single_path(
#     path="data/273_Havard_Oxford"
# )
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

##########################################################
# %% initialise mode and
##########################################################
model = LSTM(input_size=train_dataset[0][0].shape[1], hidden_dim=16, seq_len=train_dataset[0][0].shape[0], num_layers=1,
             output_size=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=args.weight_decay)
criterion = nn.CrossEntropyLoss().to(device)

results = []
# for _ in range(500):
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)


train_loss_list, test_loss_list, training_acc, testing_acc = [], [], [], []
for epoch in range(700):
    model.train()
    train_loss, correct, total = 0, 0, 0
    val_loss, val_correct, val_total = 0, 0, 0

    ### train ###
    for batch_id, (train_x, train_y) in enumerate(train_loader):
        # Preparing Data
        train_x, train_y = train_x.to(device), train_y.to(device)
        # Feedforward
        optimizer.zero_grad()
        predict = model(train_x)

        # Compute the loss
        loss = criterion(predict.squeeze(), train_y.long())
        loss.backward()
        optimizer.step()

        correct += num_correct(predict, train_y)
        total += len(train_y)
        train_loss += loss.item()

    train_loss_list.append(train_loss / total)
    training_acc.append(int(correct) / total * 100)

    ### test ###
    model.eval()
    with torch.no_grad():
        for val_batch_id, (val_x, val_y) in enumerate(test_loader):
            val_x, val_y = val_x.to(device), val_y.to(device)

            val_predict = model(val_x)
            val_correct += num_correct(val_predict, val_y)

            val_total += len(val_y)
            val_loss += criterion(val_predict.squeeze(), val_y.long()).item()

    test_loss_list.append(val_loss / val_total)
    testing_acc.append(int(val_correct) / val_total * 100)

    if epoch % 50 == 0:
        print(
            f"====>Training: Epoch: {epoch}, Train loss: {train_loss_list[-1]:.3f}, Accuracy: {training_acc[-1]:.3f}")
        print(f"Test loss: {test_loss_list[-1]:.3f}, Accuracy: {testing_acc[-1]:.3f}")

results.append([train_loss_list[-1], training_acc[-1], test_loss_list[-1], testing_acc[-1]])

history = {
    "train_loss": train_loss_list,
    "train_acc": training_acc,
    "test_loss": test_loss_list,
    "test_acc": testing_acc,
}
history = pd.DataFrame(history)

results = pd.DataFrame(results, columns=['train_loss', 'train_acc', 'test_loss', 'test_acc'])

### save
if SAVE:
    # SAVE TRAINED MODEL and history
    history.to_csv(save_path + 'epochs.csv')
    results.to_csv(save_path + 'results.csv')
    # save model
    torch.save(model.state_dict(), save_path + 'LSTM.pth')

#########################################################
# %% Plot result
#########################################################
plot_train_result(history, save_path=save_path)

"""
#########################################################
# %% Plot result
#########################################################
model = LSTM(input_size=train_dataset[0][0].shape[1], hidden_dim=16, seq_len=train_dataset[0][0].shape[0], num_layers=1,
             output_size=2).to(device)
model.load_state_dict(torch.load("./outputs/LSTMCV_2020-11-09-20:47_273_MSDL/LSTM.pth"))
### test ###
model.eval()
embedded = []
label = []
with torch.no_grad():
    for _, (data, y) in enumerate(train_loader):  # Iterate in batches over the training dataset.
        label.append(y)
        data = data.to(device)
        val_predict = model.interpret(data)
        embedded.append(val_predict.detach().cpu())
        # embedded.append(val_predict.detach().cpu())

embedded_points = np.concatenate(embedded, axis=0)

tsne = TSNE(n_components=3, verbose=0, perplexity=40, n_iter=300)
tsne_pca_results = tsne.fit_transform(embedded_points)
### PCA
# pca = PCA(n_components=2)
# pca_result = pca.fit_transform(embedded_points)

df = pd.DataFrame(tsne_pca_results, columns=['tsne_1', 'tsne_2', 'tsne_3'])
df['y'] = pd.Series(np.concatenate(label, axis=0))



# traces = []
# for i in np.unique(df['y']):  # range(3,5): # TODO: change it back to original form
#     df_filter = df[df['y'] == i]
#     scatter = go.Scatter3d(x=df_filter['tsne_1'].values,
#                            y=df_filter['tsne_2'].values,
#                            z=df_filter['tsne_3'].values,
#                            mode='markers',
#                            marker=dict(size=3, opacity=1),
#                            name=f'Cluster {i}')
#     traces.append(scatter)
# layout = dict(title='Latent Representation, colored by GT clustered')
# fig_3d_1 = go.Figure(data=traces, layout=layout)
# fig_3d_1.update_layout(margin=dict(l=0, r=0, b=0, t=0), showlegend=True, legend=dict(y=-.1))
# plotly.offline.plot(fig_3d_1, auto_open=True)
"""