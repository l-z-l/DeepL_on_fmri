import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
import os
from utils.data import *
from models.MLP import Linear
from utils.config import args
from utils.helper import plot_train_result, num_correct, plot_evaluation_matrix, cross_validation_train_vec_loader
from datetime import datetime
from utils.datasets import DatasetFactory
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.linear_model import Lasso
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
# plot
import matplotlib.pyplot as plt
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from sklearn.model_selection import KFold

##########################################################
# %% Meta
###############train_test_split###########################
SAVE = False
MODEL_NANE = f'MLP_{datetime.now().strftime("%Y-%m-%d-%H:%M")}'
datadir = './data/augmented/'
outdir = './outputs'
dataset_name = '271_100_5_sliced_AAL'
device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

if SAVE:
    save_path = os.path.join(outdir, f'{MODEL_NANE}_{dataset_name}/') if SAVE else ''
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
else:
    save_path = ''
##########################################################
# %% Load Data
###############train_test_split###########################################
# train_dataset, test_dataset = DatasetFactory.create_train_test_connectivity_datasets_from_path(
#     train_path="./data/273_ICA_200_n50_train",
#     test_path="./data/273_ICA_200_n50_test"
# )
train_dataset, test_dataset = DatasetFactory.create_train_test_connectivity_datasets_from_single_path(
    path="data/271_AAL"
)

train_idx, valid_idx = np.random.permutation(range(len(train_dataset))), np.random.permutation(range(len(test_dataset)))
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, sampler=train_sampler)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, sampler=valid_sampler)

##########################################################
# %% initialise model and loss func
##########################################################
print("--------> Using ", device)
# def train_glm(config, checkpoint_dir=None):
model = Linear(len(train_dataset[0][0]), 1)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=args.weight_decay)

criterion = torch.nn.BCEWithLogitsLoss().to(device)

train_loss_list, test_loss_list, training_acc, testing_acc = [], [], [], []
for epoch in range(300):
    model.train()
    train_loss, correct, total = 0, 0, 0
    val_loss, val_correct, val_total = 0, 0, 0

    for batch_id, (train_x, train_y) in enumerate(train_loader):
        # Preparing Data
        train_x, train_y = train_x.to(device), train_y.to(device)

        optimizer.zero_grad()
        predict = model(train_x)

        correct += num_correct(predict, train_y)
        total += len(train_y)

        # Compute the loss
        loss = criterion(predict.squeeze(), train_y)  # , requires_grad=True)
        # Calculate gradients.
        loss.backward()
        # Minimise the loss according to the gradient.
        optimizer.step()

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
            val_loss += criterion(val_predict.squeeze(), val_y).item()

    test_loss_list.append(val_loss / val_total)
    testing_acc.append(int(val_correct) / val_total * 100)


    # tune.report(train_loss=train_loss_list[-1], train_accuracy=training_acc[-1], test_loss=test_loss_list[-1],
    #             test_accuracy=testing_acc[-1])
    if epoch % 50 == 0:
        print(f"====>Training: Epoch: {epoch}, Train loss: {train_loss_list[-1]:.3f}, Accuracy: {training_acc[-1]:.3f}")
        print(f"Test loss: {test_loss_list[-1]:.3f}, Accuracy: {testing_acc[-1]:.3f}")


# search_space = {
#     "lr": tune.grid_search([1e-4, 5e-4, 1e-3, 5e-3]), # tune.loguniform(1e-6, 1e-1),
#     "weight_decay": tune.grid_search([1e-4, 5e-4, 1e-3, 5e-3]), #tune.loguniform(1e-6, 1e-1),
#     # 'activation': tune.choice(["relu", "tanh"])
# }

# hyperopt_search = HyperOptSearch(search_space, metric="mean_accuracy", mode="max")
# bayesopt = BayesOptSearch(utility_kwargs={
#         "kind": "ucb",
#         "kappa": 2.5,
#         "xi": 0.0
#     })
# reporter = CLIReporter(
#     # parameter_columns=["l1", "l2", "lr", "batch_size"],
#     metric_columns=["train_loss_list", "train_accuracy", "test_loss", "test_accuracy"])
#
# analysis = tune.run(train_glm, resources_per_trial={"cpu": 12, "gpu": 1},
#                     num_samples=1,
#                     scheduler=ASHAScheduler(max_t=1000),
#                     # scheduler=ASHAScheduler(metric="test_accuracy", mode="max", max_t=800),
#                     metric="test_accuracy",
#                     mode="max",
#                     config=search_space,
#                     progress_reporter=reporter,)



history = {
    "train_loss": train_loss_list,
    "train_acc": training_acc,
    "test_loss": test_loss_list,
    "test_acc": testing_acc,
}
history = pd.DataFrame(history)
### save
if SAVE:
    # SAVE TRAINED MODEL and history
    history.to_csv(save_path + 'epochs.csv')
    # save model
    torch.save(model, save_path + 'MLP.pth')

#########################################################
# %% Plot result
#########################################################
plot_train_result(history, save_path=save_path)


#########################################################
# %% Evaluate result
#########################################################
'''
### test ###
model.eval()
label_truth = []
label_pred = []
label_pred_raw = []
with torch.no_grad():
    for val_batch_id, (val_x, val_y) in enumerate(test_loader):
        label_truth.append(val_y.numpy().tolist())

        val_x, val_y = val_x.to(device), val_y.to(device)

        val_predict = model(val_x)
        label_pred_raw.append(val_predict.cpu().numpy().tolist())
        pred = val_predict.max(dim=-1)[-1] if val_predict.shape[1] > 1 else val_predict > 0.5

        label_pred.append(pred.cpu().numpy().tolist())

plot_evaluation_matrix(label_truth, label_pred, label_pred_raw, save_path)
'''