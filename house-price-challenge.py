import pandas as pd
import torch
from torch import nn
from torch.nn import MSELoss
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import numpy as np


class Price_Data(Dataset):
    def __init__(self):
        self.df = pd.read_csv("./california-house-price/housing_train.csv")
        self.label = torch.tensor(self.df.iloc[:, 8].values.reshape([-1, 1]), dtype=torch.float32)

        numeric = self.df.dtypes[self.df.dtypes != 'object'].index
        self.df[numeric] = self.df[numeric].apply(lambda x: (x - x.mean()) / x.std())
        self.df[numeric] = self.df[numeric].fillna(0)

        self.df_1 = pd.get_dummies(self.df, dummy_na=True)
        self.data1 = torch.tensor(self.df_1.iloc[:, :8].values, dtype=torch.float32)
        self.data2 = torch.tensor(self.df_1.iloc[:, 9:].values, dtype=torch.float32)
        self.data = torch.concat([self.data1, self.data2], dim=1)

        # print(self.data[1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.label[item]


class Test_Data(Dataset):
    def __init__(self):
        self.df = pd.read_csv("./california-house-price/housing_test.csv")

        self.label = torch.tensor(self.df.iloc[:, 0].values.reshape([-1, 1]), dtype=torch.int32)

        numeric = self.df.dtypes[self.df.dtypes != 'object'].index
        self.df[numeric] = self.df[numeric].apply(lambda x: (x - x.mean()) / x.std())
        self.df[numeric] = self.df[numeric].fillna(0)

        self.df_1 = pd.get_dummies(self.df, dummy_na=True)

        print(self.df_1.iloc[:, 9].values)

        print(self.df_1.iloc[:, 1:].dtypes)

        self.data = torch.tensor(self.df_1.iloc[:, 1:].values, dtype=torch.float32)

        print(self.data[1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.label[item]


class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=14, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=64, out_features=1)
        )

    def forward(self, x):
        return self.model(x)


def log_rmse(y_pred, y_true):
    clipped_preds = torch.clamp(y_pred, min=1, max=float('inf'))
    return torch.sqrt(MSELoss()(torch.log(clipped_preds), torch.log(y_true)))


epoch = 100
lr = 0.01
wd = 0.0001

data = Price_Data()


def train():
    data = Price_Data()
    train_loader = DataLoader(data, batch_size=16, shuffle=True)

    # data_pre = Test_Data()
    # pred_loader = DataLoader(data_pre, batch_size=1, shuffle=False)

    model = Mymodel().cuda()
    loss_func = log_rmse
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    for i in range(epoch):
        eopch_loss = []
        for index, item in enumerate(train_loader):
            input, label = item
            input = input.cuda()
            label = label.cuda()
            optim.zero_grad()
            output = model(input)
            l = loss_func(output, label)
            l.backward()
            optim.step()
            eopch_loss.append(l)
        print(f"epoch{i}, loss{sum(eopch_loss) / len(eopch_loss)}")

    torch.save(model, "./california-house-price/1.pth")


def K_Fold_Vali():
    kf = KFold(n_splits=3, shuffle=True)
    loss_val_sum = []
    loss_train_sum = []
    for train_index, val_index in kf.split(data):
        train_data = torch.utils.data.dataset.Subset(data, train_index)
        val_data = torch.utils.data.dataset.Subset(data, val_index)

        train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
        val_loader = DataLoader(dataset=val_data, batch_size=16, shuffle=False)
        model = Mymodel().cuda()
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        loss_func = log_rmse

        for i in range(epoch):
            eopch_loss = []
            for index, item in enumerate(train_loader):
                input, label = item
                input = input.cuda()
                label = label.cuda()
                optim.zero_grad()
                output = model(input)
                l = loss_func(output, label)
                l.backward()
                optim.step()
                loss_train_sum.append(l)
                eopch_loss.append(l)
            print(f"epoch{i}, loss{sum(eopch_loss) / len(eopch_loss)}")

        val_loss = []
        for index, item in enumerate(val_loader):
            input, label = item
            input = input.cuda()
            label = label.cuda()
            output = model(input)
            loss = loss_func(output, label)
            loss_val_sum.append(loss)
            val_loss.append(loss)
        print(f" val, loss{sum(val_loss) / len(val_loss)}")

    print(sum(loss_train_sum) / len(loss_train_sum))
    print(sum(loss_val_sum) / len(loss_val_sum))


def pred():
    model = torch.load("./california-house-price/1.pth").cpu()
    model.eval()
    data_pre = Test_Data()
    pred_loader = DataLoader(data_pre, batch_size=1, shuffle=False)

    i = 0

    for index, item in enumerate(pred_loader):
        input, index = item
        output = model(input)
        # index = torch.tensor([int(i) for i in index]).reshape([-1, 1])
        # x = torch.concat([index, output], dim=0).T.detach().numpy()
        #
        # print(x)
        df = {
            'Id': i,
            'median_house_value': output[0].detach().numpy()
        }
        df = pd.DataFrame(df)
        df.to_csv("./house-price-challenge/submission.csv", mode='a', index=False, header=False)

        i += 1


if __name__ == "__main__":
    # K_Fold_Vali()
    # train()
    pred()
