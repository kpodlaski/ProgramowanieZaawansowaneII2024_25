import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from sklearn.metrics._classification import confusion_matrix
from torchsummary import summary
import numpy as np


class ML_Wrapper():
    def __init__(self, network, optimizer, base_path, device):
        self.network=network
        self.network.to(device)
        self.optimizer = optimizer
        self.device = device
        self.base_path = base_path
        self.train_losses = []
        self.val_losses = []

    def train(self, epoch, data_loader, val_loader = None, loss_fun = F.nll_loss):
        self.network.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(data_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            self.optimizer.zero_grad()
            output = self.network(data)
            loss = loss_fun(output, target)
            train_loss+= loss.item()
            loss.backward()
            self.optimizer.step()
        train_loss = train_loss/len(data_loader.dataset)
        print('Train Epoch: {} \tLoss: {:.6f}'.format(
            epoch, train_loss))
        self.train_losses.append(train_loss)
        if (val_loader):
            self.test(val_loader, val_test=True)

    def test(self, data_loader, val_test=False, create_confusion=False, loss_fun = F.nll_loss):
        if self.optimizer is None:
            raise Exception("No optimizer is set")
        self.network.eval()
        correct = 0
        if create_confusion:
            inialized = None
        with torch.no_grad():
            test_loss = 0
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.network(data)
                loss = loss_fun(output, target)
                test_loss += loss.item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
                if create_confusion:
                    if inialized == None:
                        results = pred.cpu().detach().numpy().reshape(-1)
                        expected = target.cpu().detach().numpy().reshape(-1)
                        inialized = True
                    else:
                        results = np.concatenate((results,pred.cpu().detach().numpy().reshape(-1)), axis = None)
                        expected = np.concatenate((expected,target.cpu().detach().numpy().reshape(-1)), axis = None)
        if (val_test):
            test_loss = test_loss/len(data_loader.dataset)
            self.val_losses.append(test_loss)
            print ("Test loss:",test_loss)
        else:
            print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                test_loss, correct, len(data_loader.dataset),
                100. * correct / len(data_loader.dataset)))
        if create_confusion:
            cf_matrix = confusion_matrix(expected, results)
            return cf_matrix

    def save_model(self, fname):
        torch.save(self.network.state_dict(), self.base_path + '/out/{}_model.pth'.format(fname))
        #torch.save(optimizer.state_dict(), base_path+'/results/optimizer.pth')

    @staticmethod
    def load_model(base_path, file_name, model, optimizer=None, device = None ):
        file_path = base_path+"/out/"+file_name
        model.load_state_dict(torch.load(file_path))
        return ML_Wrapper(model, optimizer,base_path,device)


    def training_plot(self):
        fig = plt.figure()
        plt.plot([*range(len(self.train_losses))], self.train_losses, color='blue')
        if (len(self.val_losses))>0:
            plt.plot([*range(len(self.val_losses))], self.val_losses, color='red')
        plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
        plt.xlabel('epoch')
        plt.ylabel('negative log likelihood loss')
        fig.show()

    def summary(self, size):
        x = torch.Tensor(size).to(self.device)
        summary(self.network, size)