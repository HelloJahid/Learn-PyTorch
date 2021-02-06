'''
## Author
+ Name: Jahid Hasan
+ 𝐏𝐡𝐨𝐧𝐞:   (+880) 1772905097 (Whatsapp)
+ 𝘔𝘢𝘪𝘭:     jahidnoyon36@gmail.com
+ LinkedIn: http://linkedin.com/in/hellojahid

'''


# import module
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


# create datasets
n_point = 100
centers = [[-0.5, 0.5], [0.5, -0.5]]
x, y = make_blobs(n_samples=n_point, random_state=101, centers=centers, cluster_std=0.4)

# convert data into Tensor
xdata = torch.Tensor(x)
ydata=torch.Tensor(y.reshape(100,1))


# plotting data
def scatter_plot():
    plt.scatter(x[y==0,0], x[y==0, 1])
    plt.scatter(x[y==1,0], x[y==1, 1])
    # plt.show()


#scatter_plot()


class Perceptron(nn.Module):

    def __init__(self, input_size, output_size):
        super(Perceptron, self).__init__()  # # nn.Module class itself initialized
        # nn layer
        self.layer1 = nn.Linear(input_size, output_size)

    # building Perceptron network
    def forward(self, x):
        pred = torch.sigmoid(self.layer1(x))

        return pred

    # predict function
    def predict(self, x):
        pred = torch.sigmoid(self.layer1(x))

        if pred >= 0.5:
            return 1
        else:
            return 0



# randomseed
torch.manual_seed(2)
model = Perceptron(2, 1)

# loss and  OPTIMIZER
model_loss = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# model parameters
def get_parameters(model):
    [m, b] = model.parameters()
    m1, m2 = m.view(2)
    b1 = b[0]

    return(m1.item(), m2.item(), b1.item())




# train the model
loss_list = []
EPOCHS = 100

for e in range(EPOCHS):
    # set gradient to zero
    optimizer.zero_grad()

    # xdata = xdata.squeeze(1)
    ypred = model.forward(xdata) # predict data from input
    loss = model_loss(ypred, ydata) # calculate loss
    loss.backward()  # calculate gradient
    optimizer.step() # update wieght based on gradient

    loss_list.append(loss)

    #print("EPOCHS: {}, Loss: {}".format(e+1, loss.item()))

    plt.title("Training")
    m1, m2, b1 = get_parameters(model)
    x1 = torch.tensor([-2.0,2.0],requires_grad=False)
    y1 = ((m1 * x1) +b1) / -m2
    plt.plot(x1, y1, 'r')
    scatter_plot()
    plt.pause(0.05)
    plt.clf()


plt.close()




def plot_fit(model):
    plt.title("After traning - fit line")
    m1, m2, b1 = get_parameters(model)
    x1 = torch.tensor([-2.0,2.0],requires_grad=False)
    y1 = ((m1 * x1) +b1) / -m2
    plt.plot(x1, y1, 'r')
    scatter_plot()
    plt.show()

plot_fit(model)
