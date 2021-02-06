'''
## Author
+ Name: Jahid Hasan
+ 𝐏𝐡𝐨𝐧𝐞:   (+880) 1772905097 (Whatsapp)
+ 𝘔𝘢𝘪𝘭:     jahidnoyon36@gmail.com
+ LinkedIn: http://linkedin.com/in/hellojahid
'''


import torch
from torch.autograd import Variable

# define Tensor
xdata = Variable(torch.Tensor([ [1.0], [2.0], [3.0]   ]))
ydata = Variable(torch.Tensor([ [2.0], [3.0], [4.0]   ]))


# Define Linear Regression(LR) Class
class LRM(torch.nn.Module):

    def __init__(self):
        super(LRM,self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    # computes output Tensors from input Tensors.
    def forward(self, x):
        ypred = self.linear(x)

        return ypred

# instance of LR class
model = LRM()

# loss function and optimizer
get_loss = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)


# traning the LR model
for epoch in range(500):
    optimizer.zero_grad() # the gradients are set to 0
    
    predy = model(xdata) # pred input data 
    loss = get_loss(predy, ydata) # calculate loss 
    loss.backward() # calculates the gradient 
    optimizer.step()  # updates the weights based on the gradients

    print('Epoch: {}, Loss: {}'.format(epoch, loss)) # print loss


# Test New value
newvar = Variable(torch.Tensor([4.0]))
prediction = model(newvar)
print(prediction.data[0])


