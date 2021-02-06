'''
## Author
+ Name: Jahid Hasan
+ 𝐏𝐡𝐨𝐧𝐞:   (+880) 1772905097 (Whatsapp)
+ 𝘔𝘢𝘪𝘭:     jahidnoyon36@gmail.com
+ LinkedIn: http://linkedin.com/in/hellojahid

'''

import torch

x = torch.tensor(2.0, requires_grad=True)

y  = 2*x*2

# Differentiation
x.grad

print(y)
