
import torch.nn.functional as F
from torch import nn
import torch

# 64 128 128 10368
class PatchEncoder(nn.Module):

    def __init__(self):
        super(PatchEncoder, self).__init__()

        self.restored = False
        self.n_dim = 1296 * 4
        self.red_dim = 60

        self.encoder1 = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
        )
        self.encoder2 = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
        )
        self.encoder3 = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
        )
        self.encoder4 = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
        )
        self.encoder5 = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
        )

        self.encoder6 = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
        )

        self.encoder7 = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
        )

        self.encoder8 = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            
        )

        self.encoder9 = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
        )

        self.fc1 = nn.Linear(self.n_dim,self.red_dim)
        self.fc2 = nn.Linear(self.n_dim,self.red_dim)
        self.fc3 = nn.Linear(self.n_dim,self.red_dim)
        self.fc4 = nn.Linear(self.n_dim,self.red_dim)
        self.fc5 = nn.Linear(self.n_dim,self.red_dim)
        self.fc6 = nn.Linear(self.n_dim,self.red_dim)
        self.fc7 = nn.Linear(self.n_dim,self.red_dim)
        self.fc8 = nn.Linear(self.n_dim,self.red_dim)
        self.fc9 = nn.Linear(self.n_dim,self.red_dim)

    def forward(self, patch1, patch2, patch3, patch4, patch5, patch6, patch7, patch8, patch9):
        conv_out1 = self.fc1(self.encoder1(patch1).view(-1,self.n_dim))
        conv_out2 = self.fc2(self.encoder2(patch2).view(-1,self.n_dim))
        conv_out3 = self.fc3(self.encoder3(patch3).view(-1,self.n_dim))
        conv_out4 = self.fc4(self.encoder4(patch4).view(-1,self.n_dim))
        conv_out5 = self.fc5(self.encoder5(patch5).view(-1,self.n_dim))
        conv_out6 = self.fc6(self.encoder6(patch6).view(-1,self.n_dim))
        conv_out7 = self.fc7(self.encoder7(patch7).view(-1,self.n_dim))
        conv_out8 = self.fc8(self.encoder8(patch8).view(-1,self.n_dim))
        conv_out9 = self.fc9(self.encoder9(patch9).view(-1,self.n_dim))

        concat_out = torch.cat((conv_out1.view(-1, 1, self.red_dim), conv_out2.view(-1, 1, self.red_dim), conv_out3.view(-1, 1, self.red_dim), conv_out4.view(-1, 1, self.red_dim),
                                conv_out5.view(-1, 1, self.red_dim), conv_out6.view(-1, 1, self.red_dim), conv_out7.view(-1, 1, self.red_dim), conv_out8.view(-1, 1, self.red_dim),
                                   conv_out9.view(-1, 1, self.red_dim)), 1)

        return concat_out

class PatchAttention(nn.Module):

    def __init__(self):
        self.restored = False
        super(PatchAttention, self).__init__()
        self.n_dim = 60
        self.W_y = nn.Parameter(torch.randn(self.n_dim, self.n_dim).cuda())
        self.W_alpha = nn.Parameter(torch.randn(self.n_dim, 1).cuda())

    def forward(self, Y):
        M = torch.tanh(torch.bmm(Y, self.W_y.unsqueeze(0).expand(Y.size(0), *self.W_y.size())))
        alpha = torch.bmm(M, self.W_alpha.unsqueeze(0).expand(Y.size(0), *self.W_alpha.size())).squeeze(-1)  # batch x T
        alpha = F.sigmoid(alpha)

        r = torch.bmm(alpha.unsqueeze(1), Y).squeeze(1)

        return r, alpha


class PatchClassifier(nn.Module):

    def __init__(self):
        super(PatchClassifier, self).__init__()
        self.fc2 = nn.Linear(60,1)

    def forward(self, feat):
        out = self.fc2(F.relu(feat))
        return out
