import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch

'''
输入当前的状态、当前agent的obs、其他agent执行的动作、当前agent的编号对应的one-hot向量、所有agent上一个timestep执行的动作
输出当前agent的所有可执行动作对应的联合Q值——一个n_actions维向量
'''


class FPCritic(nn.Module):
    def __init__(self, input_shape, args):
        super(FPCritic, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.critic_dim)
        self.fc2 = nn.Linear(args.critic_dim, args.critic_dim)
        self.fc3 = nn.Linear(args.critic_dim, self.args.n_actions)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

class FPAttention(nn.Module):
    def __init__(self, args):
        super(FPAttention, self).__init__()
        self.args = args

        self.attn = nn.Linear(args.encoder_dim+args.decoder_dim, args.decoder_dim, bias=False)
        self.v = nn.Linear(args.decoder_dim, 1, bias=False)

    def forward(self, s, enc_output):
        trasition_len = enc_output.shape[1]
        s = s.unsqueeze(1).repeat(1, trasition_len, 1)

        # energy = [batch_size, trasition_len, dec_hid_dim]
        energy = torch.tanh(self.attn(torch.cat((s, enc_output), dim = 2)))
        
        # attention = [batch_size, trasition_len]
        attention = F.softmax(self.v(energy).squeeze(2), dim=1)
        attention = attention.unsqueeze(1)
        
        # c = [batch_size, 1, args.encoder_dim]
        c = torch.bmm(attention, enc_output)
        return c

class FPActionEncoder(nn.Module):
    def __init__(self, input_shape, args):
        super(FPActionEncoder, self).__init__()
        self.args = args

        self.rnn = nn.GRU(input_shape, args.encoder_dim, batch_first=True)
        self.fc = nn.Linear(args.encoder_dim, args.decoder_dim)

    def forward(self, inputs):
        enc_output, enc_hidden = self.rnn(inputs)
        # s = torch.tanh(self.fc(torch.cat((enc_hidden[:,-1,:], enc_hidden[:,-2,:]), dim = 1)))
        s = torch.tanh(self.fc(enc_hidden[:,-1,:]))
        return enc_output, s