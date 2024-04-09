import torch
from torch import nn, Tensor

from Config import Config


class EncodingNetwork(nn.Module):
    learning_rate = 1e-4

    def __init__(
        self,
        n_tasks: int,
        m_machines: int,
        fc_out_features: int = 64,
        hidden_size: int = 64,
        out_channels: int = 16,
    ):
        super().__init__()
        self.m_machines = m_machines
        self.n_tasks = n_tasks
        self.hidden_size = hidden_size
        self.fc_out_features = fc_out_features
        self.out_channels = out_channels
        self.state_embedding_layer = nn.Linear(in_features=1, out_features=1)
        self.number_of_machines_embedding_layer = nn.Linear(
            in_features=1, out_features=1
        )
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size + 1, out_features=fc_out_features)
        self.conv = nn.Conv1d(
            in_channels=self.m_machines,
            out_channels=out_channels,
            kernel_size=3,
            padding="same",
        )
        self.relu = nn.LeakyReLU()

    def _part_one(self, x):
        return torch.concat(
            tuple(
                self.rnn(self.state_embedding_layer(x[:, :, i].unsqueeze(-1)))[-1]
                for i in range(self.m_machines)
            )
        ).transpose(0, 1)

    def _part_two(self, p):
        # h = torch.empty((self.m_machines, self.fc_out_features)).to(Config.device)
        number_of_machines_embedded = torch.full(
            (p.shape[0], p.shape[1], 1),
            self.number_of_machines_embedding_layer(
                Tensor([self.m_machines]).to(Config.device)
            ).item(),
        ).to(Config.device)
        p = torch.concat((p, number_of_machines_embedded), dim=2)
        p = self.fc(p)
        return self.relu(p)
        # for i in range(self.m_machines):
        #     h[i] = self.fc(torch.concat((p[i].unsqueeze(0), number_of_machines_embedded), dim=1))
        #     h[i] = self.relu(h[i])
        # return h

    def _part_three(self, p):
        p = self.conv(p)
        return self.relu(p)

    def forward(self, x):
        x = x.to(Config.device)
        p = self._part_one(x)
        p = self._part_two(p)
        return self._part_three(p)


encoder = EncodingNetwork(Config.n_tasks, Config.m_machines).to(Config.device)
