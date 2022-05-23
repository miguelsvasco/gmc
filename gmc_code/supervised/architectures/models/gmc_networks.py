import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from gmc_code.supervised.architectures.baselines.transformer_networks import TransformerEncoder


def get_affect_network(self_type='l', layers=1):
    if self_type in ['l', 'al', 'vl']:
        embed_dim, attn_dropout = 30, 0.1
    elif self_type in ['a', 'la', 'va']:
        embed_dim, attn_dropout = 30, 0.0
    elif self_type in ['v', 'lv', 'av']:
        embed_dim, attn_dropout = 30, 0.0
    elif self_type == 'l_mem':
        embed_dim, attn_dropout = 2 * 30, 0.1
    elif self_type == 'a_mem':
        embed_dim, attn_dropout = 2 * 30, 0.1
    elif self_type == 'v_mem':
        embed_dim, attn_dropout = 2 * 30, 0.1
    else:
        raise ValueError("Unknown network type")

    return TransformerEncoder(embed_dim=embed_dim,
                              num_heads=5,
                              layers=min(5, layers),
                              attn_dropout=attn_dropout,
                              relu_dropout=0.1,
                              res_dropout=0.1,
                              embed_dropout=0.25,
                              attn_mask=True)



class AffectJointProcessor(torch.nn.Module):
    def __init__(self, common_dim, scenario='mosei'):
        super(AffectJointProcessor, self).__init__()

        self.common_dim = common_dim
        if scenario == 'mosei':
            # Language
            self.proj_l = nn.Conv1d(300, 30, kernel_size=1, padding=0, bias=False)
            self.trans_l_with_a = get_affect_network(self_type='la', layers=5)
            self.trans_l_with_v = get_affect_network(self_type='lv', layers=5)
            self.trans_l_mem = get_affect_network(self_type='l_mem', layers=5)

            # Audio
            self.proj_a = nn.Conv1d(74, 30, kernel_size=1, padding=0, bias=False)
            self.trans_a_with_l = get_affect_network(self_type='al', layers=5)
            self.trans_a_with_v = get_affect_network(self_type='av', layers=5)
            self.trans_a_mem = get_affect_network(self_type='a_mem', layers=5)

            # Vision
            self.proj_v = nn.Conv1d(35, 30, kernel_size=1, padding=0, bias=False)
            self.trans_v_with_l = get_affect_network(self_type='vl', layers=5)
            self.trans_v_with_a = get_affect_network(self_type='va', layers=5)
            self.trans_v_mem = get_affect_network(self_type='v_mem', layers=5)
        else:
            #Language
            self.proj_l = nn.Conv1d(300, 30, kernel_size=1, padding=0, bias=False)
            self.trans_l_with_a = get_affect_network(self_type='la', layers=5)
            self.trans_l_with_v = get_affect_network(self_type='lv', layers=5)
            self.trans_l_mem = get_affect_network(self_type='l_mem', layers=5)

            # Audio
            self.proj_a = nn.Conv1d(5, 30, kernel_size=1, padding=0, bias=False)
            self.trans_a_with_l = get_affect_network(self_type='al', layers=5)
            self.trans_a_with_v = get_affect_network(self_type='av', layers=5)
            self.trans_a_mem = get_affect_network(self_type='a_mem', layers=5)

            # Vision
            self.proj_v = nn.Conv1d(20, 30, kernel_size=1, padding=0, bias=False)
            self.trans_v_with_l = get_affect_network(self_type='vl', layers=5)
            self.trans_v_with_a = get_affect_network(self_type='va', layers=5)
            self.trans_v_mem = get_affect_network(self_type='v_mem', layers=5)


        # Projector
        self.proj1 = nn.Linear(60*3, 60*3)
        self.proj2 = nn.Linear(60*3, 60*3)
        self.projector = nn.Linear(60*3, common_dim)



    def forward(self, x):
        x_l, x_a, x_v = x[0], x[1], x[2]

        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        x_l = F.dropout(x_l.transpose(1, 2), p=0.25, training=self.training)
        x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)

        # Project the textual/visual/audio features
        proj_x_l = self.proj_l(x_l)
        proj_x_a = self.proj_a(x_a)
        proj_x_v = self.proj_v(x_v)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        # (V,A) --> L
        h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)  # Dimension (L, N, d_l)
        h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)  # Dimension (L, N, d_l)
        h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
        h_ls = self.trans_l_mem(h_ls)
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        last_h_l = h_ls[-1]  # Take the last output for prediction

        # (L,V) --> A
        h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
        h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
        h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
        h_as = self.trans_a_mem(h_as)
        if type(h_as) == tuple:
            h_as = h_as[0]
        last_h_a = h_as[-1]

        # (L,A) --> V
        h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
        h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
        h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
        h_vs = self.trans_v_mem(h_vs)
        if type(h_vs) == tuple:
            h_vs = h_vs[0]
        last_h_v = h_vs[-1]

        # Concatenate
        last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)

        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=0.0, training=self.training))
        last_hs_proj += last_hs

        # Project
        return self.projector(last_hs_proj)



class AffectGRUEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, timestep, batch_first=False):
        super(AffectGRUEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim,
                          batch_first=batch_first)
        self.projector = nn.Linear(self.hidden_dim*timestep, latent_dim)

        self.ts = timestep

    def forward(self, x):
        batch = len(x)
        input = x.reshape(batch, self.ts, self.input_dim).transpose(0, 1)
        output = self.gru(input)[0].transpose(0, 1)
        return self.projector(output.flatten(start_dim=1))


class AffectEncoder(LightningModule):

    def __init__(self, common_dim, latent_dim):
        super(AffectEncoder, self).__init__()
        self.common_dim = common_dim
        self.latent_dim = latent_dim

        self.encode = nn.Linear(common_dim, latent_dim)

    def forward(self, x):
        return F.normalize(self.encode(x), dim=-1)