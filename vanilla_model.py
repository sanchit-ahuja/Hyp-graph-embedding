import torch.nn as nn
import torch
from dataset import create_dataset
from torch.utils.data import DataLoader


class BaseModel(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout):
        super(BaseModel, self).__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.lin1 = nn.Linear(hidden_size, num_classes)

    def forward(self, inp_graph):
        # inp_graph = torch.cat([inp_graph, hyp_feat], dim=-1)
        inp_graph = self.dropout(inp_graph)
        out = self.lin1(inp_graph)
        return out


if __name__ == "__main__":
    data_dir = "pheme_tmp.pkl"
    train_dataset, val_dataset, test_dataset = create_dataset(data_dir)
    train_dl = DataLoader(train_dataset, batch_size=8, shuffle=True)
    num_classes = train_dataset.num_classes
    hidden_size = 1026
    model = BaseModel(num_classes, hidden_size, 0.5).to('cuda')
    ds = next(iter(train_dl))
    x = ds['x'].to('cuda')
    fin = model(x)
    print(fin.shape, "shape")
