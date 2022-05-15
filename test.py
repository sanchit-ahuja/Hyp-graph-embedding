import torch
from sklearn.metrics import classification_report
import json
from dataset import create_dataset
from torch.utils.data import DataLoader
from vanilla_model import BaseModel

def test(model, data_loader, device, json_path):
    test_preds = []
    test_true_l = []
    test_logits = []
    model.eval()
    for data in data_loader:
        x_feats = data["x"]
        labels = data["y"]
        test_mask = data["test_mask"]
        with torch.no_grad():
            logits = model(x_feats)
            # logits = model(batch, h, c, 'test',save=save)
        test_logits.append(logits)
        true_labels = labels[test_mask == 1]
        pred = torch.argmax(logits, 1)
        test_preds.extend(pred.to("cpu"))
        test_true_l.extend(true_labels.to("cpu"))

    test_true_l = torch.tensor(test_true_l).to(device)
    test_true_l = test_true_l.detach().cpu().numpy()
    test_preds = torch.tensor(test_preds).to(device)
    test_preds = test_preds.detach().cpu().numpy()
    report = classification_report(test_true_l, test_preds, output_dict=True)
    print(report)
    # dump classification_report to json
    with open(json_path, "w") as f:
        json.dump(report, f)


def main():
    data_dir = "pheme_dgl_full_roberta_final.pkl"
    train_dataset, _, test_dataset = create_dataset(data_dir)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    num_classes = train_dataset.num_classes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BaseModel(num_classes=num_classes, hidden_size=1026, dropout=0.4).to(device)
    model.load_state_dict(torch.load("saved/0.7882496291587202_model.pt"))
    test(model, test_loader, device, "logs/test_report_cb.json")

if __name__ == "__main__":
    main()
