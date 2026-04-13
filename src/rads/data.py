import pandas as pd
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = list(labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def load_chifir_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    df = df.rename(columns={"y_report": "y"})
    df["y"] = df["y"].map({"Positive": 1, "Negative": 0})
    return df


def load_pifir_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    df = df.rename(columns={"id_label1": "y"})
    df["y"] = df["y"].map({
        "1. Known IFI": 1,
        "2. Likely IFI": 1,
        "3. Potential IFI": 1,
        "4. Unlikely IFI": 0,
        "5. Other infections / non-IFI": 0,
    })
    return df


def load_generic_text_label_csv(path: str, text_col: str = "order_results", label_col: str = "y") -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    return df[[text_col, label_col]]