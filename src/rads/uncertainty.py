import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class TextPoolDataset(Dataset):
    def __init__(self, texts):
        self.texts = list(texts)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def build_pool_loader(texts, batch_size=16):
    dataset = TextPoolDataset(texts)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def get_mc_log_probs_on_pool(
    model,
    tokenizer,
    data_loader,
    device,
    num_mc_samples=10,
    max_length=512,
):
    model.to(device)
    model.train()

    all_mc_log_probs = []

    with torch.no_grad():
        for _ in range(num_mc_samples):
            mc_log_probs_list = []
            for batch_texts in data_loader:
                enc = tokenizer(
                    list(batch_texts),
                    truncation=True,
                    padding=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(device)

                outputs = model(**enc)
                log_probs = F.log_softmax(outputs.logits, dim=-1)
                mc_log_probs_list.append(log_probs.cpu())

            mc_log_probs = torch.cat(mc_log_probs_list, dim=0)
            all_mc_log_probs.append(mc_log_probs)

    log_probs_N_K_C = torch.stack(all_mc_log_probs, dim=1)
    return log_probs_N_K_C


def compute_bald_features_from_log_probs(log_probs_N_K_C: torch.Tensor):
    log_probs = log_probs_N_K_C.double()
    probs = log_probs.exp()

    mean_probs = probs.mean(dim=1)
    mean_log_probs = mean_probs.clamp(min=1e-12).log()

    predictive_entropy = -torch.sum(mean_probs * mean_log_probs, dim=-1)
    entropy_per_mc = -torch.sum(probs * log_probs, dim=-1)
    expected_entropy = entropy_per_mc.mean(dim=1)
    mutual_info = predictive_entropy - expected_entropy

    return mean_log_probs, mean_probs, predictive_entropy, expected_entropy, mutual_info


def estimate_pseudo_prior(mean_probs: torch.Tensor):
    pseudo_labels = mean_probs.argmax(dim=1).cpu().numpy()
    pi_pos = float((pseudo_labels == 1).mean())
    pi_neg = 1.0 - pi_pos
    return pseudo_labels, pi_pos, pi_neg