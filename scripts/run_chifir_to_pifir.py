import argparse
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.rads.data import load_chifir_csv, load_pifir_csv
from src.rads.uncertainty import build_pool_loader, get_mc_log_probs_on_pool, compute_bald_features_from_log_probs, estimate_pseudo_prior
from src.rads.rl_selector import build_class_weights, build_sample_weights_from_pseudo_labels, train_rl_selector_for_prior, select_with_trained_agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df_source = load_chifir_csv(cfg["data"]["source_train"])
    df_target = load_pifir_csv(cfg["data"]["target_train"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    model = AutoModelForSequenceClassification.from_pretrained(cfg["model"]["name"], num_labels=2).to(device)

    # TODO:
    # 1. load a source-finetuned checkpoint instead of random init
    # 2. or call your train_source() function here

    pool_loader = build_pool_loader(df_target[cfg["data"]["text_col"]].tolist(), batch_size=cfg["uncertainty"]["batch_size"])
    log_probs = get_mc_log_probs_on_pool(
        model=model,
        tokenizer=tokenizer,
        data_loader=pool_loader,
        device=device,
        num_mc_samples=cfg["uncertainty"]["num_mc_samples"],
        max_length=cfg["model"]["max_length"],
    )

    mean_log_probs, mean_probs, predictive_entropy, expected_entropy, mutual_info = compute_bald_features_from_log_probs(log_probs)
    pseudo_labels, pi_pos, pi_neg = estimate_pseudo_prior(mean_probs)

    w_pos, w_neg = build_class_weights(pi_pos, cfg["selection"]["rho"])
    class_weights = build_sample_weights_from_pseudo_labels(pseudo_labels, w_pos, w_neg)

    agent, env, best_avg_reward = train_rl_selector_for_prior(
        mean_log_probs=mean_log_probs,
        predictive_entropy=predictive_entropy,
        mutual_info=mutual_info,
        class_weights=class_weights,
        budget=cfg["selection"]["budget"],
        num_episodes=cfg["selection"]["num_episodes"],
        lambda_diversity=cfg["selection"]["lambda_diversity"],
        gamma=cfg["selection"]["gamma"],
        batch_size=cfg["selection"]["batch_size"],
        lr=cfg["selection"]["lr"],
        epsilon_start=cfg["selection"]["epsilon_start"],
        epsilon_end=cfg["selection"]["epsilon_end"],
        epsilon_decay=cfg["selection"]["epsilon_decay"],
        replay_buffer_size=cfg["selection"]["replay_buffer_size"],
        target_update_every=cfg["selection"]["target_update_every"],
        device=device,
    )

    selected_idx = select_with_trained_agent(agent, env, device=device)
    print("Selected target indices:", selected_idx.tolist())
    print("Best avg reward:", best_avg_reward)


if __name__ == "__main__":
    main()