import argparse
import json
import os
import random

import numpy as np
import torch
import yaml

from src.rads.data import load_pifir_csv, load_generic_text_label_csv
from src.rads.train import train_source_classifier
from src.rads.uncertainty import (
    build_pool_loader,
    get_mc_log_probs_on_pool,
    compute_bald_features_from_log_probs,
    estimate_pseudo_prior,
)
from src.rads.rl_selector import (
    build_class_weights,
    build_sample_weights_from_pseudo_labels,
    train_rl_selector_for_prior,
    select_with_trained_agent,
)
from src.rads.evaluate import summarize_selection


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg.get("seed", 42)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    text_col = cfg["data"]["text_col"]
    label_col = "y"

    print("Loading MIMIC-CXR source data and PIFIR target data...")
    df_source_train = load_generic_text_label_csv(
        cfg["data"]["source_train"],
        text_col=text_col,
        label_col=label_col,
    )
    df_source_dev = load_generic_text_label_csv(
        cfg["data"]["source_test"],
        text_col=text_col,
        label_col=label_col,
    )
    df_target_train = load_pifir_csv(cfg["data"]["target_train"])

    print(f"Source train size: {len(df_source_train)}")
    print(f"Source dev size:   {len(df_source_dev)}")
    print(f"Target pool size:  {len(df_target_train)}")

    print("Training source-domain classifier on MIMIC-CXR...")
    source_ckpt_dir = os.path.join(output_dir, "source_model")
    artifacts = train_source_classifier(
        train_df=df_source_train,
        dev_df=df_source_dev,
        text_col=text_col,
        label_col=label_col,
        model_name=cfg["model"]["name"],
        output_dir=source_ckpt_dir,
        max_length=cfg["model"]["max_length"],
        learning_rate=cfg["training"]["learning_rate"],
        batch_size=cfg["training"]["batch_size"],
        num_train_epochs=cfg["training"]["epochs"],
        weight_decay=cfg["training"]["weight_decay"],
        early_stopping_patience=cfg["training"]["early_stopping_patience"],
        seed=cfg["seed"],
    )

    print("Extracting MC-dropout uncertainty features on PIFIR pool...")
    pool_loader = build_pool_loader(
        texts=df_target_train[text_col].tolist(),
        batch_size=cfg["uncertainty"]["batch_size"],
    )

    log_probs = get_mc_log_probs_on_pool(
        model=artifacts.model,
        tokenizer=artifacts.tokenizer,
        data_loader=pool_loader,
        device=device,
        num_mc_samples=cfg["uncertainty"]["num_mc_samples"],
        max_length=cfg["model"]["max_length"],
    )

    mean_log_probs, mean_probs, predictive_entropy, expected_entropy, mutual_info = (
        compute_bald_features_from_log_probs(log_probs)
    )

    pseudo_labels, pi_pos, pi_neg = estimate_pseudo_prior(mean_probs)
    print(f"Estimated pseudo prior on target pool: pi_pos={pi_pos:.4f}, pi_neg={pi_neg:.4f}")

    w_pos, w_neg = build_class_weights(pi_pos, cfg["selection"]["rho"])
    class_weights = build_sample_weights_from_pseudo_labels(
        pseudo_labels=pseudo_labels,
        w_pos=w_pos,
        w_neg=w_neg,
    )

    print("Training RL selector...")
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

    summary = summarize_selection(selected_idx, pseudo_labels=pseudo_labels)
    summary["best_avg_reward"] = float(best_avg_reward)
    summary["transfer_setting"] = "MIMIC_to_PIFIR"
    summary["rho"] = float(cfg["selection"]["rho"])
    summary["budget"] = int(cfg["selection"]["budget"])
    summary["pi_pos"] = float(pi_pos)
    summary["pi_neg"] = float(pi_neg)

    print("Selected target indices:", selected_idx.tolist())
    print("Selection summary:")
    print(json.dumps(summary, indent=2))

    save_json(
        {"selected_indices": [int(x) for x in selected_idx.tolist()]},
        os.path.join(output_dir, "selected_indices.json"),
    )
    save_json(
        summary,
        os.path.join(output_dir, "selection_summary.json"),
    )

    print(f"Done. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
