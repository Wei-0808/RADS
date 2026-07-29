import numpy as np
import torch


def test_import():
    import src.rads.data
    import src.rads.evaluate
    import src.rads.metrics
    import src.rads.rl_selector
    import src.rads.train
    import src.rads.uncertainty
    import src.rads.utils


def test_training_args_accepted_by_installed_transformers(tmp_path):
    """Guards against TrainingArguments keyword renames across transformers releases."""
    from src.rads.train import SourceTrainConfig, build_training_args

    cfg = SourceTrainConfig(model_name="prajjwal1/bert-tiny", output_dir=str(tmp_path))
    args = build_training_args(cfg)

    assert args.load_best_model_at_end is True
    assert args.eval_strategy == "epoch"  # IntervalStrategy is a str enum
    assert args.save_strategy == "epoch"


def test_bald_features_match_closed_form():
    from src.rads.uncertainty import compute_bald_features_from_log_probs

    # One report, two MC samples that disagree completely -> maximal mutual information.
    probs = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]).clamp(min=1e-12)
    _, mean_probs, pred_entropy, expected_entropy, mutual_info = (
        compute_bald_features_from_log_probs(probs.log())
    )

    assert np.allclose(mean_probs.numpy(), [[0.5, 0.5]])
    assert np.isclose(pred_entropy.item(), np.log(2))
    assert np.isclose(expected_entropy.item(), 0.0, atol=1e-6)
    assert np.isclose(mutual_info.item(), np.log(2))


def test_selector_env_respects_budget():
    from src.rads.rl_selector import RLSampleSelectionEnv

    n, budget = 20, 3
    env = RLSampleSelectionEnv(
        mean_log_probs=torch.full((n, 2), -float(np.log(2))),
        predictive_entropy=torch.full((n,), float(np.log(2))),
        mutual_info=torch.linspace(0.0, 1.0, n),
        class_weights=np.ones(n),
        budget=budget,
    )

    env.reset()
    done = False
    while not done:
        _, _, done = env.step(1)

    assert len(env.selected_indices) == budget
