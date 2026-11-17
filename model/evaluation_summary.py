"""Compact, explicitly labeled summaries of Co-DINO evaluation reports."""


def _normalized_mean(metric):
    normalized = metric["metric_spaces"]["native_normalized"]
    return {
        "mean_relative_l1_percent": normalized["joint"]["mean_relative_l1_percent"],
        "mean_relative_l2_percent": 100.0 * normalized["relative_l2"],
    }


def compact_evaluation_report(detailed):
    """Keep dt1 and rollout means; omit every rollout step from the simple file."""
    compact = {
        "checkpoint": detailed["checkpoint"],
        "checkpoint_kind": detailed.get("checkpoint_kind"),
        "training_stage": detailed.get("training_stage"),
        "training_policy": detailed.get("training_policy"),
        "split": detailed["split"],
        "result_format": {
            "kind": "simple_summary",
            "metric_space": "native_normalized",
            "metric_units": "percent",
            "relative_l1_l2": "mean of per-trajectory joint-channel relative errors",
            "rollout_mean": "mean over trajectories and all predicted time steps",
        },
        "evaluation_config": detailed["evaluation_config"],
        "datasets": {},
    }
    for name, entry in detailed["datasets"].items():
        dt1 = entry["one_step"]
        summary = {
            "evaluated_trajectories": entry["evaluated_trajectories"],
            "eval_channels": dt1["eval_channels"],
            "dt1": {
                "target_time": dt1["target_time"],
                "normalized": _normalized_mean(dt1),
            },
        }
        rollout = entry.get("autoregressive")
        if rollout is not None:
            summary["rollout_mean"] = {
                "steps": rollout["steps"],
                "normalized": _normalized_mean(rollout["overall"]),
            }
        compact["datasets"][name] = summary
    return compact
