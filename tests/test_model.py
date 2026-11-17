import unittest

import torch

from model import CoDINOForForecasting, count_parameters, get_model_config
from model.architecture_config import architecture_from_dict
from model.blocks.sparse_moe import communication_moe_modules
from model.objectives import forecasting_relative_l1


class CoDINOModelTests(unittest.TestCase):
    def test_four_registered_variants(self):
        expected = {
            "T": (24, 12_117_648),
            "S": (32, 21_508_352),
            "B": (38, 30_307_484),
            "L": (48, 48_317_664),
        }
        for name, (width, parameters) in expected.items():
            config = get_model_config(name)
            self.assertEqual(config.embed_dim, width)
            self.assertEqual(config.depths, (2, 2, 2, 2))
            self.assertEqual(config.patch_size, 2)
            self.assertEqual(count_parameters(CoDINOForForecasting(config))[
                "total_real_scalars"], parameters)
            self.assertEqual(architecture_from_dict(config.to_dict()), config)

    def test_forward_preserves_field_shape(self):
        model = CoDINOForForecasting(get_model_config("T")).eval()
        with torch.inference_mode():
            output = model(torch.randn(1, 4, 33, 31), torch.tensor([0.05]))
        self.assertEqual(tuple(output.shape), (1, 4, 33, 31))

    def test_only_selected_legacy_checkpoint_is_accepted(self):
        config = get_model_config("B")
        legacy = {**config.to_dict(),
                  "aggregation": "shared_mean_plus_sparse_routed",
                  "routing_mode": "top4_softmax",
                  "communication": "selected_expert_hidden_residual",
                  "spectral_mode": "radial_angular",
                  "spectral_angle_offset_deg": 0.0,
                  "spectral_directional_mixers": True,
                  "spectral_filter_cutoff": 0.25,
                  "spatial_mixer": "sparse_communication",
                  "spatial_reference_experts": 8,
                  "talklora_rank": 16,
                  "local_branch_mode": "zero"}
        self.assertEqual(architecture_from_dict(legacy), config)
        historical_l = dict(legacy)
        historical_l.update(model_size="B", embed_dim=48,
                            stage_dims=[48, 96, 192, 384],
                            expert_hidden_dims=[48, 96, 192, 384])
        self.assertEqual(architecture_from_dict(historical_l), get_model_config("L"))
        legacy["spatial_mixer"] = "sparse_moe"
        with self.assertRaises(ValueError):
            architecture_from_dict(legacy)

    def test_fixed_top4_communication(self):
        model = CoDINOForForecasting(get_model_config("T")).eval()
        module = next(iter(communication_moe_modules(model).values()))
        with torch.inference_mode():
            _, state = module(torch.randn(2, 3, 3, 24), return_router_state=True)
        self.assertEqual(tuple(state["topk_indices"].shape), (18, 4))
        self.assertTrue(torch.allclose(state["topk_weights"].sum(-1), torch.ones(18)))
        self.assertEqual(tuple(module.communication.mix.shape), (8, 8))

    def test_spectral_masks_partition_unity(self):
        model = CoDINOForForecasting(get_model_config("T"))
        spectral = model.encoder.stages[0][0].spectral
        masks = spectral.frequency_masks(17, 19)
        self.assertEqual(tuple(masks.shape), (4, 4, 17, 10))
        self.assertTrue(torch.allclose(masks.sum((0, 1)), torch.ones(17, 10), atol=1e-6))

    def test_relative_l1_masks_ns_constant_channels(self):
        prediction = torch.zeros(1, 4, 2, 2)
        target = torch.ones_like(prediction)
        prediction[:, 0] = 1000
        prediction[:, 3] = 1000
        mask = torch.tensor([[True, False, False, True]])
        self.assertAlmostEqual(
            forecasting_relative_l1(prediction, target, mask).item(), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
