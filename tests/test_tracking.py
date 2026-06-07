import os
import unittest
from unittest.mock import patch

from trainer.tracking import resolve_wandb_settings


class WandbSettingsTest(unittest.TestCase):
    def test_defaults_to_disabled_without_entity(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = resolve_wandb_settings(
                {}, default_project="default-project"
            )

        self.assertEqual(settings["mode"], "disabled")
        self.assertEqual(settings["project"], "default-project")
        self.assertIsNone(settings["entity"])

    def test_environment_variables_override_public_config(self):
        environment = {
            "WANDB_MODE": "online",
            "WANDB_PROJECT": "local-project",
            "WANDB_ENTITY": "local-user",
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = resolve_wandb_settings(
                {
                    "wandb_mode": "offline",
                    "wandb_project": "tracked-project",
                },
                default_project="default-project",
            )

        self.assertEqual(settings["mode"], "online")
        self.assertEqual(settings["project"], "local-project")
        self.assertEqual(settings["entity"], "local-user")

    def test_rejects_unknown_mode(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                resolve_wandb_settings(
                    {"wandb_mode": "sometimes"},
                    default_project="default-project",
                )


if __name__ == "__main__":
    unittest.main()
