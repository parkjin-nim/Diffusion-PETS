"""Dataset adapters used only by cross-repository comparison notebooks."""

import os

import numpy as np

from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one
from Utils.Data_utils.real_datasets import CustomDataset


class TrainScalerCustomDataset(CustomDataset):
    """Use a train-only scaler while evaluating a separate chronological CSV.

    The Diffusion-TS core dataset is inherited unchanged.  A delayed-save path
    handles ``proportion=0`` without asking the original implementation to
    inverse-transform an empty train array.
    """

    def __init__(
        self,
        scaler_data_root=None,
        proportion=0.8,
        save2npy=True,
        **kwargs,
    ):
        self._comparison_scaler_data_root = scaler_data_root
        delayed_save = bool(save2npy and float(proportion) == 0.0)
        super().__init__(
            proportion=proportion,
            save2npy=False if delayed_save else save2npy,
            **kwargs,
        )
        if delayed_save:
            self.save2npy = True
            self._save_test_outputs()

    def read_data(self, filepath, name=""):
        rawdata, scaler = CustomDataset.read_data(filepath, name)
        if self._comparison_scaler_data_root is not None:
            _, scaler = CustomDataset.read_data(
                self._comparison_scaler_data_root,
                name,
            )
        return rawdata, scaler

    def _save_test_outputs(self):
        if self.period != "test":
            return
        np.save(
            os.path.join(
                self.dir,
                f"{self.name}_ground_truth_{self.window}_test.npy",
            ),
            self.unnormalize(self.samples),
        )
        normalized_truth = (
            unnormalize_to_zero_to_one(self.samples)
            if self.auto_norm
            else self.samples
        )
        np.save(
            os.path.join(
                self.dir,
                f"{self.name}_norm_truth_{self.window}_test.npy",
            ),
            normalized_truth,
        )
        if self.missing_ratio is not None:
            np.save(
                os.path.join(
                    self.dir,
                    f"{self.name}_masking_{self.window}.npy",
                ),
                self.masking,
            )
