from typing import Any
from functools import partial
import logging
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from lightning.pytorch.callbacks import Callback
from torch import Tensor, nn
from torchgeo.datasets.utils import unbind_samples
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryJaccardIndex, BinaryPrecision, BinaryRecall

from terratorch.models.model import AuxiliaryHead, ModelOutput
from terratorch.tasks.loss_handler import LossHandler
from terratorch.tasks.tiled_inference import tiled_inference
from terratorch.tasks.base_task import TerraTorchTask

BATCH_IDX_FOR_VALIDATION_PLOTTING = 10

logger = logging.getLogger("terratorch")


def to_segmentation_prediction(y: ModelOutput) -> Tensor:
    y_hat = y.output
    return nn.Sigmoid()(y_hat).round()

class BinarySemanticSegmentationTask(TerraTorchTask):
    """BinarySemantic Segmentation Task that accepts models from a range of sources.

        - Does NOT accept the specification of a model factory
        - Logs metrics per class
        - Does not have any callbacks by default (TorchGeo tasks do early stopping by default)
        - Allows the setting of optimizers in the constructor
        - Allows to evaluate on multiple test dataloaders
    """

    def __init__(
        self,
        model: nn.Module,
        loss: str = "ce",
        aux_heads: list[AuxiliaryHead] | None = None,
        aux_loss: dict[str, float] | None = None,
        focal_alpha: float = None,
        ignore_index: int | None = None,
        lr: float = 0.001,
        # the following are optional so CLI doesnt need to pass them
        optimizer: str | None = None,
        optimizer_hparams: dict | None = None,
        scheduler: str | None = None,
        scheduler_hparams: dict | None = None,
        #
        freeze_backbone: bool = False,  # noqa: FBT001, FBT002
        freeze_decoder: bool = False,  # noqa: FBT002, FBT001
        freeze_head: bool = False, 
        plot_on_val: bool | int = 10,
        tiled_inference_parameters: dict = None,
        test_dataloaders_names: list[str] | None = None,
        lr_overrides: dict[str, float] | None = None,
        output_on_inference: str | list[str] = "probabilities",
        path_to_record_metrics: str = None,
        tiled_inference_on_testing: bool = False,
    ) -> None:
        """Constructor

        Args:
            Defaults to None.
            model (torch.nn.Module): Custom model.
            loss (str, optional): Loss to be used. Currently, supports 'ce'|'bce', 'jaccard', 'dice' or 'focal' loss.
                Defaults to "ce".
            aux_loss (dict[str, float] | None, optional): Auxiliary loss weights.
                Should be a dictionary where the key is the name given to the loss
                and the value is the weight to be applied to that loss.
                The name of the loss should match the key in the dictionary output by the model's forward
                method containing that output. Defaults to None.
            focal_alpha (float | None, optional): alpha coefficient for the Focal loss.
            ignore_index (int | None, optional): Label to ignore in the loss computation. Defaults to None.
            lr (float, optional): Learning rate to be used. Defaults to 0.001.
            optimizer (str | None, optional): Name of optimizer class from torch.optim to be used.
            If None, will use Adam. Defaults to None. Overriden by config / cli specification through LightningCLI.
            optimizer_hparams (dict | None): Parameters to be passed for instantiation of the optimizer.
                Overriden by config / cli specification through LightningCLI.
            scheduler (str, optional): Name of Torch scheduler class from torch.optim.lr_scheduler
                to be used (e.g. ReduceLROnPlateau). Defaults to None.
                Overriden by config / cli specification through LightningCLI.
            scheduler_hparams (dict | None): Parameters to be passed for instantiation of the scheduler.
                Overriden by config / cli specification through LightningCLI.
            freeze_backbone (bool, optional): Whether to freeze the backbone. Defaults to False.
            freeze_decoder (bool, optional): Whether to freeze the decoder. Defaults to False.
            freeze_head (bool, optional): Whether to freeze the segmentation head. Defaults to False.
            plot_on_val (bool | int, optional): Whether to plot visualizations on validation.
            If true, log every epoch. Defaults to 10. If int, will plot every plot_on_val epochs.
            tiled_inference_parameters (dict | None, optional): Inference parameters
                used to determine if inference is done on the whole image or through tiling.
            test_dataloaders_names (list[str] | None, optional): Names used to differentiate metrics when
                multiple dataloaders are returned by test_dataloader in the datamodule. Defaults to None,
                which assumes only one test dataloader is used.
            lr_overrides (dict[str, float] | None, optional): Dictionary to override the default lr in specific
                parameters. The key should be a substring of the parameter names (it will check the substring is
                contained in the parameter name) and the value should be the new lr. Defaults to None.
            output_on_inference (str | list[str]): A string or a list defining the kind of output to be saved to file during the inference, for example,
                it can be "prediction", to save just the most probable class, or ["prediction", "probabilities"] to save both prediction and probabilities.
            tiled_inference_on_testing (bool): A boolean to define if tiled inference will be used when full inference 
                fails during the test step. 
            path_to_record_metrics (str): A path to save the file containing the metrics log. 
        """

        self.tiled_inference_parameters = tiled_inference_parameters
        self.aux_loss = aux_loss
        self.aux_heads = aux_heads

        super().__init__(task="segmentation", tiled_inference_on_testing=tiled_inference_on_testing,
                         path_to_record_metrics=path_to_record_metrics)

        self.model = model

        self.train_loss_handler = LossHandler(self.train_metrics.prefix)
        self.test_loss_handler: list[LossHandler] = []
        for metrics in self.test_metrics:
            self.test_loss_handler.append(LossHandler(metrics.prefix))
        self.val_loss_handler = LossHandler(self.val_metrics.prefix)
        self.monitor = f"{self.val_metrics.prefix}loss"
        self.plot_on_val = int(plot_on_val)
        self.output_on_inference = output_on_inference

        # Processing the `output_on_inference` argument.
        self.output_prediction = lambda y: (nn.Sigmoid()(y).round(), "pred")
        self.output_logits = lambda y: (y, "logits")
        self.output_probabilities = lambda y: (nn.Sigmoid()(y), "probabilities")

        # The possible methods to define outputs.
        self.operation_map = {
                              "prediction": self.output_prediction, 
                              "logits": self.output_logits, 
                              "probabilities": self.output_probabilities
                              }

        # `output_on_inference` can be a list or a string.
        if isinstance(output_on_inference, list):
            list_of_selectors = ()
            for var in output_on_inference:
                if var in self.operation_map:
                    list_of_selectors += (self.operation_map[var],)
                else:
                    raise ValueError(f"Option {var} is not supported. It must be in ['logits', 'prediction', 'probabilities']")

            if not len(list_of_selectors):
                raise ValueError("The list of selectors for the output is empty, please, provide a valid value for `output_on_inference`")

            self.select_classes = lambda y: [op(y) for op in
                                                   list_of_selectors]
        elif isinstance(output_on_inference, str):
            self.select_classes = self.operation_map[output_on_inference]

        else:
            raise ValueError(f"The value {output_on_inference} isn't supported for `output_on_inference`.")

    def configure_losses(self) -> None:
        """Initialize the loss criterion.

        Raises:
            ValueError: If *loss* is invalid.
        """
        loss: str = self.hparams["loss"]
        ignore_index = self.hparams["ignore_index"]
        
        focal_alpha = self.hparams["focal_alpha"]
        if loss!= "focal" and focal_alpha is not None:
                exception_message = (
                    f"Found non-None value of {focal_alpha} for `focal_alpha`, but loss is '{loss}' and not 'focal'."
                )
                raise RuntimeError(exception_message)       

        if loss == "ce" or loss == "bce":
            if ignore_index is not None:
                exception_message = (
                    f"BCE loss does not support ignore_index, but found non-None value of {ignore_index}."
                )
                raise RuntimeError(exception_message)
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss == "jaccard":
            if ignore_index is not None:
                exception_message = (
                    f"Jaccard loss does not support ignore_index, but found non-None value of {ignore_index}."
                )
                raise RuntimeError(exception_message)
            self.criterion = smp.losses.JaccardLoss(mode="binary")
        elif loss == "focal":
            self.criterion = smp.losses.FocalLoss("binary", alpha=focal_alpha, ignore_index=ignore_index, normalized=True)
        elif loss == "dice":
            self.criterion = smp.losses.DiceLoss("binary", ignore_index=ignore_index)
        else:
            exception_message = (
                f"Loss type '{loss}' is not valid. Currently, supports 'ce'|'bce', 'jaccard', 'dice' or 'focal' loss."
            )
            raise ValueError(exception_message)

    def configure_metrics(self) -> None:
        """Initialize the performance metrics."""
        ignore_index: int = self.hparams["ignore_index"]
        metrics = MetricCollection(
            {
                "F1_Score": BinaryF1Score(
                    ignore_index=ignore_index,
                ),
                "Precision": BinaryPrecision(
                    ignore_index=ignore_index,
                ),
                "Recall": BinaryRecall(
                    ignore_index=ignore_index,
                ),
                "Accuracy": BinaryAccuracy(
                    ignore_index=ignore_index
                ),
                "IoU": BinaryJaccardIndex(
                        ignore_index=ignore_index,
                )
            }
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        if self.hparams["test_dataloaders_names"] is not None:
            self.test_metrics = nn.ModuleList(
                [metrics.clone(prefix=f"test/{dl_name}/") for dl_name in self.hparams["test_dataloaders_names"]]
            )
        else:
            self.test_metrics = nn.ModuleList([metrics.clone(prefix="test/")])

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        """Compute the train loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        # Testing because of failures.
        x = batch["image"]
        y = batch["mask"]
        other_keys = batch.keys() - {"image", "mask", "filename"}

        rest = {k: batch[k] for k in other_keys}
        model_output: ModelOutput = self(x, **rest)
        loss = self.train_loss_handler.compute_loss(model_output, y, self.criterion, self.aux_loss)
        self.train_loss_handler.log_loss(self.log, loss_dict=loss, batch_size=y.shape[0])
        y_hat_hard = to_segmentation_prediction(model_output)
        self.train_metrics.update(y_hat_hard, y)

        return loss["loss"]

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the test loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        y = batch["mask"]
        other_keys = batch.keys() - {"image", "mask", "filename"}

        rest = {k: batch[k] for k in other_keys}

        model_output = self.handle_full_or_tiled_inference(x, None, **rest)

        if dataloader_idx >= len(self.test_loss_handler):
            msg = "You are returning more than one test dataloader but not defining enough test_dataloaders_names."
            raise ValueError(msg)
        loss = self.test_loss_handler[dataloader_idx].compute_loss(model_output, y, self.criterion, self.aux_loss)
        self.test_loss_handler[dataloader_idx].log_loss(
            partial(self.log, add_dataloader_idx=False),  # We don't need the dataloader idx as prefixes are different
            loss_dict=loss,
            batch_size=y.shape[0],
        )
        y_hat_hard = to_segmentation_prediction(model_output)
        self.test_metrics[dataloader_idx].update(y_hat_hard, y)

        self.record_metrics(dataloader_idx, y_hat_hard, y)


    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the validation loss and additional metrics.
        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        y = batch["mask"]
        file_coords = batch["filename"]

        other_keys = batch.keys() - {"image", "mask", "filename"}
        rest = {k: batch[k] for k in other_keys}
        model_output: ModelOutput = self(x, **rest)

        loss = self.val_loss_handler.compute_loss(model_output, y, self.criterion, self.aux_loss)
        self.val_loss_handler.log_loss(self.log, loss_dict=loss, batch_size=y.shape[0])
        y_hat_hard = to_segmentation_prediction(model_output)
        
        self.val_metrics.update(y_hat_hard, y)

        if self._do_plot_samples(batch_idx):
            try:
                datamodule = self.trainer.datamodule
                batch["prediction"] = y_hat_hard

                if isinstance(batch["image"], dict):
                    rgb_modality = getattr(datamodule, 'rgb_modality', None) or list(batch["image"].keys())[0]
                    batch["image"] = batch["image"][rgb_modality]

                for key in ["image", "mask", "prediction"]:
                    batch[key] = batch[key].cpu()
                sample = unbind_samples(batch)[0]
                fig = datamodule.val_dataset.plot(sample)
                if fig:
                    summary_writer = self.logger.experiment
                    if hasattr(summary_writer, "add_figure"):
                        summary_writer.add_figure(f"image/{batch_idx}", fig, global_step=self.global_step)
                    elif hasattr(summary_writer, "log_figure"):
                        summary_writer.log_figure(
                            self.logger.run_id, fig, f"epoch_{self.current_epoch}_{batch_idx}.png"
                        )
            except ValueError:
                pass
            finally:
                plt.close()

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        """Compute the predicted class probabilities.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            Output predicted probabilities.
        """
        x = batch["image"]
        file_names = batch["filename"] if "filename" in batch else None
        other_keys = batch.keys() - {"image", "mask", "filename"}

        rest = {k: batch[k] for k in other_keys}

        def model_forward(x,  **kwargs):
            return self(x, **kwargs).output

        if self.tiled_inference_parameters:
            y_hat: Tensor = tiled_inference(
                model_forward,
                x,
                **self.tiled_inference_parameters,
                **rest,
            )
        else:
            y_hat: Tensor = self(x, **rest).output

        y_hat_ = self.select_classes(y_hat)

        return y_hat_, file_names
