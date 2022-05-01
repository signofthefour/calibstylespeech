import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.model import get_model, get_vocoder
from utils.tools import to_device, log, synth_one_sample, get_mask_from_lengths
from model import CalibratedStyleSpeechLoss
from dataset import Dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(models, step, configs, logger=None, vocoder=None):
    model = models[0]
    cs_mi_net = models[1]
    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    # Get loss function
    Loss = CalibratedStyleSpeechLoss(preprocess_config, model_config).to(device)

    # Evaluation
    loss_sums = [0 for _ in range(7)]
    for batchs in loader:
        for batch in batchs:
            batch = to_device(batch, device)
            with torch.no_grad():

                src_masks = get_mask_from_lengths(lengths=batch[4], max_len=batch[5])
                mel_masks = get_mask_from_lengths(lengths=batch[7], max_len=batch[8])


                style = model.mel_style_encoder(batch[6], mel_masks).detach()
                content = model.mel_content_encoder(batch[6], mel_masks).detach()
                content, style = content.squeeze(1), style.squeeze(1)
                # Forward
                output = model(*(batch[2:]))
                mi_cs_loss = model_config["mi"]["mi_weight"] * cs_mi_net.forward(content, style)

                # Cal Loss
                losses = list(Loss(batch, output))
                losses[0] = losses[0] + mi_cs_loss
                losses = losses + [mi_cs_loss]

                for i in range(len(losses)):
                    loss_sums[i] += losses[i].item() * len(batch[0])
                
    loss_means = [loss_sum / len(dataset) for loss_sum in loss_sums]

    message = "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}, Content Loss {:4f}, MI loss (:4f)".format(
        *([step] + [l for l in loss_means])
    )

    if logger is not None:
        fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
            batch,
            output,
            vocoder,
            model_config,
            preprocess_config,
        )

        log(logger, step, losses=loss_means)
        log(
            logger,
            fig=fig,
            tag="Validation/step_{}_{}".format(step, tag),
        )
        #sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        #log(
        #    logger,
        #    audio=wav_reconstruction,
        #    sampling_rate=sampling_rate,<
        #    tag="Validation/step_{}_{}_reconstructed".format(step, tag),
        #)
        #log(
        #    logger,
        #    audio=wav_prediction,
        #    sampling_rate=sampling_rate,
        #    tag="Validation/step_{}_{}_synthesized".format(step, tag),
        #)

    return message
