import os
import json
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import MelContentEncoder, MelStyleEncoder, PhonemeEncoder, MelDecoder, VarianceAdaptor
from utils.tools import get_mask_from_lengths

import numpy as np

class CalibratedStyleSpeech(nn.Module):
    """ CalibratedStyleSpeech """

    def __init__(self, preprocess_config, model_config):
        super(CalibratedStyleSpeech, self).__init__()
        self.model_config = model_config

        self.mel_style_encoder = MelStyleEncoder(preprocess_config, model_config)
        self.mel_content_encoder = MelContentEncoder(preprocess_config, model_config)
        self.phoneme_encoder = PhonemeEncoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.mel_decoder = MelDecoder(model_config)

        self.phoneme_linear = nn.Linear(
            model_config["transformer"]["encoder_hidden"],
            model_config["transformer"]["encoder_hidden"],
        )
        self.phoneme_transform = nn.Linear(
            model_config["transformer"]["encoder_hidden"],
            model_config["melencoder"]["encoder_hidden"]
        )
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels,
        mel_lens,
        max_mel_len,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = get_mask_from_lengths(mel_lens, max_mel_len)

        style_vector = self.mel_style_encoder(mels, mel_masks)
        #speaker_emb = style_vector.detach().cpu().numpy()
        #speaker_emb = np.array(speaker_emb[0][0])
        #print(speaker_emb)
        #speaker_name = len(os.listdir("/data/tuong/Yen/calibrated_stylespeech/speaker_emb"))
        #np.save(open(os.path.join("/data/tuong/Yen/calibrated_stylespeech/speaker_emb", str(speakers[0]) + "_" + str(speaker_name) + ".npy"), "wb+"), speaker_emb)


        mel_content = self.mel_content_encoder(mels, mel_masks)

        phoneme_content = self.phoneme_encoder(texts, style_vector, src_masks)
        output = self.phoneme_linear(phoneme_content)
        phoneme_content = self.phoneme_transform(phoneme_content)
       

        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )
        
        

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )


        output, mel_masks = self.mel_decoder(output, style_vector, mel_masks)
        output = self.mel_linear(output)

        return (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            mel_content,
            phoneme_content,
            style_vector
            )
