import argparse
import os
from numpy import average
import json

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import DistributedSampler, DataLoader
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from torch.cuda import amp

from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample, get_mask_from_lengths
from model import StyleSpeechLoss, CalibratedStyleSpeechLoss
from dataset import Dataset

import glob
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav, spectral_normalize_torch
from hifigan import Generator

from evaluate import evaluate

torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '54321'
os.environ["WORLD_SIZE"] = str(1)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(rank, args, configs, batch_size, num_gpus):
    print("Start training on GPU {} with batchsize {}".format(rank, batch_size))
    preprocess_config, model_config, train_config = configs
    os.environ["RANK"] = str(rank)
    if num_gpus > 1:
        init_process_group(
            backend=train_config["dist_config"]['dist_backend'],
            #init_method=train_config["dist_config"]['dist_url'],
            #world_size=train_config["dist_config"]['world_size'] * num_gpus,
            #rank=rank,
            #init_method="env://"
        )
    device = torch.device('cuda:{}'.format(rank))
    # Get dataset
    dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    data_sampler = DistributedSampler(dataset) if num_gpus > 1 else None
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=False,
        sampler=data_sampler,
        collate_fn=dataset.collate_fn,
    )

    # Prepare model
    model, optimizer, cs_mi_net, cs_mi_net_optimizer = get_model(args, configs, device, train=True)
    if num_gpus > 1:
        model = DistributedDataParallel(model, device_ids=[rank]).to(device)
        cs_mi_net = DistributedDataParallel(cs_mi_net, device_ids=[rank], find_unused_parameters=True).to(device)
    scaler = amp.GradScaler(enabled=args.use_amp)
    num_param = get_param_num(model)
    Loss = CalibratedStyleSpeechLoss(preprocess_config, model_config).to(device)
    print("Number of Calibrated Parameters:", num_param)

    # Load vocoder
    print("Loading vocoder: {}".format(args.vocoder))
    config_file = os.path.join('/data/tuong/Yen/hifi-gan/cp_hifigan/config.json')
    with open(config_file) as f:
        data = f.read()

    #json_config = json.loads(data)
    #h = AttrDict(json_config)
    #vocoder = Generator(h)
    #state_dict_g = torch.load(args.vocoder, map_location=device)
    #vocoder.load_state_dict(state_dict_g["generator"])
    #vocoder.eval()
    #vocoder.remove_weight_norm()
    #vocoder.to(device)
    print("Done")
    vocoder = None

    # Init logger
    if rank == 0:
        for p in train_config["path"].values():
            os.makedirs(p, exist_ok=True)
        train_log_path = os.path.join(train_config["path"]["log_path"], "train")
        val_log_path = os.path.join(train_config["path"]["log_path"], "val")
        os.makedirs(train_log_path, exist_ok=True)
        os.makedirs(val_log_path, exist_ok=True)
        train_logger = SummaryWriter(train_log_path)
        val_logger = SummaryWriter(val_log_path)

    # Training
    step = args.restore_step + 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    if rank == 0:
        outer_bar = tqdm(total=total_step, desc="Training", position=0)
        outer_bar.n = args.restore_step
        outer_bar.update()

    while True:
        if rank == 0:
            inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        if num_gpus > 1:
            data_sampler.set_epoch(epoch)
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)

                # MI fisrt formward to update the prior distribution 
                # q_theta(y=style|x=content)
                # Get detached content and style vectors from current Stylenet
                #_,_,_,_,_,_,_,_,_,content,_,style = model(*batch[2:]) #[bs, mel_len, 80]
                src_masks = get_mask_from_lengths(lengths=batch[4], max_len=batch[5])
                mel_masks = get_mask_from_lengths(lengths=batch[7], max_len=batch[8])

                #print("Welcome ")
                if num_gpus > 1:
                    style = model.module.mel_style_encoder(batch[6], mel_masks).detach()
                    content = model.module.mel_content_encoder(batch[6], mel_masks).detach()
                    content, style = content.squeeze(1), style.squeeze(1)
                    for i in range(model_config["mi"]["mi_iters"]):
                        cs_mi_net_optimizer.zero_grad()
                        lld_cs_mi = -cs_mi_net.module.loglikeli(content, style)
                        lld_cs_mi.backward()
                        cs_mi_net_optimizer.step_and_update_lr()
                else:
                #print("Hello wel
                    style = model.mel_style_encoder(batch[6], mel_masks).detach()
                    content = model.mel_content_encoder(batch[6], mel_masks).detach()
                    content, style = content.squeeze(1), style.squeeze(1)
                    for i in range(model_config["mi"]["mi_iters"]):
                        cs_mi_net_optimizer.zero_grad()
                        lld_cs_mi = -cs_mi_net.loglikeli(content, style)
                        lld_cs_mi.backward()
                        cs_mi_net_optimizer.step_and_update_lr()
                # Forward
                output = model(*(batch[2:]))
                # MI second forward to update the joint distribution 
                # p(x=content, y=style)
                mi_cs_loss = model_config["mi"]["mi_weight"] * \
                                cs_mi_net.forward(content, style)
                # Cal Loss
                losses = Loss(batch, output)
                total_loss = losses[0] + mi_cs_loss

                # Backward
                total_loss = total_loss / grad_acc_step
              
                # Backward
                scaler.scale(total_loss).backward()
                if step % grad_acc_step == 0:
                    # Clipping gradients to avoid gradient explosion
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                    # Update weights
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()
                if rank == 0:
                    if step % log_step == 0:
                        losses = [l.item() for l in losses]
                        message1 = "Step {}/{}, ".format(step, total_step)
                        message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}, Content Loss : {:4f}, MI Loss: {:.4f}".format(
                        losses[0] + mi_cs_loss, *losses[1:], mi_cs_loss
                    )

                        with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                            f.write(message1 + message2 + "\n")

                        outer_bar.write(message1 + message2)
                        #losses.update({"MI loss": mi_cs_loss})
                        losses.append(mi_cs_loss)
                        log(train_logger, step, losses=losses)

                    if step % synth_step == 0:
                        fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                            batch,
                            output,
                            vocoder,
                            model_config,
                            preprocess_config,
                        )
                        log(
                            train_logger,
                            fig=fig,
                            tag="Training/step_{}_{}".format(step, tag),
                        )
                        #sampling_rate = preprocess_config["preprocessing"]["audio"][
                        #    "sampling_rate"
                        #]
                        #log(
                        #    train_logger,
                        #    audio=wav_reconstruction,
                        #    sampling_rate=sampling_rate,
                        #    tag="Training/step_{}_{}_reconstructed".format(step, tag),
                    #)
                        #log(
                        #    train_logger,
                        #    audio=wav_prediction,
                        #sampling_rate=sampling_rate,
                        #tag="Training/step_{}_{}_synthesized".format(step, tag),
                    #)

                    if step % val_step == 0:
                        model.eval()
                        models = (model, cs_mi_net)
                        message = evaluate(models, step, configs, val_logger, vocoder)
                        with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                            f.write(message + "\n")
                        outer_bar.write(message)

                        model.train()

                    if step % save_step == 0:
                        if num_gpus > 1: torch.save(
                            {
                                "model": model.module.state_dict(),
                                "optimizer": optimizer._optimizer.state_dict(),
                                "cs_mi_net": cs_mi_net.module.state_dict(),
                                "cs_mi_net_optimizer": cs_mi_net_optimizer._optimizer.state_dict()
                            },
                            os.path.join(
                            train_config["path"]["ckpt_path"],
                            "{}.pth.tar".format(step),
                        ),
                    )
                        else: torch.save(
                            {
                                "model": model.state_dict(),
                                "optimizer": optimizer._optimizer.state_dict(),
                                "cs_mi_net": cs_mi_net.state_dict(),
                                "cs_mi_net_optimizer": cs_mi_net_optimizer._optimizer.state_dict()
                            },
                            os.path.join(
                            train_config["path"]["ckpt_path"],
                            "{}.pth.tar".format(step),
                        ),)


                if step == total_step:
                    quit()
                step += 1
                if rank == 0:
                    outer_bar.update(1)

            inner_bar.update(1)
        epoch += 1


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CPU training is not allowed."
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    parser.add_argument(
        "--vocoder", type=str, default="/data/tuong/Yen/hifi-gan/cp_hifigan/g_00320000", 
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    num_gpus = torch.cuda.device_count()
    batch_size = int(train_config["optimizer"]["batch_size"] / num_gpus)

    print("\n==================================== Training Configuration ====================================")
    print(' ---> Automatic Mixed Precision:', args.use_amp)
    print(' ---> Number of used GPU:', num_gpus)
    print(' ---> Batch size per GPU:', batch_size)
    print(' ---> Batch size in total:', batch_size * num_gpus)
    print(" ---> Type of Duration Modeling:", "unsupervised" if model_config["duration_modeling"]["learn_alignment"] else "supervised")
    print("=================================================================================================")
    print("Prepare training ...")

    if num_gpus > 1:
        mp.spawn(train, nprocs=num_gpus, args=(args, configs, batch_size, num_gpus))
    else:
        train(0, args, configs, batch_size, num_gpus)
