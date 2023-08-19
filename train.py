import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import logging

from vocos import Vocos

logging.getLogger("matplotlib").setLevel(logging.INFO)
logging.getLogger("h5py").setLevel(logging.INFO)
logging.getLogger("numba").setLevel(logging.INFO)

from data_utils import TextAudioSpeakerLoader, TextAudioSpeakerCollate
import models
import commons
import utils
# from hifigan import NsfHifiGAN

global_step = 0


def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '7998'

    hps = utils.get_hparams()
    mp.spawn(train_and_eval, nprocs=n_gpus, args=(n_gpus, hps,))


def train_and_eval(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    dist.init_process_group(backend= 'gloo' if os.name == 'nt' else 'nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)

    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True)
    collate_fn = TextAudioSpeakerCollate()
    train_loader = DataLoader(train_dataset, num_workers=3, shuffle=False,
                              batch_size=hps.train.batch_size, pin_memory=True,
                              drop_last=True, collate_fn=collate_fn, sampler=train_sampler, persistent_workers=True)
    if rank == 0:
        val_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data, val=True)
        val_loader = DataLoader(val_dataset, num_workers=0, shuffle=False,
                                batch_size=1, pin_memory=True,
                                drop_last=True, collate_fn=collate_fn)

    generator = models.FlowGenerator(
        n_vocab=0,
        out_channels=hps.data.n_mel_channels,
        **hps.model).cuda(rank)
    vocoder = Vocos.from_pretrained('pretrain/vocos/config.yaml', 'pretrain/vocos/pytorch_model.bin').cuda()
    optimizer_g = commons.Adam(generator.parameters(), scheduler=hps.train.scheduler,
                               dim_model=hps.model.hidden_channels, warmup_steps=hps.train.warmup_steps,
                               lr=hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)


    # optimizer_g = commons.Adam(generator.parameters(), scheduler=hps.train.scheduler,
    #                            dim_model=hps.model.hidden_channels, warmup_steps=hps.train.warmup_steps,
    #                            lr=hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
    generator = DDP(generator)
    epoch_str = 1
    global_step = 0
    #
    # try:
    #     _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), generator,
    #                                                optimizer_g)
    #     optimizer_g.step_num = (epoch_str - 1) * len(train_loader)
    #     optimizer_g._update_learning_rate()
    #     global_step = (epoch_str - 1) * len(train_loader)
    # except:
    #     if hps.train.ddi and os.path.isfile(os.path.join(hps.model_dir, "ddi_G.pth")):
    #         _ = utils.load_checkpoint(os.path.join(hps.model_dir, "ddi_G.pth"), generator, optimizer_g)
    #
    skip_optimizer = False
    try:
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), generator,
                                                   optimizer_g, False)
        epoch_str += 1
        optimizer_g.step_num = (epoch_str - 1) * len(train_loader)
        optimizer_g._update_learning_rate()
        global_step = (epoch_str - 1) * len(train_loader)
    except:
        epoch_str = 1
        global_step = 0
    if skip_optimizer:
        epoch_str = 1
        global_step = 0

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            save_interval = 60
            if epoch % save_interval == 0:
                evaluate(rank, epoch, hps, generator, optimizer_g, val_loader, logger, writer_eval, vocoder)
            train(rank, epoch, hps, generator, optimizer_g, train_loader, logger, writer)
            if epoch % save_interval == 0:
                utils.save_checkpoint(generator, optimizer_g, hps.train.learning_rate, epoch,
                                      os.path.join(hps.model_dir, "G_{}.pth".format(epoch)))
                try:
                    to_remove_path = os.path.join(hps.model_dir, "G_{}.pth".format(epoch - save_interval* 3))
                    os.remove(to_remove_path)
                    print(f'removing {to_remove_path}')
                except:
                    print(f'removing {to_remove_path} failed')
        else:
            train(rank, epoch, hps, generator, optimizer_g, train_loader, None, None)


def train(rank, epoch, hps, generator, optimizer_g, train_loader, logger, writer):
    train_loader.sampler.set_epoch(epoch)
    global global_step

    generator.train()
    for batch_idx, (x, mel,mel_lengths,wav, wav_lengths, speakers, f0) in enumerate(tqdm(train_loader)):
        mel, mel_lengths = mel.cuda(rank, non_blocking=True), mel_lengths.cuda(rank, non_blocking=True)
        speakers = speakers.cuda(rank, non_blocking=True)
        x = x.cuda(rank, non_blocking=True)
        f0 = f0.cuda(rank, non_blocking=True)

        # Train Generator
        optimizer_g.zero_grad()

        (z, z_m, z_logs, logdet, z_mask), l_noise = generator(x, mel, mel_lengths,f0, g=speakers, gen=False)
        l_mle = commons.mle_loss(z, z_m, z_logs, logdet, z_mask)

        loss_gs = [l_mle, l_noise]
        loss_g = sum(loss_gs)

        loss_g.backward()
        grad_norm = commons.clip_grad_value_(generator.parameters(), 5)
        optimizer_g.step()
        if rank == 0:
            if batch_idx % hps.train.log_interval == 0:
                y_gen, _ = generator.module(x[:1],f0=f0[:1], g=speakers[:1], gen=True, glow=True)
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(x), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                    loss_g.item()))
                lr = optimizer_g._optim.param_groups[0]['lr']
                logger.info([x.item() for x in loss_gs] + [global_step, lr])

                scalar_dict = {"loss/g/total": loss_g, "learning_rate": lr, "grad_norm": grad_norm}
                scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(loss_gs)})
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images={"train/gt/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
                            "train/gen/mel": utils.plot_spectrogram_to_numpy(y_gen[0].data.cpu().numpy())
                            },
                    scalars=scalar_dict)
        global_step += 1

    if rank == 0:
        logger.info('====> Epoch: {}'.format(epoch))


def evaluate(rank, epoch, hps, generator, optimizer_g, val_loader, logger, writer_eval, vocoder):
    if rank == 0:
        global global_step
        generator.eval()
        audio_dict = {}
        img_dict = {}
        with torch.no_grad():
            for batch_idx, (x, mel,mel_lengths,wav, wav_lengths, speakers, f0) in enumerate(
                    val_loader):
                mel, mel_lengths = mel.cuda(rank, non_blocking=True), mel_lengths.cuda(rank, non_blocking=True)
                speakers = speakers.cuda(rank, non_blocking=True)
                x = x.cuda(rank, non_blocking=True)
                f0 = f0.cuda(rank, non_blocking=True)

                mel_flow, pred_f0 = generator.module(x, f0=f0, g=speakers, gen=True, glow=True)
                y_flow = vocoder.decode(mel_flow)

                # mel_diff, pred_f0 = generator.module(x, f0=f0,g=speakers, gen=True, glow=False)
                # y_diff = vocoder.spec2wav(mel_diff.squeeze(0).transpose(0, 1).cpu().numpy(),
                #                          f0=pred_f0[0, 0, :].cpu().numpy())

                y_rec = vocoder.decode(mel)

                img_dict.update({f"gt/mel_{batch_idx}": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
                                 f"gen/mel_flow_{batch_idx}": utils.plot_spectrogram_to_numpy(mel_flow[0].data.cpu().numpy()),
                                 # f"gen/mel_diff_{batch_idx}": utils.plot_spectrogram_to_numpy(mel_diff[0].data.cpu().numpy()),
                                 })
                audio_dict.update({
                    # "gen/wav_gen_{}_diff".format(batch_idx): y_diff,
                    "gen/wav_gen_{}_flow".format(batch_idx): y_flow[0].cpu().numpy(),
                    "gt/wav_gen_{}_rec".format(batch_idx): y_rec[0].cpu().numpy()
                })

        utils.summarize(
            writer=writer_eval,
            global_step=global_step,
            images=img_dict,
            audios=audio_dict,
            audio_sampling_rate=hps.data.sampling_rate
        )
        logger.info('====> Epoch: {}'.format(epoch))


if __name__ == "__main__":
    main()
