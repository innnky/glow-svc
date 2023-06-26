import os
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data_utils import TextMelLoader, TextMelCollate
import models
import commons
import utils
from text.symbols import symbols
from vocos import Vocos
from hifigan import NsfHifiGAN

global_step = 0


def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8000'

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

    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)

    train_dataset = TextMelLoader(hps.data.training_files, hps.data)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True)
    collate_fn = TextMelCollate(1)
    train_loader = DataLoader(train_dataset, num_workers=3, shuffle=False,
                              batch_size=hps.train.batch_size, pin_memory=True,
                              drop_last=True, collate_fn=collate_fn, sampler=train_sampler)
    if rank == 0:
        val_dataset = TextMelLoader(hps.data.validation_files, hps.data)
        val_loader = DataLoader(val_dataset, num_workers=0, shuffle=False,
                                batch_size=1, pin_memory=True,
                                drop_last=True, collate_fn=collate_fn)

    generator = models.FlowGenerator(
        n_vocab=len(symbols) + getattr(hps.data, "add_blank", False),
        out_channels=hps.data.n_mel_channels,
        **hps.model).cuda(rank)
    # vocoder = Vocos.from_pretrained('vocos/config.yaml', 'vocos/pytorch_model.bin').cuda()
    vocoder = NsfHifiGAN('cuda')
    optimizer_g = commons.Adam(generator.parameters(), scheduler=hps.train.scheduler,
                               dim_model=hps.model.hidden_channels, warmup_steps=hps.train.warmup_steps,
                               lr=hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
    generator = DDP(generator)
    epoch_str = 1
    global_step = 0
    scaler = GradScaler(enabled=hps.train.fp16_run)

    try:
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), generator,
                                                   optimizer_g)
        epoch_str += 1
        optimizer_g.step_num = (epoch_str - 1) * len(train_loader)
        optimizer_g._update_learning_rate()
        global_step = (epoch_str - 1) * len(train_loader)
    except:
        if hps.train.ddi and os.path.isfile(os.path.join(hps.model_dir, "ddi_G.pth")):
            _ = utils.load_checkpoint(os.path.join(hps.model_dir, "ddi_G.pth"), generator, optimizer_g)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            evaluate(rank, epoch, hps, generator, optimizer_g, val_loader, logger, writer_eval, vocoder)
            utils.save_checkpoint(generator, optimizer_g, hps.train.learning_rate, epoch,
                                  os.path.join(hps.model_dir, "G_{}.pth".format(epoch)))
            try:
                to_remove_path = os.path.join(hps.model_dir, "G_{}.pth".format(epoch - 3))
                os.remove(to_remove_path)
                print(f'removing {to_remove_path}')
            except:
                print(f'removing {to_remove_path} failed')
            train(rank, epoch, hps, generator, optimizer_g, scaler, train_loader, logger, writer)
        else:
            train(rank, epoch, hps, generator, optimizer_g, scaler, train_loader, None, None)


def train(rank, epoch, hps, generator, optimizer_g, scaler, train_loader, logger, writer):
    train_loader.sampler.set_epoch(epoch)
    global global_step

    generator.train()
    for batch_idx, (x, x_lengths, tones, y, y_lengths, sid) in enumerate(train_loader):
        x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
        y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)
        tones = tones.cuda(rank, non_blocking=True)
        sid = sid.cuda(rank, non_blocking=True)

        # Train Generator
        optimizer_g.zero_grad()

        with autocast(enabled=hps.train.fp16_run):
            (z, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (attn, logw, logw_) = generator(x, tones, x_lengths, y,
                                                                                                     y_lengths, g=sid,
                                                                                                     gen=False)
            l_mle = commons.mle_loss(z, z_m, z_logs, logdet, z_mask)
            l_length = commons.duration_loss(logw, logw_, x_lengths)

            loss_gs = [l_mle, l_length]
            loss_g = sum(loss_gs)

        optimizer_g.zero_grad()
        scaler.scale(loss_g).backward()
        scaler.unscale_(optimizer_g)
        grad_norm = commons.clip_grad_value_(generator.parameters(), 5)
        scaler.step(optimizer_g)
        scaler.update()
        if rank == 0:
            if batch_idx % hps.train.log_interval == 0:
                (y_gen, *_), *_ = generator.module(x[:1], tones[:1], x_lengths[:1], g=sid[:1], gen=True)
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(x), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                    loss_g.item()))
                lr = optimizer_g.get_lr()
                logger.info([x.item() for x in loss_gs] + [global_step, lr])

                scalar_dict = {"loss/g/total": loss_g, "learning_rate": lr, "grad_norm": grad_norm}
                scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(loss_gs)})
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images={"train/gt/y_org": utils.plot_spectrogram_to_numpy(y[0].data.cpu().numpy()),
                            "train/gen/y_gen": utils.plot_spectrogram_to_numpy(y_gen[0].data.cpu().numpy()),
                            "train/attn": utils.plot_alignment_to_numpy(attn[0, 0].data.cpu().numpy()),
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
            for batch_idx, (x, x_lengths, tones, y, y_lengths, sid) in enumerate(val_loader):
                x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
                y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)
                tones = tones.cuda(rank, non_blocking=True)
                sid = sid.cuda(rank, non_blocking=True)

                (y_gen, *_), *_ = generator.module(x[:1], tones[:1], x_lengths[:1], g=sid[:1], gen=True)
                wav_gen = vocoder.decode(y_gen)
                wav_rec = vocoder.decode(y)
                img_dict.update({f"gt/y_org_{batch_idx}": utils.plot_spectrogram_to_numpy(y[0].data.cpu().numpy()),
                                 f"gen/y_gen_{batch_idx}": utils.plot_spectrogram_to_numpy(y_gen[0].data.cpu().numpy()),
                                 })
                audio_dict.update({
                    "gen/wav_gen_{}".format(batch_idx): wav_gen,
                    "gt/wav_rec_{}".format(batch_idx): wav_rec,
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
