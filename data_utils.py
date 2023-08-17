import time
import os
import random
import numpy as np
import torch
import torch.utils.data

import commons
import mel_processing
from utils import load_filepaths_and_text
import torch.nn.functional as F
"""Multi speaker version"""


class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """

    def __init__(self, audiopaths_sid_text, hparams, val=False):
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sampling_rate = hparams.sampling_rate
        self.spk_map = hparams.spk2id

        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 300)
        self.hps = hparams
        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)
        self._filter(val)
        self.fcpe = None

    def _filter(self, val):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_sid_text_new = []
        lengths = []
        skipped = 0
        for item in self.audiopaths_sid_text:
            _id, spk = item[:2]
            audiopath =  f'dataset/{spk}/{_id}.wav'
            if not os.path.exists(audiopath):
                skipped += 1
                continue
            length_ = os.path.getsize(audiopath) // (2 * self.hop_length)
            if (length_ < 120 or length_>1400 ) and not val:
                skipped += 1
                continue
            audiopaths_sid_text_new.append([audiopath, spk])

        print("skipped: ", skipped, ", total: ", len(self.audiopaths_sid_text))
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths


    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        # separate filename, speaker_id and text
        audiopath, sid = audiopath_sid_text

        mel, wav = self.get_spec(audiopath)

        ssl = torch.load(audiopath.replace(".wav", ".ssl.pt"))
        ssl = F.interpolate(ssl, size=mel.shape[-1], mode="nearest")

        sid = torch.LongTensor([int(self.spk_map[sid])])
        f0 = self.get_pitch(wav[0], mel.shape[1], audiopath)
        return (ssl, mel, wav, sid, f0)

    def get_spec(self, filename):
        wav_torch, _ = mel_processing.load_wav_to_torch(filename, target_sr=self.hps.sampling_rate)
        mel_path = filename.replace(".wav", ".mel.pt")
        if os.path.exists(mel_path):
            mel = torch.load(mel_path)
            return mel, wav_torch.unsqueeze(0)

        mel = mel_processing.get_mel(wav_torch,
                                     self.hps.sampling_rate,
                                     self.hps.n_mel_channels,
                                     self.hps.filter_length,
                                     self.hps.win_length,
                                     self.hps.hop_length,
                                     self.hps.mel_fmin,
                                     self.hps.mel_fmax)
        torch.save(mel, mel_path)
        return mel, wav_torch.unsqueeze(0)

    def get_text(self, text, tone, language):
        text_norm, tone, language = cleaned_text_to_sequence(text, tone, language)
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
            tone = commons.intersperse(tone, 0)
            language = commons.intersperse(language, 0)
        text_norm = torch.LongTensor(text_norm)
        tone = torch.LongTensor(tone)
        language = torch.LongTensor(language)
        return text_norm, tone, language

    def get_pitch(self, wav, p_len, wavpath):
        f0_path = wavpath.replace(".wav", ".f0.pt")
        if os.path.exists(f0_path):
            return torch.load(f0_path)


        if self.fcpe is None:
            from f0_extractor.FCPEF0Predictor import FCPEF0Predictor
            print("init fcpe")
            self.fcpe = FCPEF0Predictor(sampling_rate=self.sampling_rate, hop_length=self.hop_length)
        pred_f0, uv = self.fcpe.compute_f0_uv(wav, p_len=p_len)
        f0 = torch.FloatTensor(pred_f0)
        torch.save(f0, f0_path)
        return f0

    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)


class TextAudioSpeakerCollate():
    """ Zero-pads model inputs and targets
    """

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0, descending=True)

        max_mel_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        mel_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))

        c_padded = torch.FloatTensor(len(batch), batch[0][0].size(1), max_mel_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        mel_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_mel_len)
        f0_padded = torch.FloatTensor(len(batch), max_mel_len)
        c_padded.zero_()
        mel_padded.zero_()
        wav_padded.zero_()
        f0_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            content = row[0][0,:, :]
            c_padded[i,:, :content.size(1)] = content

            mel = row[1]
            mel_padded[i, :, :mel.size(1)] = mel
            mel_lengths[i] = mel.size(1)

            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            sid[i] = row[3]

            f0 = row[4]
            f0_padded[i, :f0.size(0)] = f0

        return c_padded, mel_padded, mel_lengths,wav_padded, wav_lengths,\
            sid, f0_padded


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]

            # subsample
            ids_bucket = ids_bucket[self.rank::self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [bucket[idx] for idx in ids_bucket[j * self.batch_size:(j + 1) * self.batch_size]]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size
