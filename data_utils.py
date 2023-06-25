import os
import random
import numpy as np
import torch
import torch.utils.data
import torchaudio

import commons 
from utils import load_wav_to_torch, load_filepaths_and_text
from text.symbols import symbols
from text import cleaned_text_to_sequence
from vocos.feature_extractors import MelSpectrogramFeatures


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.add_noise = hparams.add_noise
        self.add_blank = getattr(hparams, "add_blank", False) # improved version
        self.feature_extractor = MelSpectrogramFeatures()
        self.spk_map = hparams.spk2id
        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)
        self._filter()

    def _filter(self):
        audiopaths_sid_text_new = []
        lengths = []
        skip_num = 0
        for _id, spk, text, tone in self.audiopaths_and_text:
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                audiopath = f'dataset/{spk}/{_id}.wav'
                if not os.path.exists(audiopath):
                    skip_num += 1
                    continue
                length_ = os.path.getsize(audiopath) // (2 * self.hop_length)
                if length_ < 60 or length_ > 1500:
                    skip_num += 1
                    continue
                text = text.split(" ")
                tone = tone.split(" ")
                audiopaths_sid_text_new.append([audiopath, spk, text, tone])
                lengths.append(length_)
        print("skip:", skip_num, "samples！")
        self.audiopaths_and_text = audiopaths_sid_text_new
        self.lengths = lengths

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, sid, text, tone = audiopath_and_text
        text, tone = self.get_text(text, tone)
        mel = self.get_spec(audiopath)
        return (text, mel,sid,tone)


    def get_spec(self, filename):
        y, sr = torchaudio.load(filename)
        assert y.size(0) == 1
        assert sr == self.sampling_rate

        mel_filename = filename.replace(".wav", ".mel.pt")
        assert mel_filename.endswith(".pt")
        if os.path.exists(mel_filename):
            try:
                mel = torch.load(mel_filename)
            except:
                mel = self.get_mel(y, mel_filename)
        else:
            mel = self.get_mel(y, mel_filename)
        return mel

    def get_mel(self, audio_norm, mel_filename):
        mel = self.feature_extractor(audio_norm)
        mel = mel.squeeze(0)
        torch.save(mel, mel_filename)
        return mel

    def get_text(self, text, tone):
        text_norm = cleaned_text_to_sequence(text)
        tone = [int(i) for i in tone]
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
            tone = commons.intersperse(tone, 0)
        text_norm = torch.LongTensor(text_norm)
        tone = torch.LongTensor(tone)
        return text_norm, tone

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per step
    """
    def __init__(self, n_frames_per_step=1):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0, descending=True)

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        tone_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        text_padded.zero_()
        tone_padded.zero_()
        spec_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)


            sid[i] = row[2]

            tone = row[3]
            tone_padded[i, :tone.size(0)] = tone

        return text_padded, text_lengths, tone_padded, spec_padded, spec_lengths, sid

