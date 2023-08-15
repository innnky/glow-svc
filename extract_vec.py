import math
import multiprocessing
import os
import argparse
from pathlib import Path
from random import shuffle

import torch
from glob import glob
from tqdm import tqdm

from feature_extractor import contentvec768
import utils
import logging

logging.getLogger("numba").setLevel(logging.WARNING)
import librosa


def process_one(file_path, model):
    path = Path(file_path)
    
    ssl_path = file_path.replace(".wav", ".ssl.pt")
    # try:
    #     torch.load(ssl_path)
    # except:
    if not os.path.exists(ssl_path):
        print(111)
        print(ssl_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        wav16k, sr = librosa.load(path, sr=16000)
        wav16k = torch.from_numpy(wav16k).to(device)
        ssl_content = contentvec768.get_content(model, wav_16k_tensor=wav16k)
        torch.save(ssl_content.cpu(), ssl_path)
        if not os.path.exists(ssl_path):
            print("errrrrrrrrrrrrrrrrr"*1000)
        # exit(0)
        

def process_batch(filenames):
    print("Loading hubert for content...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ssl_model = contentvec768.get_model().to(device)
    print("Loaded hubert.")
    for filename in tqdm(filenames):
        process_one(filename, ssl_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_dir", type=str, default="dataset", help="path to input dir"
    )

    args = parser.parse_args()
    filenames = glob(f"{args.in_dir}/*/*.wav", recursive=True)  # [:10]
    print(len(filenames))
    shuffle(filenames)
    multiprocessing.set_start_method("spawn", force=True)

    num_processes = 1
    chunk_size = int(math.ceil(len(filenames) / num_processes))
    chunks = [
        filenames[i : i + chunk_size] for i in range(0, len(filenames), chunk_size)
    ]
    print([len(c) for c in chunks])
    processes = [
        multiprocessing.Process(target=process_batch, args=(chunk,)) for chunk in chunks
    ]
    for p in processes:
        p.start()
