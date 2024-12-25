import os
import glob
import torchaudio
import json
from utils.mp3_info import get_mp3_metadata
import argparse

def prepare_data(input_dir, output_dir, segment_length=2.0, sample_rate=48000):
    """
    Prepares data for training by:
    1. Scanning input_dir for mp3 files
    2. Extracting audio to wav
    3. Segmenting into uniform segments
    4. Saving metadata (from mp3info) as JSON next to each wav
    """
    os.makedirs(output_dir, exist_ok=True)
    mp3_files = glob.glob(os.path.join(input_dir, "*.mp3"))
    index = []

    for f in mp3_files:
        audio, sr = torchaudio.load(f)
        if sr != sample_rate:
            audio = torchaudio.functional.resample(audio, sr, sample_rate)
        
        duration = audio.shape[-1] / sample_rate
        seg_samples = int(segment_length * sample_rate)
        
        meta = get_mp3_metadata(f)
        
        base_name = os.path.splitext(os.path.basename(f))[0]
        num_segments = int(duration // segment_length)
        for i in range(num_segments):
            seg = audio[:, i*seg_samples:(i+1)*seg_samples]
            if seg.shape[-1] < seg_samples:
                continue
            seg_path = os.path.join(output_dir, f"{base_name}_{i}.wav")
            torchaudio.save(seg_path, seg, sample_rate)
            meta_path = seg_path.replace(".wav", ".json")
            with open(meta_path, "w") as metaf:
                json.dump(meta, metaf)
            index.append(seg_path)
    
    # Save index file
    with open(os.path.join(output_dir, "train_index.txt"), "w") as f:
        for line in index:
            f.write(line + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--segment_length", type=float, default=2)
    parser.add_argument("--sample_rate", type=int, default=48000)
    args = parser.parse_args()
    prepare_data(args.input_dir, args.output_dir, args.segment_length, args.sample_rate)
