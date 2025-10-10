import os
import shutil
from tqdm import tqdm
import random


# 故事音频分割训练集和验证集
def split_dataset_story(src, tgt):
    """将数据集分割为训练集和验证集"""
    if not os.path.exists(tgt):
        os.makedirs(tgt)
    if not os.path.exists(os.path.join(tgt, "train")):
        os.makedirs(os.path.join(tgt, "train"))
    if not os.path.exists(os.path.join(tgt, "val")):
        os.makedirs(os.path.join(tgt, "val"))
    files = []
    for root, dirs, _ in os.walk(src):
        for dir in dirs:
            if "wav" not in dir:
                continue
            for sub_root, sub_dirs, sub_filenames in os.walk(os.path.join(root, dir)):
                l = len(sub_filenames)
                if l == 0:
                    continue

                select_one = random.randint(0, len(sub_filenames) - 1)
                # print(sub_filenames)

                select_one_file = os.path.join(sub_root, sub_filenames[select_one])
                print(f"Selecting {select_one} from {select_one_file} for validation")
                shutil.copy(select_one_file, os.path.join(tgt, "val"))

                for filename in sub_filenames:
                    if filename == sub_filenames[select_one]:
                        continue
                    src_file = os.path.join(sub_root, filename)
                    shutil.copy(src_file, os.path.join(tgt, "train"))


def split_dataset_ratio(src_path, tgt_path, ratio=0.9):

    all_files = []
    for root, dirs, files in os.walk(src_path):
        for file in files:
            # print(f"Found file: {file} in {root}")
            f_path = os.path.join(root, file)
            print(f"Adding file: {f_path}")
            all_files.append(f_path)

    split_index = int(len(all_files) * ratio)
    train_files = all_files[:split_index]
    val_files = all_files[split_index:]
    os.makedirs(os.path.join(tgt_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(tgt_path, "val"), exist_ok=True)
    for file in tqdm(train_files, desc="Copying training files"):
        shutil.copy(file, os.path.join(tgt_path, "train"))

    for file in tqdm(val_files, desc="Copying validation files"):
        shutil.copy(file, os.path.join(tgt_path, "val"))


if __name__ == "__main__":
    split_dataset_ratio(
        "/01_Data/03_Audio/09_VoiceConversion/2025/1007个故事原音",
        "/99_TemporaryData/haochen75/seed_vc_training/story_audio_train_val_99",
    )
