from modules.openvoice.se_extractor import split_audio_vad
import os
import librosa
import shutil
import numpy as np
import concurrent.futures
from tqdm import tqdm


def process_audio_file(args):
    """处理单个音频文件的任务函数"""
    root, filename, tgt_path = args
    try:
        audio_path = os.path.join(root, filename)
        audio, sample_rate = librosa.load(audio_path, sr=None)
        audio_duration = len(audio) / sample_rate
        audio_duration = np.ceil(audio_duration)

        if audio_duration < 1:  # 少于1s的音频不处理
            return f"Skipped {filename}: too short (< 1s)"

        audio_n = filename.rsplit(".", 1)[0]

        if audio_duration > 28 and audio_duration <= 58:  # 分割为两半
            split_audio_vad(
                audio_path,
                target_dir=tgt_path,
                audio_name=audio_n,
                split_seconds=audio_duration / 2,
            )
        elif audio_duration > 58:  # 分割为20秒片段
            split_audio_vad(
                audio_path,
                target_dir=tgt_path,
                audio_name=audio_n,
                split_seconds=20,
            )
        else:  # 直接复制
            audio_na = audio_n.split("_")[0]
            s_path = f"{tgt_path}/{audio_na}"
            if not os.path.exists(s_path):
                os.makedirs(s_path)
            shutil.copy2(audio_path, s_path)
        return f"Processed {filename}: {audio_duration} seconds"
    except Exception as e:
        return f"Error processing {filename}: {str(e)}"


def split_data(src_path, tgt_path, max_workers=8):
    # 收集所有需要处理的文件
    tasks = []
    for root, dir, filenames in os.walk(src_path):
        for filename in filenames:
            if filename.endswith(".wav") or filename.endswith(".mp3"):
                tasks.append((root, filename, tgt_path))

    # 使用线程池并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_task = {
            executor.submit(process_audio_file, task): task for task in tasks
        }

        # 使用tqdm显示进度
        for future in tqdm(
            concurrent.futures.as_completed(future_to_task),
            total=len(tasks),
            desc="Processing audio files",
        ):
            result = future.result()
            # 可以选择打印结果或记录日志
            # print(result)


def split_data_by_directory(src_path, tgt_path, max_workers=8):
    tasks = []
    for root, dirs, _ in os.walk(src_path):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            tasks.append(dir_path)

    l = len(tasks)

    ratio = int(l * 0.9)
    train_tasks = tasks[:ratio]
    val_tasks = tasks[ratio:]
    print(f"训练集目录数: {len(train_tasks)}, 验证集目录数: {len(val_tasks)}")

    # 处理训练集
    for t in tqdm(train_tasks, desc="Processing training directories"):
        shutil.copytree(t, os.path.join(tgt_path, "train", t.split("/")[-1]))

    for t in tqdm(val_tasks, desc="Processing validation directories"):
        shutil.copytree(t, os.path.join(tgt_path, "val", t.split("/")[-1]))

    print(tasks)


if __name__ == "__main__":
    # 1.分割参考音频
    # src_path = r"/99_TemporaryData/haochen75/seed_vc_training/ref_audio_unique"
    # tgt_path = r"/99_TemporaryData/haochen75/seed_vc_training/ref_audio_train_val/"
    # split_data(src_path, tgt_path, max_workers=128)

    # 2.分割故事音频
    # src_path = r"/01_Data/03_Audio/09_VoiceConversion/2025/1007个故事原音"
    # tgt_path = r"/99_TemporaryData/haochen75/story_audio_split"
    # split_data(src_path, tgt_path, max_workers=128)

    # 3. 分割用户音频
    # src_path = r"/99_TemporaryData/haochen75/seed_vc_training/ref_audio_filtered/"
    # tgt_path = r"/99_TemporaryData/haochen75/seed_vc_training/ref_audio_split2"
    # split_data(src_path, tgt_path, max_workers=128)

    # 4. 分割用户音频
    # src_path = r"/99_TemporaryData/haochen75/seed_vc_training/ref_audio_max_audio/"
    # tgt_path = r"/99_TemporaryData/haochen75/seed_vc_training/ref_audio_max_audio_split"
    # split_data(src_path, tgt_path, max_workers=128)

    # 5. 分割用户音频，按目录
    src_path = r"/99_TemporaryData/haochen75/seed_vc_training/ref_audio_split"
    tgt_path = r"/99_TemporaryData/haochen75/seed_vc_training/ref_audio_split_by_dir"
    split_data_by_directory(src_path, tgt_path, max_workers=128)
