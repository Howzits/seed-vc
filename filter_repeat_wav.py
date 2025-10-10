# Load additional modules
import torch

import os
from tqdm import tqdm
import shutil
import os
from pathlib import Path
import concurrent.futures
from cal_similarity import CalSimilarity




def find_min_size_from_reference_audio(
    src_dir: str, dst_dir: str, num_workers: int = 128
):
    """
    优化后的多线程版本，复制每个子目录中最小的音频文件

    参数:
        src_dir: 源目录路径
        dst_dir: 目标目录路径
        num_workers: 线程池大小 (默认4)
    """
    dst_path = Path(dst_dir)
    dst_path.mkdir(parents=True, exist_ok=True)

    def process_dir(dir_path):
        dir_path = Path(dir_path)
        min_file = None
        min_size = -1 #float("inf")  # 初始化最小文件大小为无穷大

        # 使用rglob高效遍历所有文件
        for file_path in dir_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in [
                ".wav",
                ".mp3",
                ".flac",
            ]:
                file_size = file_path.stat().st_size

                # 只保留最小的文件
                if file_size > min_size:
                    min_size = file_size
                    min_file = file_path

        if min_file:
            target_path = dst_path / min_file.name
            shutil.copy2(min_file, target_path)
            return f"Copied: {min_file} -> {target_path}"
        return f"No audio files in: {dir_path}"

    # 收集所有待处理的子目录
    dirs_to_process = []
    for root, dirs, _ in os.walk(src_dir):
        dirs_to_process.extend(os.path.join(root, d) for d in dirs)

    # 使用线程池并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for dir_path in dirs_to_process:
            futures.append(executor.submit(process_dir, dir_path))

        # 使用tqdm显示进度
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Processing directories",
        ):
            try:
                result = future.result()
                print(result)
            except Exception as e:
                print(f"Error processing directory: {e}")


def find_duplicates(embeddings_dict, threshold=0.95):
    """
    基于向量相似度找出重复文件
    参数:
        embeddings_dict: {文件路径: 向量} 的字典
        threshold: 相似度阈值(默认0.95)
    返回:
        重复文件组的列表 [[主文件, 重复文件1, ...], ...]
    """
    files = list(embeddings_dict.keys())
    embeddings = list(embeddings_dict.values())
    duplicates = []
    processed = set()

    for i in range(len(files)):
        if files[i] in processed:
            continue

        # 找出与当前文件相似的所有文件
        similar_files = [files[i]]
        for j in range(i + 1, len(files)):
            if files[j] in processed:
                continue

            similarity = torch.cosine_similarity(
                embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)
            ).item()

            if similarity > threshold:
                similar_files.append(files[j])
                processed.add(files[j])

        if len(similar_files) > 1:
            duplicates.append(similar_files)

    return duplicates


def filter_and_copy(
    src_dir: str,
    dst_dir: str,
    num_workers: int = 128,
    similarity_threshold: float = 0.7,
):
    """
    带重复过滤的智能复制功能
    1. 为每个文件生成嵌入向量
    2. 基于向量相似度过滤重复
    3. 只保留每组重复文件中最大的文件
    """
    dst_path = Path(dst_dir)
    dst_path.mkdir(parents=True, exist_ok=True)

    # 第一步: 收集所有音频文件路径
    audio_files = []
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith((".wav", ".mp3", ".flac")):
                audio_files.append(os.path.join(root, file))

    # 第二步: 多线程生成嵌入向量
    embeddings = {}
    cal_similarity = CalSimilarity()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {
            executor.submit(cal_similarity.audio_to_embedding, f): f
            for f in audio_files
        }

        for future in tqdm(
            concurrent.futures.as_completed(future_to_file),
            total=len(audio_files),
            desc="Generating embeddings",
        ):
            file = future_to_file[future]
            try:
                embeddings[file] = future.result()
            except Exception as e:
                print(f"Error processing {file}: {e}")

    # 第三步: 找出重复文件组
    duplicate_groups = find_duplicates(embeddings, threshold=similarity_threshold)

    # 第四步: 处理重复文件 - 每组只保留最大的文件
    files_to_copy = set()
    for group in duplicate_groups:
        # 找出组中最大的文件
        max_file = max(group, key=lambda x: os.path.getsize(x))
        files_to_copy.add(max_file)

    # 添加非重复文件
    for file in audio_files:
        is_duplicate = any(file in group for group in duplicate_groups)
        if not is_duplicate:
            files_to_copy.add(file)

    # 第五步: 复制选定的文件
    for file in tqdm(files_to_copy, desc="Copying files"):
        target_path = dst_path / Path(file).name
        shutil.copy2(file, target_path)

    print(f"\n完成! 共处理 {len(audio_files)} 个文件")
    print(f"发现 {len(duplicate_groups)} 组重复文件")
    print(f"最终保留 {len(files_to_copy)} 个唯一文件")


if __name__ == "__main__":
    # 2. 找到同一个文件夹下size最小的
    src_path = r"/99_TemporaryData/haochen75/seed_vc_training/ref_audio_split_by_dir/val"
    tgt_path = r"/99_TemporaryData/haochen75/seed_vc_training/ref_audio_split_by_dir/val_mini_one"
    find_min_size_from_reference_audio(src_path, tgt_path)

    # 3. 过滤重复音频并复制到新目录
    # filter_and_copy('/99_TemporaryData/haochen75/seed_vc_training/ref_audio_split_max_audio/', '/99_TemporaryData/haochen75/seed_vc_training/ref_audio_unique1/', num_workers=1, similarity_threshold=0.7)