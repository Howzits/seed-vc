import glob
from pathlib import Path
import torch
from modules.campplus.DTDNN import CAMPPlus
import torchaudio


class CalSimilarity:
    def __init__(
        self,
        model_path=r"./checkpoints/models--funasr--campplus/snapshots/fb71fe990cbf6031ae6987a2d76fe64f94377b7e/campplus_cn_common.bin",
        device=torch.device("cuda"),
    ):
        self.model_path = model_path
        self.device = device
        self.campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        self.campplus_model.load_state_dict(
            torch.load(self.model_path, map_location=self.device, weights_only=True)
        )
        self.campplus_model.eval()

    def audio_to_embedding(self, audio_path):
        waveform, orig_sr = torchaudio.load(audio_path)
        resampled_waveform = torchaudio.functional.resample(waveform, orig_sr, 16000)
        mel_s = torchaudio.compliance.kaldi.fbank(
            resampled_waveform, num_mel_bins=80, dither=0, sample_frequency=16000
        )
        # print(f"Resampled waveform shape: {resampled_waveform.shape}")
        with torch.no_grad():
            embedding = self.campplus_model(mel_s.unsqueeze(0))
            # print(f"Embedding shape: {embedding.shape}")
            return embedding.squeeze(0)

    def cal_similarity(self, audio_path1, audio_path2):
        """
        计算两个音频文件的相似度
        """
        embedding1 = self.audio_to_embedding(audio_path1)
        embedding2 = self.audio_to_embedding(audio_path2)
        similarity = torch.cosine_similarity(
            embedding1.unsqueeze(0), embedding2.unsqueeze(0)
        ).item()
        return similarity


def batch_cal_similarity(batch):
    wave_path1, wave_path2 = batch
    if wave_path1 is None or wave_path2 is None:
        return wave_path1, wave_path2, 0
    cs = CalSimilarity()
    similarity = cs.cal_similarity(wave_path1, wave_path2)
    return wave_path1, wave_path2, similarity


def analyze_similarity():
    # refs = glob.glob(r"/99_TemporaryData/haochen75/vc_demo/ref_audio/*.wav")
    refs = glob.glob(r"/99_TemporaryData/haochen75/vc_demo/ref_audio_clean/*.wav")
    # seed_vc = glob.glob(r"/99_TemporaryData/haochen75/vc_demo/seed_vc/*.wav")
    # seed_vc = glob.glob(r"/99_TemporaryData/haochen75/vc_demo/online_vc/*.wav")
    # seed_vc = glob.glob(r"/99_TemporaryData/haochen75/vc_demo/2.0/seed_vc_fine/*.wav")
    # seed_vc = glob.glob(r"/99_TemporaryData/haochen75/vc_demo/seed_vc_fine_f0/*.wav")
    # seed_vc = glob.glob(r"/99_TemporaryData/haochen75/vc_demo/2.0/seed_vc_fine_story_clean_double/*.wav")
    seed_vc = glob.glob(r"/99_TemporaryData/haochen75/vc_demo/3.0_clean/seed_vc_fine_story_clean_double/*.wav")

    src_audio = glob.glob(r"/99_TemporaryData/haochen75/vc_demo/src_audio/*.mp3")
    ids_list = []

    for i in range(len(src_audio)):
        path = Path(src_audio[i])
        ids = path.name.split("_")[0]
        ids_list.append(ids)

    print(f"找到 {len(refs)} 个参考音频, 其中包含 {len(set(ids_list))} 个唯一ID")

    print(f"找到 {len(refs)} 个参考音频和 {len(seed_vc)} 个源音频")

    batch = []
    for i in range(len(refs)):
        refs[i] = Path(refs[i])
        file_name = refs[i].name
        # print(f"处理参考音频: {file_name}")
        for id in ids_list:
            find_file = None
            for j in range(len(seed_vc)):
                seed_vc[j] = Path(seed_vc[j])
                # print(f"处理源音频: {src[j].name}")
                if (
                    file_name.split(".")[0].split("_")[0] in seed_vc[j].name
                    and id in seed_vc[j].name
                ):
                    find_file = seed_vc[j]
                    print(f"匹配到源音频: {seed_vc[j].name}")
            batch.append((refs[i], find_file))

    print(f"处理完成，共 {len(batch)} 个音频")
    # print(batch)

    import pandas as pd

    csv_path = "similarity/seed_vc_fine_f0.csv"

    with open(csv_path, "w") as f:
        writer = pd.DataFrame(columns=["ref", "src", "similarity"])
        data = []
        for b in batch:
            if len(b) != 2:
                print(f"跳过无效批次: {b}")
                data.append(
                    {"ref": result[0].name, "src": result[1].name, "similarity": 0}
                )
                continue
            result = batch_cal_similarity(b)
            data.append(
                {
                    "ref": result[0].name if result[0] else None,
                    "src": result[1].name if result[1] else None,
                    "similarity": result[2],
                }
            )
            print(
                f"处理完成: {result[0].name} 和 {result[1].name if result[1] else None} 相似度: {result[2]:.4f}"
            )
        writer = pd.DataFrame(data)
        writer.to_csv(csv_path, index=False)


if __name__ == "__main__":
    analyze_similarity()
