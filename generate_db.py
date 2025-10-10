from modules.openvoice.api import ToneColorConverter
from hf_utils import load_custom_model_from_hf
import torch
import os
import librosa
from modules.openvoice.api import ToneColorConverter

ckpt_converter, config_converter = load_custom_model_from_hf(
    "myshell-ai/OpenVoiceV2",
    "converter/checkpoint.pth",
    "converter/config.json",
)
tone_color_converter = ToneColorConverter(config_converter, device="cuda:0")
# 计算参数量, 单位是M
print(
    f"Number of parameters in ToneColorConverter: {sum(p.numel() for p in tone_color_converter.model.ref_enc.parameters() if p.requires_grad)/1e6}"
)
# 计算FLOPS


audios = []
lens = []
for root, dir, filenames in os.walk(
    "/99_TemporaryData/haochen75/seed_vc_training/ref_audio_train_val/train"
):
    for filename in filenames:
        if filename.endswith(".wav") or filename.endswith(".mp3"):
            audio_path = os.path.join(root, filename)
            print(audio_path)
            try:
                audio = librosa.load(audio_path)
            except:
                continue
            if audio:
                audio = librosa.load(audio_path)
                audio = torch.tensor(audio[0]).float()
                audios.append(audio)
                lens.append(len(audio))

feat = tone_color_converter.extract_se(audios, lens)
torch.save(feat, "se_db.pt")

feat1 = torch.load(
    "checkpoints/models--Plachta--Seed-VC/snapshots/257283f9f41585055e8f858fba4fd044e5caed6e/se_db_origin.pt",
    map_location="cuda:0",
    weights_only=True,
)
feat2 = torch.load("se_db.pt", map_location="cuda:0",weights_only=True)
feat = torch.cat([feat1, feat2], dim=0)
torch.save(feat, "se_db_final.pt")
