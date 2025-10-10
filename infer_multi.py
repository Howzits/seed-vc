import os
import numpy as np

os.environ["HF_HUB_CACHE"] = "./checkpoints/hf_cache"
import shutil
import warnings
import argparse
import torch
import yaml
import torch.multiprocessing as mp

warnings.simplefilter("ignore")

# load packages
import random

from modules.commons import *
import time

import torchaudio
import librosa
from modules.commons import str2bool

from hf_utils import load_custom_model_from_hf


# Load model and configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

fp16 = False

def load_models_for_process(args, device):
    """为每个进程加载模型"""
    global fp16
    fp16 = args.fp16
    if not args.f0_condition:
        if args.checkpoint is None:
            dit_checkpoint_path, dit_config_path = load_custom_model_from_hf(
                "Plachta/Seed-VC",
                "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
                "config_dit_mel_seed_uvit_whisper_small_wavenet.yml",
            )
        else:
            dit_checkpoint_path = args.checkpoint
            dit_config_path = args.config
        f0_fn = None
    else:
        if args.checkpoint is None:
            dit_checkpoint_path, dit_config_path = load_custom_model_from_hf(
                "Plachta/Seed-VC",
                "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth",
                "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml",
            )
        else:
            dit_checkpoint_path = args.checkpoint
            dit_config_path = args.config
        # f0 extractor
        from modules.rmvpe import RMVPE

        model_path = load_custom_model_from_hf(
            "lj1995/VoiceConversionWebUI", "rmvpe.pt", None
        )
        f0_extractor = RMVPE(model_path, is_half=False, device=device)
        f0_fn = f0_extractor.infer_from_audio

    config = yaml.safe_load(open(dit_config_path, "r"))
    model_params = recursive_munch(config["model_params"])
    model_params.dit_type = "DiT"
    model = build_model(model_params, stage="DiT")
    hop_length = config["preprocess_params"]["spect_params"]["hop_length"]
    sr = config["preprocess_params"]["sr"]

    # Load checkpoints
    model, _, _, _ = load_checkpoint(
        model,
        None,
        dit_checkpoint_path,
        load_only_params=True,
        ignore_modules=[],
        is_distributed=False,
    )
    for key in model:
        model[key].eval()
        model[key].to(device)
    model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

    # Load additional modules
    from modules.campplus.DTDNN import CAMPPlus

    campplus_ckpt_path = load_custom_model_from_hf(
        "funasr/campplus", "campplus_cn_common.bin", config_filename=None
    )
    campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
    campplus_model.eval()
    campplus_model.to(device)

    vocoder_type = model_params.vocoder.type

    if vocoder_type == "bigvgan":
        from modules.bigvgan import bigvgan

        bigvgan_name = model_params.vocoder.name

        bigvgan_model = bigvgan.BigVGAN.from_pretrained(
            bigvgan_name, use_cuda_kernel=False
        )
        # remove weight norm in the model and set to eval mode
        bigvgan_model.remove_weight_norm()
        bigvgan_model = bigvgan_model.eval().to(device)
        vocoder_fn = bigvgan_model
    elif vocoder_type == "hifigan":
        from modules.hifigan.generator import HiFTGenerator
        from modules.hifigan.f0_predictor import ConvRNNF0Predictor

        hift_config = yaml.safe_load(open("configs/hifigan.yml", "r"))
        hift_gen = HiFTGenerator(
            **hift_config["hift"],
            f0_predictor=ConvRNNF0Predictor(**hift_config["f0_predictor"]),
        )
        hift_path = load_custom_model_from_hf(
            "FunAudioLLM/CosyVoice-300M", "hift.pt", None
        )
        hift_gen.load_state_dict(torch.load(hift_path, map_location="cpu"))
        hift_gen.eval()
        hift_gen.to(device)
        vocoder_fn = hift_gen
    elif vocoder_type == "vocos":
        vocos_config = yaml.safe_load(open(model_params.vocoder.vocos.config, "r"))
        vocos_path = model_params.vocoder.vocos.path
        vocos_model_params = recursive_munch(vocos_config["model_params"])
        vocos = build_model(vocos_model_params, stage="mel_vocos")
        vocos_checkpoint_path = vocos_path
        vocos, _, _, _ = load_checkpoint(
            vocos,
            None,
            vocos_checkpoint_path,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        _ = [vocos[key].eval().to(device) for key in vocos]
        _ = [vocos[key].to(device) for key in vocos]
        total_params = sum(
            sum(p.numel() for p in vocos[key].parameters() if p.requires_grad)
            for key in vocos.keys()
        )
        print(f"Vocoder model total parameters: {total_params / 1_000_000:.2f}M")
        vocoder_fn = vocos.decoder
    else:
        raise ValueError(f"Unknown vocoder type: {vocoder_type}")

    speech_tokenizer_type = model_params.speech_tokenizer.type
    if speech_tokenizer_type == "whisper":
        # whisper
        from transformers import AutoFeatureExtractor, WhisperModel

        whisper_name = model_params.speech_tokenizer.name
        whisper_model = WhisperModel.from_pretrained(
            whisper_name, torch_dtype=torch.float16
        ).to(device)
        del whisper_model.decoder
        whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

        def semantic_fn(waves_16k):
            ori_inputs = whisper_feature_extractor(
                [waves_16k.squeeze(0).cpu().numpy()],
                return_tensors="pt",
                return_attention_mask=True,
            )
            ori_input_features = whisper_model._mask_input_features(
                ori_inputs.input_features, attention_mask=ori_inputs.attention_mask
            ).to(device)
            with torch.no_grad():
                ori_outputs = whisper_model.encoder(
                    ori_input_features.to(whisper_model.encoder.dtype),
                    head_mask=None,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
            S_ori = ori_outputs.last_hidden_state.to(torch.float32)
            S_ori = S_ori[:, : waves_16k.size(-1) // 320 + 1]
            return S_ori

    elif speech_tokenizer_type == "cnhubert":
        from transformers import (
            Wav2Vec2FeatureExtractor,
            HubertModel,
        )

        hubert_model_name = config["model_params"]["speech_tokenizer"]["name"]
        hubert_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            hubert_model_name
        )
        hubert_model = HubertModel.from_pretrained(hubert_model_name)
        hubert_model = hubert_model.to(device)
        hubert_model = hubert_model.eval()
        hubert_model = hubert_model.half()

        def semantic_fn(waves_16k):
            ori_waves_16k_input_list = [
                waves_16k[bib].cpu().numpy() for bib in range(len(waves_16k))
            ]
            ori_inputs = hubert_feature_extractor(
                ori_waves_16k_input_list,
                return_tensors="pt",
                return_attention_mask=True,
                padding=True,
                sampling_rate=16000,
            ).to(device)
            with torch.no_grad():
                ori_outputs = hubert_model(
                    ori_inputs.input_values.half(),
                )
            S_ori = ori_outputs.last_hidden_state.float()
            return S_ori

    elif speech_tokenizer_type == "xlsr":
        from transformers import (
            Wav2Vec2FeatureExtractor,
            Wav2Vec2Model,
        )

        model_name = config["model_params"]["speech_tokenizer"]["name"]
        output_layer = config["model_params"]["speech_tokenizer"]["output_layer"]
        wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        wav2vec_model = Wav2Vec2Model.from_pretrained(model_name)
        wav2vec_model.encoder.layers = wav2vec_model.encoder.layers[:output_layer]
        wav2vec_model = wav2vec_model.to(device)
        wav2vec_model = wav2vec_model.eval()
        wav2vec_model = wav2vec_model.half()

        def semantic_fn(waves_16k):
            ori_waves_16k_input_list = [
                waves_16k[bib].cpu().numpy() for bib in range(len(waves_16k))
            ]
            ori_inputs = wav2vec_feature_extractor(
                ori_waves_16k_input_list,
                return_tensors="pt",
                return_attention_mask=True,
                padding=True,
                sampling_rate=16000,
            ).to(device)
            with torch.no_grad():
                ori_outputs = wav2vec_model(
                    ori_inputs.input_values.half(),
                )
            S_ori = ori_outputs.last_hidden_state.float()
            return S_ori

    else:
        raise ValueError(f"Unknown speech tokenizer type: {speech_tokenizer_type}")
    # Generate mel spectrograms
    mel_fn_args = {
        "n_fft": config["preprocess_params"]["spect_params"]["n_fft"],
        "win_size": config["preprocess_params"]["spect_params"]["win_length"],
        "hop_size": config["preprocess_params"]["spect_params"]["hop_length"],
        "num_mels": config["preprocess_params"]["spect_params"]["n_mels"],
        "sampling_rate": sr,
        "fmin": config["preprocess_params"]["spect_params"].get("fmin", 0),
        "fmax": (
            None
            if config["preprocess_params"]["spect_params"].get("fmax", "None") == "None"
            else 8000
        ),
        "center": False,
    }
    from modules.audio import mel_spectrogram

    to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)

    return (
        model,
        semantic_fn,
        f0_fn,
        vocoder_fn,
        campplus_model,
        to_mel,
        mel_fn_args,
    )


def adjust_f0_semitones(f0_sequence, n_semitones):
    factor = 2 ** (n_semitones / 12)
    return f0_sequence * factor


def crossfade(chunk1, chunk2, overlap):
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
    if len(chunk2) < overlap:
        chunk2[:overlap] = (
            chunk2[:overlap] * fade_in[: len(chunk2)]
            + (chunk1[-overlap:] * fade_out)[: len(chunk2)]
        )
    else:
        chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
    return chunk2


def worker_process(rank, args, task_queue, result_queue, num_gpus):
    """工作进程函数"""
    # 设置当前进程使用的GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % num_gpus)
        device = torch.device(f"cuda:{rank % num_gpus}")
    else:
        device = torch.device("cpu")
    
    print(f"Worker {rank} started on device {device}")
    
    # 为当前进程加载模型
    model, semantic_fn, f0_fn, vocoder_fn, campplus_model, mel_fn, mel_fn_args = load_models_for_process(args, device)
    
    while True:
        try:
            # 从任务队列获取任务
            task = task_queue.get(timeout=10)  # 10秒超时
            if task is None:  # 结束信号
                break
                
            src_path, ref_path = task
            print(f"Worker {rank} processing: {os.path.basename(src_path)} with {os.path.basename(ref_path)}")

            src_name = os.path.basename(src_path).split(".")[0]
            ref_name = os.path.basename(ref_path).split(".")[0]
            output_filename = f"{src_name}&{ref_name}.wav"
            output_path = os.path.join(args.output, output_filename)
            if os.path.exists(output_path):
                print(f"Worker {rank} skipping existing file: {output_filename}")
                continue

            
            # 执行推理
            result = inference_single_pair(
                model, semantic_fn, f0_fn, vocoder_fn, campplus_model, mel_fn, mel_fn_args,
                args, src_path, ref_path, device
            )
            
            # 将结果放入结果队列
            result_queue.put((src_path, ref_path, result))
            
        except Exception as e:
            print(f"Worker {rank} error: {e}")
            continue
    
    print(f"Worker {rank} finished")


def safe_autocast_inference(model, condition, condition_lengths, mel2, style2, diffusion_steps, inference_cfg_rate, device, fp16):
    """安全的推理函数，处理autocast相关的梯度问题"""
    # 确保所有输入张量都在正确的设备上且具有梯度
    condition = condition.to(device)
    condition_lengths = condition_lengths.to(device)
    mel2 = mel2.to(device)
    style2 = style2.to(device)
    
    # 使用torch.no_grad()确保不计算梯度
    with torch.no_grad():
        if fp16:
            # 对于FP16，使用autocast但确保不保留计算图
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                vc_target = model.cfm.inference(
                    condition,
                    condition_lengths,
                    mel2,
                    style2,
                    None,
                    diffusion_steps,
                    inference_cfg_rate=inference_cfg_rate,
                )
        else:
            # 对于FP32，直接推理
            vc_target = model.cfm.inference(
                condition,
                condition_lengths,
                mel2,
                style2,
                None,
                diffusion_steps,
                inference_cfg_rate=inference_cfg_rate,
            )
    
    return vc_target


def safe_vocoder_inference(vocoder_fn, vc_target, device):
    """安全的声码器推理函数"""
    # 确保输入在正确的设备上
    vc_target = vc_target.to(device)
    
    # 使用torch.no_grad()确保不计算梯度
    with torch.no_grad():
        if hasattr(vocoder_fn, 'module'):
            # 如果是DataParallel包装的模型
            vc_wave = vocoder_fn.module(vc_target.float())
        else:
            vc_wave = vocoder_fn(vc_target.float())
    
    return vc_wave


def inference_single_pair(model, semantic_fn, f0_fn, vocoder_fn, campplus_model, mel_fn, mel_fn_args, 
                         args, src_path, ref_path, device):
    """对单个源-参考对进行推理"""
    sr = mel_fn_args["sampling_rate"]
    f0_condition = args.f0_condition
    auto_f0_adjust = args.auto_f0_adjust
    pitch_shift = args.semi_tone_shift
    diffusion_steps = args.diffusion_steps
    length_adjust = args.length_adjust
    inference_cfg_rate = args.inference_cfg_rate
    
    try:
        # 加载音频文件
        source_audio, _ = librosa.load(src_path, sr=sr)
        ref_audio, _ = librosa.load(ref_path, sr=sr)

        hop_length = 256
        max_context_window = sr // hop_length * 30
        overlap_frame_len = 16
        overlap_wave_len = overlap_frame_len * hop_length

        # Process audio - 使用detach()确保不保留计算图
        source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device).detach()
        ref_audio = torch.tensor(ref_audio[: sr * 25]).unsqueeze(0).float().to(device).detach()

        time_vc_start = time.time()
        
        # Resample - 使用detach()
        converted_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000).detach()
        
        # 语义提取 - 使用no_grad确保不计算梯度
        with torch.no_grad():
            if converted_waves_16k.size(-1) <= 16000 * 30:
                S_alt = semantic_fn(converted_waves_16k)
            else:
                overlapping_time = 5  # 5 seconds
                S_alt_list = []
                buffer = None
                traversed_time = 0
                while traversed_time < converted_waves_16k.size(-1):
                    if buffer is None:  # first chunk
                        chunk = converted_waves_16k[
                            :, traversed_time : traversed_time + 16000 * 30
                        ]
                    else:
                        chunk = torch.cat(
                            [
                                buffer,
                                converted_waves_16k[
                                    :,
                                    traversed_time : traversed_time
                                    + 16000 * (30 - overlapping_time),
                                ],
                            ],
                            dim=-1,
                        )
                    S_alt_chunk = semantic_fn(chunk)
                    if traversed_time == 0:
                        S_alt_list.append(S_alt_chunk)
                    else:
                        S_alt_list.append(S_alt_chunk[:, 50 * overlapping_time :])
                    buffer = chunk[:, -16000 * overlapping_time :]
                    traversed_time += (
                        30 * 16000
                        if traversed_time == 0
                        else chunk.size(-1) - 16000 * overlapping_time
                    )
                S_alt = torch.cat(S_alt_list, dim=1)

        ori_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000).detach()
        with torch.no_grad():
            S_ori = semantic_fn(ori_waves_16k)

        # 计算mel频谱 - 使用detach()
        mel = mel_fn(source_audio.to(device).float()).detach()
        mel2 = mel_fn(ref_audio.to(device).float()).detach()

        target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
        target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

        # 提取特征 - 使用detach()
        feat2 = torchaudio.compliance.kaldi.fbank(
            ori_waves_16k, num_mel_bins=80, dither=0, sample_frequency=16000
        ).detach()
        feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
        
        with torch.no_grad():
            style2 = campplus_model(feat2.unsqueeze(0))

        if f0_condition:
            with torch.no_grad():
                F0_ori = f0_fn(ori_waves_16k[0].cpu().numpy(), thred=0.03)
                F0_alt = f0_fn(converted_waves_16k[0].cpu().numpy(), thred=0.03)

            F0_ori = torch.from_numpy(F0_ori).to(device)[None].detach()
            F0_alt = torch.from_numpy(F0_alt).to(device)[None].detach()

            voiced_F0_ori = F0_ori[F0_ori > 1]
            voiced_F0_alt = F0_alt[F0_alt > 1]

            log_f0_alt = torch.log(F0_alt + 1e-5)
            voiced_log_f0_ori = torch.log(voiced_F0_ori + 1e-5)
            voiced_log_f0_alt = torch.log(voiced_F0_alt + 1e-5)
            median_log_f0_ori = torch.median(voiced_log_f0_ori)
            median_log_f0_alt = torch.median(voiced_log_f0_alt)

            # shift alt log f0 level to ori log f0 level
            shifted_log_f0_alt = log_f0_alt.clone()
            if auto_f0_adjust:
                shifted_log_f0_alt[F0_alt > 1] = (
                    log_f0_alt[F0_alt > 1] - median_log_f0_alt + median_log_f0_ori
                )
            shifted_f0_alt = torch.exp(shifted_log_f0_alt)
            if pitch_shift != 0:
                shifted_f0_alt[F0_alt > 1] = adjust_f0_semitones(
                    shifted_f0_alt[F0_alt > 1], pitch_shift
                )
        else:
            F0_ori = None
            F0_alt = None
            shifted_f0_alt = None

        # Length regulation - 使用no_grad
        with torch.no_grad():
            cond, _, codes, commitment_loss, codebook_loss = model.length_regulator(
                S_alt, ylens=target_lengths, n_quantizers=3, f0=shifted_f0_alt
            )
            prompt_condition, _, codes, commitment_loss, codebook_loss = model.length_regulator(
                S_ori, ylens=target2_lengths, n_quantizers=3, f0=F0_ori
            )

        max_source_window = max_context_window - mel2.size(2)
        # split source condition (cond) into chunks
        processed_frames = 0
        generated_wave_chunks = []
        
        # generate chunk by chunk and stream the output
        while processed_frames < cond.size(1):
            chunk_cond = cond[:, processed_frames : processed_frames + max_source_window]
            is_last_chunk = processed_frames + max_source_window >= cond.size(1)
            cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
            
            # 使用安全的推理函数
            vc_target = safe_autocast_inference(
                model, 
                cat_condition,
                torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                mel2,
                style2,
                diffusion_steps,
                inference_cfg_rate,
                device,
                fp16
            )
            vc_target = vc_target[:, :, mel2.size(-1) :]
            
            # 使用安全的声码器推理
            vc_wave = safe_vocoder_inference(vocoder_fn, vc_target.float(), device)
            vc_wave = vc_wave.squeeze()
            vc_wave = vc_wave[None, :]
            
            if processed_frames == 0:
                if is_last_chunk:
                    output_wave = vc_wave[0].cpu().numpy()
                    generated_wave_chunks.append(output_wave)
                    break
                output_wave = vc_wave[0, :-overlap_wave_len].cpu().numpy()
                generated_wave_chunks.append(output_wave)
                previous_chunk = vc_wave[0, -overlap_wave_len:]
                processed_frames += vc_target.size(2) - overlap_frame_len
            elif is_last_chunk:
                output_wave = crossfade(
                    previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len
                )
                generated_wave_chunks.append(output_wave)
                processed_frames += vc_target.size(2) - overlap_frame_len
                break
            else:
                output_wave = crossfade(
                    previous_chunk.cpu().numpy(),
                    vc_wave[0, :-overlap_wave_len].cpu().numpy(),
                    overlap_wave_len,
                )
                generated_wave_chunks.append(output_wave)
                previous_chunk = vc_wave[0, -overlap_wave_len:]
                processed_frames += vc_target.size(2) - overlap_frame_len
        
        vc_wave = torch.tensor(np.concatenate(generated_wave_chunks))[None, :].float()
        time_vc_end = time.time()
        print(f"RTF: {(time_vc_end - time_vc_start) / vc_wave.size(-1) * sr}")

        source_name = os.path.basename(src_path).split(".")[0]
        target_name = os.path.basename(ref_path).split(".")[0]
        os.makedirs(args.output, exist_ok=True)
        output_path = os.path.join(
            args.output,
            f"{source_name}&{target_name}.wav",
        )
        
        # 保存音频文件
        torchaudio.save(
            output_path,
            vc_wave.cpu(),
            sr,
        )
        
        return output_path  # 返回生成的音频路径
        
    except Exception as e:
        print(f"Error processing {src_path} with {ref_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def multi_gpu_inference(args):
    """多GPU推理主函数"""
    # 创建输出目录
    # if os.path.exists(args.output):
    #     shutil.rmtree(args.output)
    os.makedirs(args.output, exist_ok=True)
    
    # 收集所有任务
    tasks = []
    for root, dirs, src_files in os.walk(args.src_dir):
        for src in src_files:
            if src.endswith((".mp3", ".wav")):
                src_path = os.path.join(root, src)
                for ref_root, ref_dirs, ref_files in os.walk(args.ref_dir):
                    for ref in ref_files:
                        if ref.endswith((".mp3", ".wav")):
                            ref_path = os.path.join(ref_root, ref)
                            tasks.append((src_path, ref_path))
    
    print(f"Total tasks: {len(tasks)}")
    
    # 设置多进程
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    num_processes = min(num_gpus * 2, len(tasks))  # 每个GPU最多2个进程
    
    # 创建任务队列和结果队列
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    
    # 将任务放入队列
    for task in tasks:
        task_queue.put(task)
    
    # 添加结束信号
    for _ in range(num_processes):
        task_queue.put(None)
    
    # 启动工作进程
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=worker_process, args=(rank, args, task_queue, result_queue, num_gpus))
        p.start()
        processes.append(p)
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    # 收集结果
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    
    # 保存结果到CSV
    import pandas as pd
    cvs_path = os.path.join(args.output, "compare.csv")
    df_data = [{"src": src, "ref": ref, "output": output_path} 
               for src, ref, output_path in results if output_path is not None]
    df = pd.DataFrame(df_data)
    df.to_csv(cvs_path, index=False)
    
    print(f"Processing completed. Results saved to {cvs_path}")


# ... 其余代码保持不变 ...

def patch(args):
    """单进程模式（保持原有功能）"""
    
    model, semantic_fn, f0_fn, vocoder_fn, campplus_model, mel_fn, mel_fn_args = (
        load_models_for_process(args, device)
    )

    import pandas as pd

    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    os.makedirs(args.output, exist_ok=True)

    cvs_path = os.path.join(args.output, "cmpare.csv")

    if os.path.exists(cvs_path):
        os.remove(cvs_path)

    data = []
    for root, dirs, src_files in os.walk(
        "/99_TemporaryData/haochen75/seed_vc_training/story_audio_train_val/val_src_clean"
    ):
        for src in src_files:
            if src.endswith(".mp3") or src.endswith(".wav"):
                file_path = os.path.join(root, src)
                print(f"Processing source file: {file_path}")
                for ref_root, ref_dirs, ref_files in os.walk(
                    "/99_TemporaryData/haochen75/seed_vc_training/ref_audio_split_by_dir/val_mini_one_clean"
                ):
                    for ref in ref_files:
                        if ref.endswith(".mp3") or ref.endswith(".wav"):
                            print(f"Processing {file_path} with reference {ref}")
                            ref_path = os.path.join(ref_root, ref)
                            args.source = file_path
                            args.target = ref_path
                            # 使用单进程推理
                            result = inference_single_pair(
                                model,
                                semantic_fn,
                                f0_fn,
                                vocoder_fn,
                                campplus_model,
                                mel_fn,
                                mel_fn_args,
                                args,
                                file_path,
                                ref_path,
                                device,
                            )
                            data.append(
                                {"ref": ref_path, "src": file_path, "output": result}
                            )

    writer = pd.DataFrame(data)
    writer.to_csv(cvs_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, default="/99_TemporaryData/haochen75/seed_vc_training/story_audio_train_val/val_src_clean")
    parser.add_argument("--ref_dir", type=str, default="/99_TemporaryData/haochen75/seed_vc_training/ref_audio_split_by_dir/val_clean_1000")
    parser.add_argument("--output", type=str, default="/99_TemporaryData/haochen75/seed_vc_training/val_converted_audio1")
    parser.add_argument(
        "--src-dir", type=str, help="Source audio directory for batch processing"
    )
    parser.add_argument(
        "--ref-dir", type=str, help="Reference audio directory for batch processing"
    )
    parser.add_argument("--diffusion-steps", type=int, default=30)
    parser.add_argument("--length-adjust", type=float, default=1.0)
    parser.add_argument("--inference-cfg-rate", type=float, default=0.7)
    parser.add_argument("--f0-condition", type=str2bool, default=True)
    parser.add_argument("--auto-f0-adjust", type=str2bool, default=True)
    parser.add_argument("--semi-tone-shift", type=int, default=0)
    parser.add_argument(
        "--checkpoint", type=str, help="Path to the checkpoint file", default=None
    )
    parser.add_argument(
        "--config", type=str, help="Path to the config file", default=None
    )
    parser.add_argument("--fp16", type=str2bool, default=True)
    parser.add_argument(
        "--multi-gpu", type=str2bool, default=True, help="Enable multi-GPU processing"
    )
    args = parser.parse_args()

    # 设置多进程启动方法
    mp.set_start_method("spawn", force=True)
    args.output = "/99_TemporaryData/haochen75/seed_vc_training/val_converted_audio2"
    args.checkpoint = (
        "runs/run_dit_mel_seed_uvit_whisper_small_wavenet/my_run/ft_model.pth"
    )
    args.config = "runs/run_dit_mel_seed_uvit_whisper_small_wavenet/my_run/config_dit_mel_seed_uvit_whisper_small_wavenet.yml"

    if args.multi_gpu and args.src_dir and args.ref_dir:
        # 多GPU批量处理模式
        multi_gpu_inference(args)
    else:
        # 单文件处理模式（保持兼容性）
        if args.multi_gpu:
            print("Multi-GPU mode requires --src-dir and --ref-dir arguments")
        patch(args)
