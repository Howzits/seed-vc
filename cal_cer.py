# import whisper
import jiwer
import editdistance
import os
import glob
import string
import pandas as pd


def get_file_list(path):
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


def get_all_audio_files(refs, vc_path1, vc_path2):
    id_path = get_file_list
    file_list = []
    for ref in get_file_list(refs):

        ref_name = os.path.basename(ref).rsplit(".", 1)[0].split("_")[0]

        vc1 = None
        vc2 = None
        for vc_path in get_file_list(vc_path1):
            if ref_name in vc_path:
                vc1 = vc_path

        for vc_path in get_file_list(vc_path2):
            if ref_name in vc_path:
                vc2 = vc_path

        file_list.append((ref, vc1, vc2))
    return file_list


# 计算 WER (Word Error Rate)
def calculate_wer(reference, hypothesis):
    wer = jiwer.wer(reference, hypothesis)
    return wer


# 计算 CER (Character Error Rate)
def calculate_cer(reference, hypothesis):
    cer = editdistance.eval(reference, hypothesis) / len(reference)
    return cer


# 删除所有的标点符号。
def remove_punctuation(text):
    # 英文标点
    english_punct = string.punctuation
    # 中文标点
    chinese_punct = (
        "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—''‛"
        "„‟…‧﹏"
    )

    # 创建翻译表
    punct = english_punct + chinese_punct
    trans_table = str.maketrans("", "", punct)

    # 删除所有标点
    return text.translate(trans_table)


def batch_cal_cer(src_path, tgt_path):
    df = pd.DataFrame(columns=["src", "tgt", "WER", "CER"])

    src_files = glob.glob(os.path.join(src_path, "*.txt"))
    tgt_files = glob.glob(os.path.join(tgt_path, "*.txt"))
    ref_files = glob.glob(
        os.path.join("/99_TemporaryData/haochen75/vc_demo/ref_audio", "*.wav")
    )
    story_ids = []
    ref_ids = []
    for src_file in src_files:
        src_file_name = src_file.split("/")[-1].split(".")[0]
        story_id = src_file_name.split("_")[0]
        story_ids.append(story_id)

    for ref_id in ref_files:
        ref_file_name = ref_id.split("/")[-1].split(".")[0]
        ref_id = ref_file_name.split("_")[0]
        ref_ids.append(ref_id)

    # print(story_ids, ref_ids)

    temp = []

    for s_id in story_ids:
        for r_id in ref_ids:
            g_src_file = ""
            for src_file in src_files:
                if s_id in src_file:
                    g_src_file = src_file
                    with open(src_file, "r") as f:
                        src_text = f.read()

            for tgt_file in tgt_files:
                if s_id in tgt_file and r_id in tgt_file:
                    # print(s_id, r_id, tgt_file)
                    with open(tgt_file, "r") as f:
                        tgt_text = f.read()
                        if len(tgt_text) > 0 and len(src_text) > 0:
                            wer = calculate_wer(src_text, tgt_text)
                            cer = calculate_cer(src_text, tgt_text)
                        else:
                            wer = 1
                            cer = 1
                            # print(remove_punctuation(src_text))
                        src_file_name = os.path.basename(g_src_file)
                        tgt_file_name = os.path.basename(tgt_file)
                        temp.append(
                            {
                                "src": src_file_name,
                                "tgt": tgt_file_name,
                                "WER": wer,
                                "CER": cer,
                            }
                        )

                        print(
                            f"{src_file_name,tgt_file_name}: WER={wer:.4f}, CER={cer:.4f}"
                        )
    df = pd.DataFrame(temp)
    df.to_csv("cer/fine_cer_results.csv", index=False)


if __name__ == "__main__":

    ref_path = r"/99_TemporaryData/haochen75/vc_demo/ref_audio/"
    vc_path1 = r"/99_TemporaryData/haochen75/vc_demo/online_vc/"
    vc_path2 = r"/99_TemporaryData/haochen75/vc_demo/seed_vc/"
    # src_path = r"/99_TemporaryData/haochen75/vc_demo/src_audio"
    src_path = r"/99_TemporaryData/haochen75/vc_demo/src_separated/vocals"
    vc_path_f0 = r"/99_TemporaryData/haochen75/vc_demo/2.0_clean/seed_vc_fine_story_clean"
    vc_path_fine = r"/99_TemporaryData/haochen75/vc_demo/2.0/seed_vc_fine"
    vc_path_clean = r"/99_TemporaryData/haochen75/vc_demo/3.0_clean/seed_vc_fine_story_clean_double"

    batch_cal_cer(src_path, vc_path_clean)
