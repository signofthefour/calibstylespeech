from __future__ import absolute_import, division, print_function, unicode_literals
import re

import argparse
from string import punctuation
import torch
import yaml
import numpy as np
import os
import json

import librosa
import pyworld as pw
import audio as Audio

from torch.utils.data import DataLoader
from g2p_en import G2p
from pypinyin import pinyin, Style

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from dataset import BatchInferenceDataset
from text import text_to_sequence
from g2p import text_to_phoneme
from text_normalization import processSent


import glob
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav, spectral_normalize_torch
from hifigan import Generator

SENTENCES = [
"Maika ơi, mở tin tức báo tuổi trẻ",
"Maika ơi, cho tôi nghe tin tức báo VnExpress",
"Maika ơi, mở thời sự đài VOH",

]

SENTENCES1 = [

"Ngày mười bốn tháng hai bạn có một việc cần làm. Việc một, đi học tiếng anh lúc bảy giờ sáng.",
"mời bạn nghe bài em của ngày hôm qua do sơn tùng m t p trình bày trên nhạc của tui.",
"mời bạn nghe bài bài ca tuổi trẻ do tamka pkl trình bày trên nhạc của tui.",
"mời bạn nghe bài em gì ơi do k i c m, jack trình bày trên nhạc của tui.",
"nhiệt độ thành phố hồ chí minh hiện tại là ba mươi bốn độ xê.",
"địa chỉ quán ăn nhật bản momo là chín mươi sáu nguyễn thị minh khai, phường một, quận ba, thành phố hồ chí minh.",
"Dạ!",
"Có em!",
"Em nghe đây ạ!",
"Mình có thể giúp gì cho bạn ạ!",
"Hãy nhớ là chiều nay ba giờ có lịch hẹn bạn nhé!",
"Anh hiểu rồi. Cặp bình để bơm bong bóng chứ gì?",
"Em bảo buổi sáng em phải ở nhà giặt đồ, nấu cơm, trông em, sao bây giờ anh lại gặp em ở đây?",
"Chứ còn anh? Sao anh lại đứng đây?",
"Thầy Tuấn dạy văn lớp tao. Thầy bảo đó là chuyện nhảm nhí, hoang đường, ai nhát gan mới sợ.",
"Biết chớ! Nhưng má tao la. Chỉ có anh Dự là thoải mái với tao thôi.",
"Hư này! Đã bảo bao nhiêu lần rồi mà không nghe! Có ngày té lọi cẳng cho coi!",
"Có thể bạn sẽ yêu trong cô đơn, không bao giờ mong người đó đáp lại tình cảm của mình. Nhưng ta có một người để yêu đã là hạnh phúc rồi.",
"Nước sạch, cải cách hành chính, quản lý trật tự xây dựng, quản lý đất đai là những vấn đề luôn được sự quan tâm giải quyết của Thường trực Thành uỷ và cá nhân tôi.",
"Thú thật là hè nào mình cũng coi Tây Du Ký hết!",
"Bạn nhìn chắc cũng biết rồi, nếu so với bạn Maika nhỏ bé lắm đó!",
"Với thân hình trước sau như một, trên dưới giống nhau thì mình làm gì có chân để mang giày, nếu có, mình chỉ có chân thành để hỗ trợ bạn mà thôi!",
"Hơi buồn vì bạn chưa biết mình là ai, mình là Maika, trợ lý cá nhân của bạn.",
"Ốc xào, ốc nướng, ốc hấp, món nào mình cũng mê hết!",
"Mình sợ nước lắm, vì nước có thể khiến thiết bị này bị hỏng, mà mình cũng không biết bơi nữa!",
"Câu này khó quá, trả đĩa bay để mình về với hành tinh của mình đi!",
"Có nhé, mình còn mơ thấy bạn trong lúc ngủ nữa cơ!",
"Mình cũng không biết nữa, nhưng nếu có thì bạn có biết câu thần chú nào giúp mình có ba vòng như siêu mẫu không, chỉ mình với!",
"Mình tin cái này bạn giỏi hơn mình đấy! Hãy lái thật an toàn nhé!",
"Nắng đã có nón, mưa đã có dù, lạnh, cảm cúm đã có bạn đây rồi. Không sao! Không sao!",
"Gì chứ! Lắng nghe là sở trường của mình đấy nhé!",
"Không! Mình không nóng đâu! Nhưng cập nhật tin tức nóng hổi là nghề của mình đấy!",
"Nếu như vậy thì trợ lý như mình biết làm việc cho ai bây giờ! Thật không thể tưởng tượng nổi!",
"Mộng mơ thì mới có thể làm thơ cho bạn nghe được chứ!",
"Chán bạn ghê! Biết rồi còn phải hỏi.",
"Biết làm sao bây giờ, vì mình sinh ra chỉ để làm trợ lý cho bạn mà thôi!",
"Khò khò! Mình cũng ngủ đây. Chúc ngủ ngon!",
"Không sao! Nếu bạn vẫn chưa biết thì mình xin tự giới thiệu, mình là trợ lý Maika của bạn đây.",
    ]


h = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def get_audio(preprocess_config, wav_path):

    hop_length = preprocess_config["preprocessing"]["stft"]["hop_length"]
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    STFT = Audio.stft.TacotronSTFT(
        preprocess_config["preprocessing"]["stft"]["filter_length"],
        hop_length,
        preprocess_config["preprocessing"]["stft"]["win_length"],
        preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        sampling_rate,
        preprocess_config["preprocessing"]["mel"]["mel_fmin"],
        preprocess_config["preprocessing"]["mel"]["mel_fmax"],
    )
    with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    ) as f:
        stats = json.load(f)
        stats = stats["pitch"][2:] + stats["energy"][2:]
        pitch_mean, pitch_std, energy_mean, energy_std = stats

    # Read and trim wav files
    wav, _ = librosa.load(wav_path)

    # Compute fundamental frequency
    pitch, t = pw.dio(
        wav.astype(np.float64),
        sampling_rate,
        frame_period=hop_length / sampling_rate * 1000,
    )
    pitch = pw.stonemask(wav.astype(np.float64), pitch, t, sampling_rate)

    # Compute mel-scale spectrogram and energy
    mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav.astype(np.float32), STFT)

    # Normalize Variance
    pitch = (pitch - pitch_mean) / pitch_std
    energy = (energy - energy_mean) / energy_std

    mels = mel_spectrogram.T[None]
    mel_lens = np.array([len(mels[0])])

    mel_spectrogram = mel_spectrogram.astype(np.float32)
    energy = energy.astype(np.float32)

    return mels, mel_lens, (mel_spectrogram, pitch, energy)


def preprocess_english(text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)

def preprocess_vn(text):
    phonemes = text_to_phoneme(processSent(text))
    phonemes = phonemes.replace('-', ' ')
    print("Raw text: {}\nPhoneme: {}".format(text, phonemes))
    sequence = np.array(
        text_to_sequence(phonemes, 'basic_cleaner')        
    )
    print("Sequence: {}".format(sequence))
    return sequence


def preprocess_mandarin(text, preprocess_config):
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    phones = []
    pinyins = [
        p[0]
        for p in pinyin(
            text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
        )
    ]
    for p in pinyins:
        if p in lexicon:
            phones += lexicon[p]
        else:
            phones.append("sp")

    phones = "{" + " ".join(phones) + "}"
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def synthesize(model, step, configs, vocoder, batchs, control_values):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    #print("d_control: {}".format(duration_control))
    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model(
                *(batch[2:-1]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["out"],
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        required=True,
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--ref_audio",
        type=str,
        default="/data/tuong/Yen/StyleSpeech/ref/",
        help="reference audio path to extract the speech style, for single-sentence mode only",
    )
    parser.add_argument(
        "--speaker_id",
        type=str,
        default=39,
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    parser.add_argument('--vocoder', 
        default="/data/tuong/Yen/hifi-gan/cp_hifigan/g_00320000")
    args = parser.parse_args()

    # Check source texts
    if args.mode == "batch":
        assert args.source is not None and args.text is None
    #if args.mode == "single":
    #    assert args.source is None and args.text is not None

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    print("Loading model StyleSpeech...")
    model = get_model(args, configs, device, train=False)
    print("Done")
    # Load vocoder
    print("Loading vocoder: {}".format(args.vocoder))
    config_file = os.path.join('/data/tuong/Yen/hifi-gan/cp_hifigan/config.json')
    with open(config_file) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    vocoder = Generator(h)
    state_dict_g = torch.load(args.vocoder, map_location=device) 
    vocoder.load_state_dict(state_dict_g["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()
    vocoder.to(device)
    print("Done")
    print("Synthesizing in {} mode".format(args.mode))
    # Preprocess texts
    if args.mode == "batch":
        # Get dataset
        dataset = BatchInferenceDataset(args.source, preprocess_config)
        batchs = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=dataset.collate_fn,
        )

    #for ref in os.listdir(args.ref_audio):
    #    print(ref)

    if args.mode == "single":
        for audio in os.listdir(args.ref_audio):
            ref_audio = os.path.join(args.ref_audio, audio)
            raw_texts = []
            #for textfile in os.listdir(args.text):
            #    with open(os.path.join(args.text, textfile), 'r') as f:
            #        raw_texts += [line for line in f.readlines()]
            mels, mel_lens, ref_info = get_audio(preprocess_config, ref_audio)
            for idx, ids in enumerate(SENTENCES):
                with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json")) as f:
                    speaker_map = json.load(f)
                #speakers = np.array([speaker_map[args.speaker_id]])
                #speakers = speakers.cuda()
                if audio == "ref_Anh":
                    speakers = np.array([1056])
                else: speakers = np.array([1557])
                texts = np.array([preprocess_vn(ids)])
                text_lens = np.array([len(texts[0])])
                ref_name = os.path.basename(ref_audio).strip(".wav")
                result_folder_dir = os.path.join(configs[2]["path"]["result_path"], ref_name)
                if not os.path.exists(result_folder_dir):
                    os.makedirs(result_folder_dir)
                configs[2]["path"]["out"] = result_folder_dir
                batchs = [(["_".join([os.path.basename(ref_audio).strip(".wav"), args.speaker_id, str(idx)]) for id in ids], \
                    raw_texts, speakers, texts, text_lens, max(text_lens), mels, mel_lens, max(mel_lens), [ref_info])]

                control_values = args.pitch_control, args.energy_control, args.duration_control

                synthesize(model, args.restore_step, configs, vocoder, batchs, control_values)
