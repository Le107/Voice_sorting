import os
import shutil
import requests
import torch
import numpy as np
import subprocess
import onnxruntime as ort
import pygame
import time
from pydub import AudioSegment

# --- Настройки путей и окружения ---
current_dir = os.path.dirname(os.path.abspath(__file__))
torch.hub.set_dir(current_dir)

os.environ["PATH"] += os.pathsep + current_dir
ffmpeg_exe = os.path.join(current_dir, "ffmpeg.exe")
AudioSegment.converter = ffmpeg_exe

MODEL_URL = "https://huggingface.co/MCplayer/voxblink2_samresnet100_ft/resolve/main/models/voxblink2_samresnet100_ft/avg_model.onnx?download=true"
VOX_MODEL_PATH = "voxblink2_model.onnx"
WORK_DIR = "work"
INPUT_DIR = "input"
OUTPUT_DIR = "sorted"
THRESHOLD = 0.60 

# Загрузка нейросети Silero VAD
print("Загрузка модели VAD...")
vad_model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    trust_repo=True
)
(get_speech_timestamps, _, _, _, _) = utils

def load_for_vad(path):
    audio = AudioSegment.from_file(path).set_frame_rate(16000).set_channels(1)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
    return torch.from_numpy(samples)

def slice_with_vad():
    if os.path.exists(INPUT_DIR):
        shutil.rmtree(INPUT_DIR)
    os.makedirs(INPUT_DIR)
    
    files = [f for f in os.listdir(WORK_DIR) if f.lower().endswith(('.mp3', '.wav', '.flac', '.m4a'))]
    if not files:
        print("В папке work нет файлов для нарезки.")
        return False

    for filename in files:
        print(f"VAD анализ: {filename}...")
        path = os.path.join(WORK_DIR, filename)
        name_part, ext = os.path.splitext(filename)
        
        try:
            wav = load_for_vad(path)
            speech_timestamps = get_speech_timestamps(
                wav, vad_model, sampling_rate=16000,
                threshold=0.6,              # ПОВЫСИТЬ (был 0.4). Чем выше, тем строже ищет речь.
                min_silence_duration_ms=10, # Минимальная пауза
                speech_pad_ms=0,            # Без отступов
                window_size_samples=512     # МЕНЬШЕ ОКНО (по умолчанию 512 или 1024). Увеличивает точность границ.
            )

            if not speech_timestamps:
                continue

            full_audio = AudioSegment.from_file(path)
            for i, ts in enumerate(speech_timestamps, start=1):
                start_ms, end_ms = (ts['start'] / 16000) * 1000, (ts['end'] / 16000) * 1000
                chunk = full_audio[start_ms:end_ms]
                output_filename = f"{name_part}_{i:03d}{ext}"
                chunk.export(os.path.join(INPUT_DIR, output_filename), format=ext.replace(".", ""))
                print(f"  Нарезано: {output_filename}")
        except Exception as e:
            print(f"  Ошибка при обработке {filename}: {e}")
            
    return True

def load_audio_with_ffmpeg(file_path, target_sr=16000):
    command = [ffmpeg_exe if os.path.exists(ffmpeg_exe) else "ffmpeg",
               '-v', 'error', '-i', str(file_path), '-f', 's16le', '-acodec', 'pcm_s16le',
               '-ar', str(target_sr), '-ac', '1', '-']
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, _ = process.communicate()
        waveform = np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0
        return torch.from_numpy(waveform).unsqueeze(0)
    except: return None

def get_embedding(session, audio_path):
    wav = load_audio_with_ffmpeg(audio_path)
    if wav is None: return None
    import torchaudio.compliance.kaldi as kaldi
    feats = kaldi.fbank(wav, num_mel_bins=80, frame_length=25, frame_shift=10, energy_floor=0.0, sample_frequency=16000)
    feats = (feats - feats.mean(dim=0)).unsqueeze(0).numpy()
    
    # ИСПРАВЛЕНО: берем первый элемент из списка результатов
    input_name = session.get_inputs()[0].name
    out = session.run(None, {input_name: feats})
    return torch.from_numpy(out[0])

def play_audio(file_path, label=""):
    try:
        print(f"--- {label}: {os.path.basename(file_path)} ---")
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy(): time.sleep(0.1)
        pygame.mixer.quit()
    except: pass

def main():
    for d in [WORK_DIR, INPUT_DIR, OUTPUT_DIR]: os.makedirs(d, exist_ok=True)
    
    work_files = [f for f in os.listdir(WORK_DIR) if f.lower().endswith(('.mp3', '.wav', '.flac', '.m4a'))]
    input_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.mp3', '.wav', '.flac', '.m4a'))]

    run_slicing = False
    if work_files and input_files:
        print(f"\nФайлы есть и в 'work' ({len(work_files)}), и 'input' ({len(input_files)})")
        choice = input("1. Нарезать заново и отсортировать, папка 'input' очистится\n2. Только отсортировать файлы в 'input'\nВыбор (1, 2): ").strip()
        if choice == "1": run_slicing = True
    elif work_files:
        run_slicing = True
    elif input_files:
        print("Папка 'work' пуста. Переходим сразу к сортировке 'input'...")
    else:
        print("Нет файлов ни в 'work', ни в 'input'.")
        return

    if run_slicing:
        if not slice_with_vad(): return

    # Подготовка модели и сортировка
    if not os.path.exists(VOX_MODEL_PATH):
        print("Загрузка модели VoxBlink..."); r = requests.get(MODEL_URL, stream=True)
        with open(VOX_MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192): f.write(chunk)

    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    
    session = ort.InferenceSession(VOX_MODEL_PATH, providers=['CPUExecutionProvider'])
    known_speakers = {} 
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.wav', '.ogg', '.mp3', '.flac'))])
    
    if not files:
        print("Папка input пуста.")
        return

    print("\n" + "!"*60 + f"\nФАЙЛОВ: {len(files)} | ПОРОГ: {THRESHOLD}\n" + "!"*60)

    total_files = len(files)

    for f_idx, file_name in enumerate(files, 1):
        file_path = os.path.join(INPUT_DIR, file_name)
        
        try:
            emb = get_embedding(session, file_path)
            if emb is None: continue

            best_group = None
            best_score = -1.0
            best_sample_path = None
            
            if known_speakers:
                for group_name, data in known_speakers.items():
                    group_scores = [torch.nn.functional.cosine_similarity(emb, e, dim=-1).item() for e in data["embs"]]
                    max_in_group = max(group_scores)
                    avg_score = sum(group_scores) / len(group_scores)
                    
                    if avg_score > best_score:
                        best_score = avg_score
                        best_group = group_name
                        best_sample_path = data["paths"][group_scores.index(max_in_group)]

            if best_score >= THRESHOLD:
                print(f"[{f_idx}/{total_files}] АВТО: {file_name} -> '{best_group}' ({best_score:.4f})")
                target = best_group
            else:
                print(f"\n" + "="*60)
                print(f"ПРОГРЕСС: [{f_idx}/{total_files}] | ФАЙЛ: {file_name}")
                print(f"СХОДСТВО: {best_score:.4f} | ПОРОГ: {THRESHOLD}")
                print("="*60)
                
                if best_sample_path:
                    play_audio(best_sample_path, f"ОБРАЗЕЦ ИЗ '{best_group}'")
                play_audio(file_path, "ТЕКУЩИЙ")

                while True:
                    ans = input(f"Кто это? (Enter='{best_group}', 's'=skip, 'r'=replay, ИМЯ): ").strip()
                    if ans.lower() == 'r':
                        if best_sample_path:
                            play_audio(best_sample_path, f"ОБРАЗЕЦ ИЗ '{best_group}'")
                        play_audio(file_path, "ТЕКУЩИЙ")
                        continue
                    break
                
                if ans.lower() == 's': continue
                target = ans if ans else best_group

            if target:
                if target not in known_speakers:
                    known_speakers[target] = {"embs": [], "paths": []}
                known_speakers[target]["embs"].append(emb)
                known_speakers[target]["paths"].append(file_path)
                
                out_p = os.path.join(OUTPUT_DIR, target)
                os.makedirs(out_p, exist_ok=True)
                shutil.copy(file_path, os.path.join(out_p, file_name))

        except Exception as e:
            print(f"Ошибка на {file_name}: {e}")

    print("\n" + "="*60 + "\nСОРТИРОВКА ЗАВЕРШЕНА\n" + "="*50)


if __name__ == "__main__":
    main()
    