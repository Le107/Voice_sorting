import os
import shutil
import requests
import torch
import numpy as np
import subprocess
import onnxruntime as ort
import pygame
import time

# FFmpeg setup
current_dir = os.path.dirname(os.path.abspath(__file__))
# Добавляем текущую директорию в PATH, чтобы subprocess видел ffmpeg.exe
os.environ["PATH"] += os.pathsep + current_dir

# Настройки
MODEL_URL = "https://huggingface.co/MCplayer/voxblink2_samresnet100_ft/resolve/main/models/voxblink2_samresnet100_ft/avg_model.onnx?download=true"
MODEL_PATH = "voxblink2_model.onnx"
INPUT_DIR = "input"
OUTPUT_DIR = "sorted"
THRESHOLD = 0.60 

def load_audio_with_ffmpeg(file_path, target_sr=16000):
    """Заменяет torchaudio.load. Использует ffmpeg.exe для декодирования."""
    ffmpeg_exe = os.path.join(current_dir, "ffmpeg.exe")
    if not os.path.exists(ffmpeg_exe):
        ffmpeg_exe = "ffmpeg" # если нет в папке, пробуем системный

    command = [
        ffmpeg_exe,
        '-v', 'error',
        '-i', str(file_path),
        '-f', 's16le',
        '-acodec', 'pcm_s16le',
        '-ar', str(target_sr),
        '-ac', '1',
        '-'
    ]
    
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()
        if process.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {err.decode()}")
        
        # Превращаем байты в тензор
        waveform = np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0
        return torch.from_numpy(waveform).unsqueeze(0) # добавляем размерность канала [1, T]
    except Exception as e:
        print(f"Ошибка декодирования {file_path}: {e}")
        return None

def get_fbank(wav):
    import torchaudio.compliance.kaldi as kaldi 
    return kaldi.fbank(wav, num_mel_bins=80, frame_length=25, frame_shift=10, 
                        energy_floor=0.0, sample_frequency=16000)

def apply_simple_dereverb(wav):
    wav = wav - wav.mean()
    peak = torch.max(torch.abs(wav))
    if peak > 0:
        wav[torch.abs(wav) < (peak * 0.02)] = 0
    return wav

def play_audio(file_path, label=""):
    try:
        if label: print(f"--- {label}: {os.path.basename(file_path)} ---")
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        pygame.mixer.quit()
    except Exception as e:
        print(f"Ошибка звука: {e}")

def get_embedding(session, audio_path):
    # Загружаем через нашу новую функцию на FFmpeg
    wav = load_audio_with_ffmpeg(audio_path)
    if wav is None: return None
    
    wav = apply_simple_dereverb(wav)
    
    # Мы оставляем импорт torchaudio только для kaldi.fbank, 
    # так как эта часть кода весит мало. Главный "жир" torchaudio — в декодерах.
    import torchaudio.compliance.kaldi as kaldi
    feats = kaldi.fbank(wav, num_mel_bins=80, frame_length=25, frame_shift=10, 
                        energy_floor=0.0, sample_frequency=16000)
    
    feats = (feats - feats.mean(dim=0)).unsqueeze(0).numpy()
    input_name = session.get_inputs()[0].name
    out = session.run(None, {input_name: feats})
    return torch.from_numpy(out[0])

def main():
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    os.makedirs(INPUT_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print("Загрузка модели..."); r = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192): f.write(chunk)

    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    known_speakers = {} 
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.wav', '.ogg', '.mp3', '.flac'))])
    
    if not files:
        print("Папка input пуста.")
        return

    print("\n" + "!"*60 + f"\nФАЙЛОВ: {len(files)} | ПОРОГ: {THRESHOLD}\n" + "!"*60)

    for file_name in files:
        file_path = os.path.join(INPUT_DIR, file_name)
        print(f"\n\n>>> АНАЛИЗ: {file_name}\n" + "*" * 30)
        
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
                    
                    print(f"Сходство с '{group_name}': среднее {avg_score:.4f} (пик {max_in_group:.4f})")
                    
                    if avg_score > best_score:
                        best_score = avg_score
                        best_group = group_name
                        best_sample_path = data["paths"][group_scores.index(max_in_group)]

            if best_score >= THRESHOLD:
                print(f"РЕЗУЛЬТАТ: Авто-подтверждение '{best_group}' ({best_score:.4f})")
                target = best_group
            else:
                if known_speakers:
                    print(f"РЕЗУЛЬТАТ: Сомнение (Score {best_score:.4f} < {THRESHOLD})")
                
                play_audio(file_path, "ТЕКУЩИЙ")
                if best_sample_path:
                    play_audio(best_sample_path, f"ОБРАЗЕЦ ИЗ '{best_group}'")

                ans = input(f"Кто это? (Enter='{best_group}', 's'=skip, ИМЯ): ").strip()
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
