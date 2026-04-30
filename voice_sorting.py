import os
import shutil
import librosa
import numpy as np
import pygame
import time
import warnings
import torch
import subprocess
from pydub import AudioSegment
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# --- НАСТРОЙКИ ---
GLOBAL_DOUBT = 0.70   
MIN_SAMPLES_FOR_AUTO = 3 
AUTO_MARGIN = 0.03    

current_dir = os.path.dirname(os.path.abspath(__file__))
torch.hub.set_dir(current_dir)
os.environ["PATH"] += os.pathsep + current_dir
ffmpeg_exe = os.path.join(current_dir, "ffmpeg.exe")
AudioSegment.converter = ffmpeg_exe

WORK_DIR = os.path.join(current_dir, "work")
INPUT_DIR = os.path.join(current_dir, "input")
OUTPUT_DIR = os.path.join(current_dir, "sorted")

# --- ЗАГРУЗКА VAD ---
print("Загрузка модели VAD...")
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, trust_repo=True)
(get_speech_timestamps, _, _, _, _) = utils

def load_for_vad(path):
    audio = AudioSegment.from_file(path).set_frame_rate(16000).set_channels(1)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
    return torch.from_numpy(samples)

def slice_with_vad():
    if os.path.exists(INPUT_DIR): shutil.rmtree(INPUT_DIR)
    os.makedirs(INPUT_DIR)
    files = [f for f in os.listdir(WORK_DIR) if f.lower().endswith(('.mp3', '.wav', '.flac', '.m4a'))]
    if not files: return False
    for filename in files:
        print(f"VAD анализ: {filename}...")
        path = os.path.join(WORK_DIR, filename); name_part, ext = os.path.splitext(filename)
        try:
            wav = load_for_vad(path)
            speech_timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=16000, threshold=0.6, min_silence_duration_ms=10, speech_pad_ms=0, window_size_samples=512)
            full_audio = AudioSegment.from_file(path)
            for i, ts in enumerate(speech_timestamps, start=1):
                start_ms, end_ms = (ts['start'] / 16000) * 1000, (ts['end'] / 16000) * 1000
                if (end_ms - start_ms) < 500: continue
                chunk = full_audio[start_ms:end_ms]
                output_filename = f"{name_part}_{i:03d}{ext}"
                chunk.export(os.path.join(INPUT_DIR, output_filename), format=ext.replace(".", ""))
                print(f"  Нарезано: {output_filename}")
        except: pass
    return True

def play_audio(path, label=""):
    if path and os.path.exists(path):
        if label: print(f">> {label}")
        pygame.mixer.init()
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy(): pygame.time.Clock().tick(10)
        pygame.mixer.quit()

def get_waveform_similarity(y1, y2, sr=16000):
    win_len = int(0.5 * sr); ml = min(len(y1), len(y2))
    if ml < win_len: return 0.0
    scores = []
    for i in range(0, ml - win_len, win_len):
        seg1 = y1[i : i + win_len]
        search_radius = int(0.2 * sr); start = max(0, i - search_radius); end = min(ml - win_len, i + search_radius)
        best_seg_score = 0
        for j in range(start, end, int(sr * 0.05)):
            seg2 = y2[j : j + win_len]; corr = np.corrcoef(seg1, seg2)
            if not np.any(np.isnan(corr)):
                val = corr[0,1] if isinstance(corr, np.ndarray) else 0
                best_seg_score = max(best_seg_score, val)
        scores.append(best_seg_score)
    return np.mean(scores) if scores else 0.0

def get_features(path):
    try:
        y, sr = librosa.load(path, sr=16000, duration=10)
        y, _ = librosa.effects.trim(y, top_db=25); y = librosa.util.normalize(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mels=64, n_mfcc=40)
        delta = librosa.feature.delta(mfcc)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        return {
            "mfcc": np.hstack([np.mean(mfcc, axis=1), np.mean(delta, axis=1), np.mean(contrast, axis=1)]).reshape(1, -1),
            "chroma": np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1).reshape(1, -1),
            "spec": np.mean(librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max), axis=1).reshape(1, -1),
            "raw_y": y
        }
    except: return None

def handle_choice(path, f_name, spk, scores, group_avg, current_sure, idx, total_groups, f_idx, total_files, sample_p):
    s, c, sp, w = scores; current_final = (s * 0.80) + (c * 0.10) + (sp * 0.05) + (w * 0.05)
    print(f"\n" + "="*80)
    # Теперь f_name выводится здесь
    print(f"ПРОГРЕСС: [{f_idx}/{total_files}] | ФАЙЛ: {f_name}")
    print(f"ГРУППА: [{spk}] ({idx+1}/{total_groups})")
    print(f"СХОДСТВО: {current_final:.4f} | СРЕДНЕЕ: {group_avg:.4f} | ПОРОГ: {current_sure:.4f}")
    print("="*80)
    
    while True:
        sample_filename = os.path.basename(sample_p)
        play_audio(sample_p, f"ОБРАЗЕЦ '{spk}': {sample_filename}")
        time.sleep(0.2)
        play_audio(path, f"ТЕКУЩИЙ: {f_name}")
        
        ans = input(f"Кто это? (Enter='{spk}', 'n'=next, 'r'=replay, 's'=skip, ИМЯ): ").strip()
        if ans.lower() == 'r': continue
        if ans == '': return 'y'
        return ans

def main():
    for d in [WORK_DIR, INPUT_DIR, OUTPUT_DIR]: os.makedirs(d, exist_ok=True)
    work_files = [f for f in os.listdir(WORK_DIR) if f.lower().endswith(('.mp3', '.wav', '.m4a'))]
    input_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.mp3', '.wav', '.m4a'))]
    
    run_slicing = False
    if work_files and input_files:
        print(f"\nФайлы есть в 'work' ({len(work_files)}) и 'input' ({len(input_files)})")
        choice = input("1. Нарезать заново и отсортировать, папка 'input' очистится\n2. Только отсортировать файлы в 'input'\nВыбор (1, 2): ").strip()
        if choice == "1": run_slicing = True
    elif work_files:
        run_slicing = True
    elif input_files:
        print("Папка 'work' пуста. Переходим сразу к сортировке 'input'...")
    else:
        print("Нет файлов ни в 'work', ни в 'input'.")
        return

    if run_slicing and not slice_with_vad(): return

    knowledge_base = {}
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.wav', '.mp3', '.m4a'))])
    total_files = len(files)

    for f_idx, f_name in enumerate(files, 1):
        path = os.path.join(INPUT_DIR, f_name); feat = get_features(path)
        if not feat: continue
        assigned_spk = None

        if not knowledge_base:
            play_audio(path, "ПЕРВЫЙ ГОЛОС")
            name = input("Имя персонажа: ").strip() or "speaker_0"
            assigned_spk = name
        else:
            all_scores = []
            for spk, data in knowledge_base.items():
                m_s = [cosine_similarity(feat['mfcc'], f['mfcc']).item() for f in data['features']]
                c_s = [cosine_similarity(feat['chroma'], f['chroma']).item() for f in data['features']]
                s_s = [cosine_similarity(feat['spec'], f['spec']).item() for f in data['features']]
                w_s = [get_waveform_similarity(feat['raw_y'], f['raw_y']) for f in data['features']]
                group_finals = [(m_s[j]*0.80 + c_s[j]*0.10 + s_s[j]*0.05 + w_s[j]*0.05) for j in range(len(m_s))]
                avg_val = np.mean(group_finals); b_idx = np.argmax(group_finals)
                all_scores.append({'name': spk, 'avg_final': avg_val, 'best_scores': (m_s[b_idx], c_s[b_idx], s_s[b_idx], w_s[b_idx]), 'sample_p': data['paths'][b_idx], 'count': len(data['features']), 'sure': data['sure_threshold']})
            
            all_scores.sort(key=lambda x: x['avg_final'], reverse=True)
            top = all_scores[0]; margin = top['avg_final'] - all_scores[1]['avg_final'] if len(all_scores) > 1 else 0.0
            
            if (len(all_scores) == 1 and top['avg_final'] >= 0.985 and top['count'] >= MIN_SAMPLES_FOR_AUTO) or \
               (len(all_scores) > 1 and ((top['avg_final'] >= top['sure'] and margin >= (AUTO_MARGIN/2)) or margin >= (AUTO_MARGIN*1.5))):
                assigned_spk = top['name']
                print(f"[{f_idx}/{total_files}] АВТО: {f_name} -> {top['name']}")
            
            if not assigned_spk:
                for i, opt in enumerate(all_scores):
                    if opt['avg_final'] > GLOBAL_DOUBT:
                        ans = handle_choice(path, f_name, opt['name'], opt['best_scores'], opt['avg_final'], opt['sure'], i, len(all_scores), f_idx, total_files, opt['sample_p'])
                        
                        if ans == '': ans = 'y'
                        
                        if ans == 'y':
                            assigned_spk = opt['name']
                            new_val = (opt['sure'] * 0.7) + (opt['avg_final'] * 0.3)
                            knowledge_base[opt['name']]['sure_threshold'] = np.clip(new_val - 0.005, 0.85, 0.980)
                            break
                        elif ans == 'n':
                            knowledge_base[opt['name']]['sure_threshold'] = np.clip(max(opt['sure'], opt['avg_final'] + 0.005), 0.85, 0.990)
                            continue
                        elif ans == 's': 
                            assigned_spk = "skip"; break
                        else:
                            assigned_spk = ans
                            if assigned_spk in knowledge_base:
                                old_sure = knowledge_base[assigned_spk]['sure_threshold']
                                new_val = (old_sure * 0.7) + (opt['avg_final'] * 0.3)
                                knowledge_base[assigned_spk]['sure_threshold'] = np.clip(new_val - 0.005, 0.85, 0.980)
                            break
                
                # ИСПРАВЛЕНО: спрашиваем только если assigned_spk всё еще пустой (не ввели в цикле)
                if not assigned_spk:
                    play_audio(path, f"НОВЫЙ: {f_name}")
                    new_n = input(f"Создать новую группу для {f_name} (ИМЯ): ").strip() or f"spk_{len(knowledge_base)}"
                    assigned_spk = new_n

        if assigned_spk and assigned_spk != "skip":
            if assigned_spk not in knowledge_base:
                knowledge_base[assigned_spk] = {'features': [], 'paths': [], 'sure_threshold': 0.965}
            knowledge_base[assigned_spk]['features'].append(feat)
            target = os.path.join(OUTPUT_DIR, assigned_spk); os.makedirs(target, exist_ok=True)
            shutil.copy2(path, os.path.join(target, f_name))
            knowledge_base[assigned_spk]['paths'].append(os.path.join(target, f_name))
    print("\nГотово.")

if __name__ == "__main__":
    main()
