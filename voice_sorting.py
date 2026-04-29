import os
import shutil
import librosa
import numpy as np
import pygame
import time
import warnings
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")
pygame.mixer.init()

# --- НАСТРОЙКИ ---
GLOBAL_DOUBT = 0.70   
MIN_SAMPLES_FOR_AUTO = 3 
AUTO_MARGIN = 0.03    # Зазор между 1-м и 2-м местом

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "sorted")

def play_audio(path, label="файл"):
    if path and os.path.exists(path):
        print(f">> {label}: {os.path.basename(path)}")
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy(): pygame.time.Clock().tick(10)

def reduce_noise(y):
    stft = librosa.stft(y)
    mag, phase = librosa.magphase(stft)
    noise_est = np.median(mag, axis=1, keepdims=True)
    mag_clean = np.maximum(mag - 1.5 * noise_est, 1e-10)
    return librosa.istft(mag_clean * phase)

def get_waveform_similarity(y1, y2, sr=16000):
    win_len = int(0.5 * sr)
    ml = min(len(y1), len(y2))
    if ml < win_len: return 0.0
    scores = []
    for i in range(0, ml - win_len, win_len):
        seg1 = y1[i : i + win_len]
        search_radius = int(0.2 * sr)
        start = max(0, i - search_radius)
        end = min(ml - win_len, i + search_radius)
        best_seg_score = 0
        for j in range(start, end, int(sr * 0.05)):
            seg2 = y2[j : j + win_len]
            corr = np.corrcoef(seg1, seg2)
            if not np.any(np.isnan(corr)):
                # Извлекаем значение коэффициента корреляции из матрицы 2x2
                if isinstance(corr, np.ndarray) and corr.shape == (2,2):
                    val = corr[0, 1]
                else:
                    val = 0
                best_seg_score = max(best_seg_score, val)
        scores.append(best_seg_score)
    return np.mean(scores) if scores else 0.0

def get_features(path):
    try:
        y, sr = librosa.load(path, sr=16000, duration=10)
        if len(y) < int(0.3 * sr): return None
        y, _ = librosa.effects.trim(y, top_db=25)
        y = librosa.util.normalize(y); y = reduce_noise(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mels=64, n_mfcc=40)
        delta = librosa.feature.delta(mfcc)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        mfcc_c = np.hstack([np.mean(mfcc, axis=1), np.mean(delta, axis=1), np.mean(contrast, axis=1)]).reshape(1, -1)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1).reshape(1, -1)
        spec = np.mean(librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max), axis=1).reshape(1, -1)
        return {"mfcc": mfcc_c, "chroma": chroma, "spec": spec, "raw_y": y}
    except: return None

def handle_choice(path, spk, scores, group_avg, current_sure, idx, total_groups, f_idx, total_files, sample_p):
    s, c, sp, w = scores
    current_final = (s * 0.80) + (c * 0.10) + (sp * 0.05) + (w * 0.05)
    print(f"\n" + "="*80)
    print(f"ПРОГРЕСС: [{f_idx}/{total_files}] | ГРУППА: [{spk}] ({idx+1}/{total_groups})")
    print(f"СХОДСТВО ФАЙЛА: {current_final:.4f} | СРЕДНЕЕ ГРУППЫ: {group_avg:.4f} | SURE: {current_sure:.4f}")
    print("="*80)
    play_audio(sample_p, "ОБРАЗЕЦ"); time.sleep(0.2); play_audio(path, "ТЕКУЩИЙ")
    while True:
        choice = input("\n[y] ДА | [n] НЕТ | [r] ПОВТОР | [s] ПРОПУСК: ").lower().strip()
        if choice == 'r': play_audio(sample_p, "ОБРАЗЕЦ"); time.sleep(0.2); play_audio(path, "ТЕКУЩИЙ")
        elif choice in ['y', 'n', 's']: pygame.mixer.music.stop(); return choice

# --- ЗАПУСК ---
knowledge_base = {}
if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)
os.makedirs(INPUT_DIR, exist_ok=True)
files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.wav', '.mp3', '.ogg', '.m4a'))])
total_files = len(files)

try:
    for f_idx, f_name in enumerate(files, 1):
        path = os.path.join(INPUT_DIR, f_name)
        feat = get_features(path)
        if not feat: continue
        assigned_spk = None

        if not knowledge_base:
            print(f"\n[{f_idx}/{total_files}] ПЕРВЫЙ ГОЛОС: {f_name}")
            play_audio(path, "СЛУШАЕМ")
            name = input("Имя персонажа: ").strip() or "speaker_0"
            knowledge_base[name] = {'features': [], 'paths': [], 'sure_threshold': 0.965}
            assigned_spk = name
        else:
            all_scores = []
            for spk, data in knowledge_base.items():
                m_s = [cosine_similarity(feat['mfcc'], f['mfcc']).item() for f in data['features']]
                c_s = [cosine_similarity(feat['chroma'], f['chroma']).item() for f in data['features']]
                s_s = [cosine_similarity(feat['spec'], f['spec']).item() for f in data['features']]
                w_s = [get_waveform_similarity(feat['raw_y'], f['raw_y']) for f in data['features']]
                
                group_finals = [(m_s[j]*0.80 + c_s[j]*0.10 + s_s[j]*0.05 + w_s[j]*0.05) for j in range(len(m_s))]
                avg_group_final = np.mean(group_finals)
                best_idx = np.argmax(group_finals)
                all_scores.append({
                    'name': spk, 'avg_final': avg_group_final, 
                    'best_scores': (m_s[best_idx], c_s[best_idx], s_s[best_idx], w_s[best_idx]),
                    'sample_p': data['paths'][best_idx], 'count': len(data['features']),
                    'sure': data['sure_threshold']
                })
            
            all_scores.sort(key=lambda x: x['avg_final'], reverse=True)
            top = all_scores[0] # Исправлено: обращаемся к первому элементу списка
            
            margin = 0.0 
            if len(all_scores) > 1:
                margin = top['avg_final'] - all_scores[1]['avg_final'] # Сравнение со ВТОРЫМ по силе
            
            can_auto = False
            if len(all_scores) == 1:
                # Если группа одна - авто только при очень высоком сходстве
                if top['avg_final'] >= 0.985 and top['count'] >= MIN_SAMPLES_FOR_AUTO:
                    can_auto = True
            else:
                # Если групп больше одной - работает механика зазора
                if top['avg_final'] >= top['sure'] and margin >= (AUTO_MARGIN / 2):
                    can_auto = True
                elif margin >= (AUTO_MARGIN * 1.5):
                    can_auto = True

            if can_auto:
                assigned_spk = top['name']
                print(f"[{f_idx}/{total_files}] АВТО: {f_name} -> [{top['name']}] (Балл: {top['avg_final']:.4f} | Зазор: {margin:.4f})")
            
            if not assigned_spk:
                for i, opt in enumerate(all_scores):
                    if opt['avg_final'] > GLOBAL_DOUBT:
                        ans = handle_choice(path, opt['name'], opt['best_scores'], opt['avg_final'], opt['sure'], i, len(all_scores), f_idx, total_files, opt['sample_p'])
                        if ans == 'y':
                            new_val = (opt['sure'] * 0.7) + (opt['avg_final'] * 0.3)
                            knowledge_base[opt['name']]['sure_threshold'] = np.clip(new_val - 0.005, 0.85, 0.980)
                            assigned_spk = opt['name']; break
                        elif ans == 'n':
                            knowledge_base[opt['name']]['sure_threshold'] = np.clip(max(opt['sure'], opt['avg_final'] + 0.005), 0.85, 0.990)
                            continue
                        elif ans == 's': assigned_spk = "skip"; break
                
                if not assigned_spk:
                    play_audio(path, "НОВЫЙ")
                    new_n = input(f"Новая группа для {f_name}: ").strip() or f"spk_{len(knowledge_base)}"
                    knowledge_base[new_n] = {'features': [], 'paths': [], 'sure_threshold': 0.965}
                    assigned_spk = new_n

        if assigned_spk and assigned_spk != "skip":
            if assigned_spk not in knowledge_base:
                knowledge_base[assigned_spk] = {'features': [], 'paths': []}
            knowledge_base[assigned_spk]['features'].append(feat)
            target = os.path.join(OUTPUT_DIR, assigned_spk); os.makedirs(target, exist_ok=True)
            shutil.copy2(path, os.path.join(target, f_name))
            knowledge_base[assigned_spk]['paths'].append(os.path.join(target, f_name))
finally:
    pygame.mixer.quit()
