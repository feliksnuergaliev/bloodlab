# bloodProcessing_records.py — обработка записей давления (п5.5)
#   python3 bloodProcessing_records.py rest.txt fitness.txt
import sys
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from scipy.signal import find_peaks, peak_prominences
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K, B = 0.4093, -10.06            # калибровка из п5.4: P = K·N + B, мм рт. ст.


def read(filename):
    with open(filename) as f:
        lines = f.readlines()
    duration = float(lines[2].split()[2])
    samples = np.array([int(x) for x in lines[4:] if x.strip()])
    return samples, duration


def moving_average(x, n):
    return np.convolve(x, np.ones(n) / n, mode="same")


def process(filename, label):
    counts, duration = read(filename)
    rate = len(counts) / duration                       # реальная частота, Гц
    t = np.linspace(0, duration, len(counts))
    p = K * counts + B                                  # мм рт. ст.

    p = moving_average(p, max(1, int(rate * 0.05)))     # сглаживание квантования
    # медленный спуск: полином по окну 1.5 с (без запаздывания, в отличие от скользящего среднего)
    trend = savgol_filter(p, int(rate * 1.5) | 1, 3)
    wave = p - trend                                    # пульсовые волны

    # рабочее окно: от максимума давления (после накачки) до падения ниже 60 мм рт. ст.
    # первые 5 с после максимума — переходный процесс открытия клапана, их пропускаем;
    # ниже 60 мм рт. ст. клапан открывают полностью, и возникают рывки, которые не пульс
    edge = int(rate * 1.5)
    start = np.argmax(trend[int(rate * 2):]) + int(rate * 2) + int(rate * 5)
    below = np.where(trend[start:] < 60)[0]
    stop = start + below[0] if len(below) else len(trend)
    start, stop = max(start, edge), min(stop, len(trend) - edge)
    t, p, trend, wave = t[start:stop], p[start:stop], trend[start:stop], wave[start:stop]

    peaks, _ = find_peaks(wave, distance=int(rate * 0.4), prominence=0.3)
    amp = peak_prominences(wave, peaks)[0]              # амплитуда каждой волны
    pap = trend[peaks]                                  # давление в манжете у волны
    tp = t[peaks]

    # сглаживаем огибающую медианой по 5 волнам, чтобы выбросы не задавали максимум
    env = median_filter(amp, size=5)
    imax = np.argmax(env)
    a_max = env[imax]

    before = np.where(env[:imax + 1] >= 0.55 * a_max)[0]        # первое достижение 0.55 (по времени)
    after = np.where(env[imax:] <= 0.8 * a_max)[0]              # падение до 0.8 после максимума
    systolic = pap[before[0]] if len(before) else np.nan
    diastolic = pap[imax + after[0]] if len(after) else np.nan
    pulse = (len(peaks) - 1) * 60 / (tp[-1] - tp[0])

    print(f"--- {label}")
    print(f"  частота дискретизации: {rate:.0f} Гц, рабочее окно {t[0]:.1f}–{t[-1]:.1f} с")
    print(f"  найдено волн: {len(peaks)}, пульс: {pulse:.0f} уд/мин")
    print(f"  систолическое: {systolic:.0f} мм рт. ст.")
    print(f"  диастолическое: {diastolic:.0f} мм рт. ст.")
    print(f"  давление у максимума амплитуды: {pap[imax]:.0f} мм рт. ст., амплитуда {a_max:.2f} мм рт. ст.")

    fig, ax = plt.subplots(3, 1, figsize=(9, 9))
    ax[0].plot(t, p, lw=0.5, label="давление")
    ax[0].plot(t, trend, label="тренд")
    ax[0].set_xlabel("время, с")
    ax[0].set_ylabel("мм рт. ст.")
    ax[0].legend()
    ax[1].plot(t, wave, lw=0.5)
    ax[1].plot(tp, wave[peaks], "r.", label="пики")
    ax[1].set_xlabel("время, с")
    ax[1].set_ylabel("пульсации, мм рт. ст.")
    ax[1].legend()
    ax[2].plot(pap, amp, ".", alpha=0.4, label="амплитуда волн")
    ax[2].plot(pap, env, label="медиана по 5 волнам")
    ax[2].axvline(systolic, color="g", ls="--", label=f"систолическое {systolic:.0f}")
    ax[2].axvline(diastolic, color="r", ls="--", label=f"диастолическое {diastolic:.0f}")
    ax[2].set_xlabel("давление в манжете, мм рт. ст.")
    ax[2].set_ylabel("амплитуда, мм рт. ст.")
    ax[2].invert_xaxis()
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(f"plot_{label}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    for name in sys.argv[1:]:
        process(name, name.replace(".txt", ""))