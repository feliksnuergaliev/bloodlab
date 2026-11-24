import os
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

K, B = 0.4093, -10.06


def read(filename):
    with open(filename) as f:
        lines = f.readlines()
    duration = float(lines[2].split()[2])
    samples = np.array([int(x) for x in lines[4:] if x.strip()])
    return samples, duration


def moving_average(x, n):
    return np.convolve(x, np.ones(n) / n, mode="same")


def process(filename, label, out_dir):
    counts, duration = read(filename)
    rate = len(counts) / duration
    t = np.linspace(0, duration, len(counts))

    p = K * counts + B
    p = moving_average(p, max(1, int(rate * 0.05)))

    trend = moving_average(p, max(1, int(rate * 1.5)))
    wave = p - trend

    edge = int(rate * 1.5)
    start = np.argmax(trend[int(rate * 2):]) + int(rate * 2) + int(rate * 5)
    below = np.where(trend[start:] < 60)[0]
    stop = start + below[0] if len(below) else len(trend)
    start, stop = max(start, edge), min(stop, len(trend) - edge)

    t_c = t[start:stop]
    p_c = p[start:stop]
    trend_c = trend[start:stop]
    wave_c = wave[start:stop]

    peaks, _ = find_peaks(wave_c, distance=int(rate * 0.4), prominence=0.3)
    troughs, _ = find_peaks(-wave_c, distance=int(rate * 0.4), prominence=0.3)

    inside = peaks[(t_c[peaks] > 10) & (t_c[peaks] < 30)]
    if len(inside) > 1:
        pulse = (len(inside) - 1) * 60 / (t_c[inside[-1]] - t_c[inside[0]])
    else:
        pulse = np.nan

    amp, tAmp, pap = [], [], []
    for a, c in zip(troughs[:-1], troughs[1:]):
        top = a + np.argmax(wave_c[a:c])
        amp.append(wave_c[top] - wave_c[a])
        tAmp.append(t_c[top])
        pap.append(trend_c[top])

    amp = np.array(amp)
    tAmp = np.array(tAmp)
    pap = np.array(pap)

    t_sys, p_sys, t_dia, p_dia = np.nan, np.nan, np.nan, np.nan
    env = np.array([])
    valid_waves = np.array([])

    if len(amp) > 0:
        env = median_filter(amp, size=7)

        valid_waves = np.where(pap > 40)[0]
        if len(valid_waves) >= 5:
            sys_idx = valid_waves[4]
            t_sys = tAmp[sys_idx]
            p_sys = pap[sys_idx]

        search_limit = max(1, len(env) - 10)
        imax = np.argmax(env[:search_limit])

        after = np.where(env[imax:] <= 0.8 * env[imax])[0]

        if len(after) > 0:
            dia_idx = imax + after[0]
            t_dia = tAmp[dia_idx]
            p_dia = pap[dia_idx]
        else:
            dia_idx = len(pap) - 1
            t_dia = tAmp[dia_idx]
            p_dia = pap[dia_idx]
            print(f"  [Внимание] Амплитуда не упала до 0.8. Диастола взята по нижнему давлению.")

    print(f"--- {label}")
    print(f"  Пульс: {pulse:.0f} уд/мин")
    print(f"  Давление: {p_sys:.0f}/{p_dia:.0f} мм рт. ст.")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14))

    ax1.plot(t, p, lw=1, label="Давление в манжете")
    if not np.isnan(p_sys):
        ax1.plot([t_sys, t_dia], [p_sys, p_dia], "r*", markersize=12,
                 label=f"Давление {p_sys:.0f}/{p_dia:.0f} мм рт. ст.")
        ax1.text(t_sys + 1, p_sys + 4, "Систола", color="red", fontsize=12)
        ax1.text(t_dia + 1, p_dia + 4, "Диастола", color="red", fontsize=12)
    ax1.set_xlabel("Время, с")
    ax1.set_ylabel("Давление, мм рт. ст.")
    ax1.set_title(f"{label}: Давление")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2.plot(t_c, wave_c, lw=0.8, label="Пульсации")
    ax2.plot(t_c[peaks], wave_c[peaks], "r.", markersize=6, label="Пики")
    ax2.set_xlim(10, 30)
    ax2.set_xlabel("Время, с")
    ax2.set_ylabel("Пульсации, мм рт. ст.")
    ax2.set_title(f"{label}: Пульс {pulse:.0f} уд/мин")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    if len(amp) > 0:
        ax3.scatter(tAmp, amp, alpha=0.4, label="амплитуда каждой волны", color="tab:blue")
        ax3.plot(tAmp, env, label="сглаженная амплитуда", color="tab:blue", lw=2)

        if len(valid_waves) >= 5:
            first5 = valid_waves[:5]
            ax3.scatter(tAmp[first5], amp[first5], facecolors='none', edgecolors='r', s=60, label="первые пять волн")
            ax3.axvline(t_sys, color='r', ls='--', alpha=0.5)
            ax3.text(t_sys - 1, max(amp) * 0.9, f"5-я волна:\nсистолическое {p_sys:.0f}", color='r', ha='right',
                     fontsize=10)

        if not np.isnan(p_dia):
            ax3.axhline(0.8 * env[imax], color='g', ls='--', lw=1, label="0,8 от наибольшей")
            ax3.axvline(t_dia, color='g', ls='--', alpha=0.5)
            ax3.text(t_dia + 1, max(amp) * 0.9, f"амплитуда\nуменьшилась:\nдиастолическое {p_dia:.0f}", color='g',
                     ha='left', fontsize=10)

        ax3.set_xlabel("Время, с")
        ax3.set_ylabel("Амплитуда волн, мм рт. ст.")
        ax3.set_title(f"{label}: Амплитуда пульсовых волн")
        ax3.legend(loc="lower left")
        ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{label}-analysis.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    out_dir = "plots"
    os.makedirs(out_dir, exist_ok=True)

    filenames = ["rest.txt", "fitness.txt", "rest_aidar.txt", "fitness_aidar.txt"]

    for filename in filenames:
        filepath = os.path.join("data", filename)
        if os.path.exists(filepath):
            label = os.path.splitext(filename)[0]
            process(filepath, label, out_dir)
        else:
            print(f"Файл {filepath} не найден!")

    print(f"\nГотово! Графики сохранены в папку '{out_dir}'.")