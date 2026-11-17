# bloodProcessing.py — калибровка давления в манжете (п5.4)
import numpy as np
import matplotlib.pyplot as plt

DATA = "data/"
CAL = [ ("rest.txt", 160),("fitness.txt", 160)]


def read(filename):
    # формат как в о3: 3 строки заголовка, пустая строка, отсчёты
    with open(filename) as f:
        lines = f.readlines()
    duration = float(lines[2].split()[2])
    samples = np.array([int(x) for x in lines[4:] if x.strip()])
    return samples, duration, len(samples)


means, pressures = [], []
for filename, p in CAL:
    samples, duration, count = read(DATA + filename)
    means.append(samples.mean())
    pressures.append(p)
    print("%s: отсчётов %d, среднее %.1f" % (filename, count, samples.mean()))

means = np.array(means)
pressures = np.array(pressures)

(k, b), cov = np.polyfit(means, pressures, 1, cov=True)
dk, db = np.sqrt(np.diag(cov))
print("P = %.4f·N %+.2f  (k ± %.4f, b ± %.2f) мм рт. ст." % (k, b, dk, db))
print("Максимальное отклонение точек: %.2f мм рт. ст." %
      np.max(np.abs(pressures - (k * means + b))))

fit_x = np.linspace(means.min(), means.max(), 100)
plt.plot(means, pressures, "*", label="Измерения")
plt.plot(fit_x, k * fit_x + b, label="P = %.3f·N %+.1f" % (k, b))
plt.xlabel("Отсчёты АЦП")
plt.ylabel("Давление, мм рт. ст.")
plt.grid(which="major")
plt.grid(which="minor", linestyle="--")
plt.minorticks_on()
plt.legend()
plt.savefig("pressure-calibration.png", dpi=200)