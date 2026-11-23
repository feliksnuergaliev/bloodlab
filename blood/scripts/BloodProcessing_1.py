from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def read(path):
    values = []
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            values.append(float(s))
        except ValueError:
            pass
    return np.array(values)

DATA = "data/"
# Укажите свои четыре файла
files = {
    40:  "40 mmHg.txt",
    80:  "80 mmHg.txt",
    120: "120 mmHg.txt",
    160: "160 mmHg.txt",
}

adc = []
pressure = []

for p, filename in files.items():
    data = read(DATA+filename)
    adc.append(data.mean())
    pressure.append(p)

adc = np.array(adc)
pressure = np.array(pressure)

# P = k*N + b
(k, b), cov = np.polyfit(adc, pressure, 1, cov=True)
err = np.sqrt(np.diag(cov))

print(f"P = {k:.4f} * N + {b:.2f}")
print(f"k = {k:.4f} ± {err[0]:.4f}")
print(f"b = {b:.2f} ± {err[1]:.2f}")

xx = np.linspace(adc.min() - 2, adc.max() + 2, 100)
yy = k * xx + b

plt.figure()
plt.scatter(adc, pressure, label="эксперимент")
plt.plot(xx, yy, label=f"P = {k:.3f}·N + {b:.2f}")
plt.xlabel("Отсчёт АЦП N")
plt.ylabel("Давление P, мм рт. ст.")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("plots/pressure-calibration.png", dpi=200)
plt.show()