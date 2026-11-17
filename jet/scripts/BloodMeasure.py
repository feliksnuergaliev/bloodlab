# bloodMeasure.py
#   python3 bloodMeasure.py "40 mmHg" 10
import sys
import os
import time
import numpy as np
from mcp3021_driver import MCP3021


def save(samples, start, finish, filename):
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    with open(filename, "w") as outfile:
        outfile.write("- Blood Lab\n")
        outfile.write("- Date: {}\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start))))
        outfile.write("- Duration: {:.2f} s\n\n".format(finish - start))
        np.savetxt(outfile, np.array(samples), fmt="%d")


if len(sys.argv) != 3:
    print('Использование: python3 bloodMeasure.py "имя" секунды')
    sys.exit(1)

name = sys.argv[1]               # имя файла без .txt
duration = float(sys.argv[2])    # секунд

samples = []
adc = MCP3021(5.0)
try:
    start = time.time()
    while time.time() - start < duration:   # в цикле только чтение и append
        samples.append(adc.get_number())
    finish = time.time()
finally:
    adc.deinit()

print("Отсчётов: %d, частота %.0f Гц" % (len(samples), len(samples) / (finish - start)))
save(samples, start, finish, "data/%s.txt" % name)