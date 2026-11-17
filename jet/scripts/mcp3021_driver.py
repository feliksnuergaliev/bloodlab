# mcp3021_driver.py — драйвер АЦП MCP3021 (10 бит) по I2C
#
# Использование:
#   from mcp3021_driver import MCP3021
#   adc = MCP3021(5.0)          # 5.0 — опорное напряжение, В
#   print(adc.get_number())     # отсчёт 0..1023
#   adc.deinit()
#
# Перед запуском: включите I2C (sudo raspi-config → Interface Options → I2C)
# и установите библиотеку: sudo apt install python3-smbus2

from smbus2 import SMBus, i2c_msg

MCP3021_ADDRESS = 0x4D   # адрес по умолчанию (1001101)


class MCP3021:
    def __init__(self, vref, bus_number=1, address=MCP3021_ADDRESS):
        self.vref = vref
        self.address = address
        self.bus = SMBus(bus_number)

    def get_number(self):
        # MCP3021 не имеет регистров: одно чтение из 2 байт
        read = i2c_msg.read(self.address, 2)
        self.bus.i2c_rdwr(read)
        first, second = list(read)
        # первый байт: 0 0 0 0 D9 D8 D7 D6
        # второй байт: D5 D4 D3 D2 D1 D0 0 0
        return ((first & 0x0F) << 6) | (second >> 2)

    def get_voltage(self):
        return self.get_number() * self.vref / 1024

    def deinit(self):
        self.bus.close()