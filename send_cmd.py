import serial
import time

ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
time.sleep(0.1)
ser.reset_input_buffer()

cmd = bytes([0x55, 0xAA, 0x12, 0x0C,
             0x2B, 0x87, 0x16, 0x3E,
             0x6F, 0x12, 0x83, 0xBA,
             0x77, 0xBE, 0xDF, 0x3E,
             0x33])

ser.write(cmd)
ser.flush()
print(f"已发送: {cmd.hex(' ')}")

time.sleep(0.2)
data = ser.read(256)
if data:
    print(f"收到回复: {data.hex(' ')}")
else:
    print("未收到回复")

ser.close()
