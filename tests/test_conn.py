import serial, time

port = "COM10"          # change to your port
baud = 115200

s = serial.Serial(port, baud, timeout=2)
time.sleep(2)          # wait for Arduino to reset after USB connect

s.write(b'\xFF')       # send one byte
time.sleep(0.1)
reply = s.read(1)

if reply == b'\xFF':
    print("SUCCESS — Arduino received and echoed back 0xFF")
else:
    print(f"FAIL — got {reply!r} instead of b'\\xFF'")

s.close()