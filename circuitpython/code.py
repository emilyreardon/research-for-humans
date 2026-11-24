import board
import neopixel
import supervisor
import sys
import time

pixels = neopixel.NeoPixel(board.NEOPIXEL, 10, brightness=1.0, auto_write=False)
pixels.fill((0, 200, 0))
pixels.show()
time.sleep(0.4)
pixels.fill((0, 0, 0))
pixels.show()

def wheel(pos):
    pos = pos & 255
    if pos < 85:
        return (255 - pos * 3, pos * 3, 0)
    elif pos < 170:
        pos -= 85
        return (0, 255 - pos * 3, pos * 3)
    else:
        pos -= 170
        return (pos * 3, 0, 255 - pos * 3)

animating = False
frame = 0
buf = ""

while True:
    if supervisor.runtime.serial_bytes_available:
        ch = sys.stdin.read(1)
        if ch in ('\r', '\n'):
            cmd = buf.strip()
            if cmd:
                print("CMD:" + cmd)
                if cmd == "ON":
                    animating = True
                    frame = 0
                elif cmd == "OFF":
                    animating = False
                    pixels.fill((0, 0, 0))
                    pixels.show()
            buf = ""
        else:
            buf += ch

    if animating:
        # Comet head travels one full lap every ~1.6s (8 ticks × 20ms × 10 pixels)
        head = (frame // 8) % 10
        # Hue base drifts slowly so colors shift as the comet cycles
        hue_base = (frame * 3) & 255

        for i in range(10):
            dist = (i - head) % 10
            r, g, b = wheel((hue_base + i * 25) & 255)
            if dist == 0:
                pixels[i] = (r, g, b)
            elif dist == 1:
                pixels[i] = (r >> 1, g >> 1, b >> 1)
            elif dist == 2:
                pixels[i] = (r >> 2, g >> 2, b >> 2)
            elif dist == 3:
                pixels[i] = (r >> 3, g >> 3, b >> 3)
            else:
                pixels[i] = (max(6, r >> 5), max(6, g >> 5), max(6, b >> 5))

        pixels.show()
        frame = (frame + 1) % 2048
        time.sleep(0.02)
