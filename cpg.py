from pyfirmata import Arduino, util
import time 

board = Arduino("/dev/cu.usbmodem11301")
# pin = board.get_pin('d:5:i')
it = util.Iterator(board)
it.start()
time.sleep(1.0)



for pin in range(0,15):
    try:
        analog = board.get_pin(f"d:{pin}:i")
        print(analog.read())
    except:
        print(
            f'Pin {pin} doesnt work'
        )
time.sleep(1.0)
time.sleep(1.0)

for pin in range(0,15):
    try:
        analog = board.get_pin(f"d:{pin}:o")
        print(analog.read())
    except:
        print(
            f'Pin {pin} doesnt work'
        )
time.sleep(1.0)
time.sleep(1.0)

for pin in range(0,15):
    try:
        analog = board.get_pin(f"d:{pin}:p")
        print(analog.read())
    except:
        print(
            f'Pin {pin} doesnt work'
        )

