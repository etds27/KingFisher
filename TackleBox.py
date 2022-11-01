import time

import cv2
import mss
import numpy as np

from FishingFrame import FishingFrame
from PIL import Image

"""
Tackle box is the repository for common tools among libraries in this program
"""

_sct = mss.mss()


def waitForFF(timeout=1000):
    mon = {'left': 0, 'top': 0, 'width': 1920, 'height': 1080}
    ff = None
    # Wait for the frame to be found
    start_time = time.time()
    while not ff:
        image = getScreenshot(mon)

        ff = FishingFrame.create_from_image(image)

        cv2.imshow("Screen", image)
        cv2.waitKey(100)

        if time.time() - start_time > timeout:
            return None

    return ff


def_mon = {'left': 0, 'top': 0, 'width': 1920, 'height': 1080}

def getScreenshot(mon=None, gray=False):
    if mon is None:
        mon = def_mon

    ss = _sct.grab(mon)

    image = np.asarray(Image.frombytes(
        'RGB',
        (ss.width, ss.height),
        ss.rgb,
    ))

    if gray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image
