import logging
import time

import cv2
import mss
import numpy as np
import pyautogui

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


def findAndClickImage(image, gray=True, threshold=0.85, mon=def_mon):
    ss = getScreenshot(mon, gray=True)

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    res = cv2.matchTemplate(ss, image, cv2.TM_CCOEFF_NORMED)
    _, max_pct, _, max_loc = cv2.minMaxLoc(res)

    logging.debug("TackleBox.findAndClickImage: max_pct = %f | max_loc = %s" % (max_pct, str(max_loc)))
    if max_pct < threshold:
        return False

    max_loc = middleOfImage(max_loc, image)
    pyautogui.moveTo(*max_loc)
    pyautogui.mouseDown()
    pyautogui.mouseUp()
    return True

def middleOfImage(loc, image):
    return loc + np.array(image.shape[:1]) / 2

def middleOfMon(mon):
    return mon["left"] + mon["width"] // 2, mon["top"] + mon["height"] // 2
