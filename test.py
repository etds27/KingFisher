import pprint
import datetime
import time


import cv2
import numpy as np
import mss
import pynput.mouse
from PIL import Image
import os
import scipy.signal
from scipy import ndimage

import Angler
import TackleBox

from pynput.mouse import Listener

import DataGatherer
import FishingFrame


def similarColorTest():
    # arr = np.array([
    #    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    #    [[21, 22, 23], [4, 5, 6], [27, 28, 29]],
    #    [[11, 12, 13], [14, 15, 16], [17, 18, 19]]])
    # print(np.logical_and(arr > [3, 4, 5], arr < [5, 6, 7]))
    # print((arr == [4, 5, 6]).all(axis=1))
    # arr[np.where(np.all(arr == [4, 5, 6], axis=-1))] = [10, 10, 10]
    # print(arr)
    # exit()

    test_image = cv2.imread("test_resources/shapes-basic-color.png")
    cv2.imshow("Original Image", test_image)
    test_image = np.array(test_image)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    # mask = np.logical_and(test_image > [239, 150, 188], test_image < [249, 160, 198]).astype('int8')
    # mask = (test_image == [244, 154, 193]).astype('int8') * 255
    mask = np.zeros(test_image.shape)
    # mask[np.where(np.all(np.logical_and(test_image > [239, 149, 188], test_image < [249, 159, 198])))] = [255, 255, 255]
    # mask[np.where(np.all(test_image == [154, 193, 244]))] = [255, 255, 255]
    # mask[np.where(np.all(test_image == [154, 193, 244], axis=-1))] = [255, 255, 255]
    # mask[np.where(np.all(test_image == [255, 255, 255], axis=-1))] = [255, 255, 255]

    mask[np.where(np.all(np.logical_and(test_image > [239, 149, 188], test_image < [249, 159, 198]), axis=-1))] = [255,
                                                                                                                   255,
                                                                                                                   255]

    # mask[np.where(test_image < [150, 150, 150])] = 255

    # mask = (test_image == [154, 193, 244]).astype('int8') * 255
    print(test_image)
    print(mask.any())
    print(np.where(mask > 1))

    print(mask.shape, test_image.shape)
    # hsv = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV)

    # mask = cv2.inRange(hsv, lower, upper)
    cv2.imshow("TEST", mask)

    cv2.waitKey(0)
    pass


def findFishingBarColor():
    """
    Find the players fishing bar by using color identification

    Start by subtracting the blue fishing background # Not needed
    Assume all colors are between the inactive transparent user bar when at the top and at the bottom
    Create an empty mask to determine bar bounds
    Search for colors between these two bounds and set the mask to true

    Sum the RGB values of mask and then sum by row. Since white is the highest absolute RGB value, we can find a
        threshold for the sum of rows that determine whether the bar covers the row
    Take the top and bottom instance of bar rows as the bounds of the fishing bar.

    Seems to work well on all test images so far

    Important color RGB
    Low Blue: [70, 114, 223] # Not needed
    High Blue: [175, 208, 243] # Not needed
    Low Inactive Green : [88, 148, 159]
    High Inactive Green: [152, 202, 165]
    Active: [130, 229, 1]

    :return:
    """
    test_image = cv2.imread("resources/fishing_still_test_faint_bar.jpg")
    cv2.imshow("Original Image", test_image)

    ff = FishingFrame.FishingFrame.create_from_image(test_image)
    ff.find_fish()
    bar = ff.bar


    # Defining colors of interest for filtering
    tolerance = 25

    active_target = np.array([130, 229, 1])  # Active Color

    blue_lower = np.array([70, 114, 223]) - 10  # Small 10 value buffer
    blue_upper = np.array([175, 208, 243]) + 10  # Small 10 value buffer

    inactive_lower = np.array([88, 148, 135]) - 25  # X value buffer
    inactive_upper = np.array([160, 220, 189]) + 40

    active_lower = active_target - tolerance // 2
    active_upper = active_target + tolerance // 2

    bar = cv2.cvtColor(bar, cv2.COLOR_BGR2RGB)

    mask = np.zeros(bar.shape)

    #bar[np.where(np.all(np.logical_and(bar > blue_lower, bar < blue_upper), axis=-1))] = [0, 0, 0]
    mask[np.where(np.all(np.logical_and(bar > active_lower, bar < active_upper), axis=-1))] = [255, 255, 255]
    mask[np.where(np.all(np.logical_and(bar > inactive_lower, bar < inactive_upper), axis=-1))] = [255, 255, 255]

    w = np.where(np.sum(mask, -1).sum(-1) > 10000)
    ff.fb_upper, ff.fb_lower = w[0][0], w[0][-1]

    ff_image = ff.get_image()
    cv2.imshow("Frame", mask)
    cv2.imshow("Frame 2", ff_image)

    cv2.waitKey(0)


def fishing_image_test():
    test_image = cv2.imread("resources/fishing_still_test3.jpg")
    cv2.imshow("Original Image", test_image)

    ff = FishingFrame.FishingFrame.create_from_image(test_image)
    ff.find_fish()
    ff.find_fishing_bar()
    ff.find_progress()

    cv2.imshow("Frame", ff.get_image())
    cv2.waitKey(0)

def findProgressByColor():
    test_image = cv2.imread("resources/fishing_still_test_faint_bar.jpg")
    cv2.imshow("Original Image", test_image)

    ff = FishingFrame.FishingFrame.create_from_image(test_image)
    ff.find_progress()
    exit()
    pb = ff.progress_bar
    pb = cv2.cvtColor(pb, cv2.COLOR_BGR2RGB)

    target = np.array([141, 81, 47])
    tolerance = 50
    mask = np.zeros(pb.shape)

    w = np.where(np.all(np.logical_and(pb > target - tolerance, pb < target + tolerance), axis=-1))
    ff.progress = 1 - (w[0][-1] / ff.pb_h)
    print(ff.progress)

    ff_image = ff.get_image()
    cv2.imshow("PB", pb)
    cv2.imshow("Frame", mask)
    cv2.imshow("Frame 2", ff_image)

    cv2.waitKey(0)

def liveTracking():

    ff = TackleBox.waitForFF()

    print("Found Frame!")
    while True:
        ff.find()
        ff.find_all()
        print(ff)
        print()
        cv2.imshow("Game", ff.get_image())
        if cv2.waitKey(100) == ord('q'):
            break

def anchorTest():
    test_image = cv2.imread("resources/fishing_still_test1.jpg")
    #test_image = cv2.imread("resources/incorrect_still.jpg")

    anchor = cv2.imread("resources/fishing_anchor.jpg")

    cv2.imshow("Original Image", test_image)

    ff = FishingFrame.FishingFrame.create_from_image(test_image)
    frame_image = ff.get_image()

    res = cv2.matchTemplate(ff.image, anchor, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print(frame_image.shape, anchor.shape)
    print(np.array(max_loc) + anchor.shape[1::-1])
    frame_image = cv2.rectangle(frame_image, max_loc, np.array(max_loc) + anchor.shape[1::-1], 255 ,2)

    cv2.imshow("Frame", frame_image)

    cv2.waitKey(0)

def inventoryVisibileTest():
    test_image_true = cv2.imread("test_resources/exclaimation_faded.jpg")
    test_image_false = cv2.imread("test_resources/inventory_false_test.jpg")

    inventory_anchor = cv2.imread("resources/inventory_anchor.jpg")

    res = cv2.matchTemplate(test_image_true, inventory_anchor, cv2.TM_CCOEFF_NORMED)
    min_pct, max_pct, min_loc, max_loc = cv2.minMaxLoc(res)

    res = cv2.matchTemplate(test_image_false, inventory_anchor, cv2.TM_CCOEFF_NORMED)
    min_pct, max_pct, min_loc, max_loc = cv2.minMaxLoc(res)

    return max_pct > 0.95

def findExclaim():


    # print(exclaimation.shape)

    while True:
        image = TackleBox.getScreenshot(gray=True)
        # print(image.shape)
        res = cv2.matchTemplate(image, exclaimation, cv2.TM_CCOEFF_NORMED)
        min_pct, max_pct, min_loc, max_loc = cv2.minMaxLoc(res)
        print(max_pct, max_loc)

        if cv2.waitKey(5) == ord('q'):
            break

def findHit():

    if TackleBox.waitForFF(timeout=3):
        return True

    return False




def recordScreenshots(path="screen_record", frame_rate=5):
    i = 0

    while True:
        image = TackleBox.getScreenshot()
        image = Image.fromarray(image)
        image.save(os.path.join(path, "ss_%s.jpg" % datetime.datetime.now().strftime("%S_%f")))

        if cv2.waitKey(frame_rate) == ord('q'):
            break

def covertImage(path):
    image = cv2.imread(path)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image, "RGB")
    image.save("resources/HIT_solid.jpg")


if __name__ == "__main__":
    # similarColorTest()
    # findFishingBarColor()
    # fishing_image_test()
    # findProgressByColor()
    # liveTracking()
    # anchorTest()
    # createTrainData()

    # inventoryVisibileTest()

    a = Angler.Angler()
    loc = (0, 0)

    # time.sleep(1)

    while True:
        a.checkForExhaustion()

        image = TackleBox.getScreenshot({'left': 15, 'top': 850, 'width': 400, 'height': 150})
        cv2.imshow("Exhaustion", image)

        if cv2.waitKey(10) == ord('q'):
            break

        if a.game_state == Angler.GameState.EXHAUSTED:
            break

    #a.run()

    #dg = DataGatherer.DataGatherer()
    #dg.gatherTrainingData()
    # dg.concatRawDataFile("training_data_20221014_154943_25_1_1.csv")
    # dg.prepareAllRawData()

    #recordScreenshots()
    #covertImage("resources/HIT_bgr.jpg")
    # findHit()


