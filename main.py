import pprint
import time

import cv2
import numpy as np
import mss
from PIL import Image
import os
import scipy.signal
from scipy import ndimage
import FishingFrame


def draw_res_on_img(image, template_image, res):
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    bottom_right = max_loc[0] + np.shape(template_image)[1], max_loc[1] + np.shape(template_image)[0]
    return cv2.rectangle(image, max_loc, bottom_right, 255, 2)

def subshot_from_res(image, template_image, res):
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    return image[max_loc[1]:max_loc[1] + template_image.shape[0], max_loc[0]:max_loc[0] + template_image.shape[1]]

def main():

    #test_image = np.array(Image.open("resources/fishing_still_test.jpg"))
    test_image = np.array(Image.open("resources/fishing_still_test1.jpg"))
    #test_image = np.array(Image.open("resources/fishing_still_test2.jpg"))
    #test_image = np.array(Image.open("resources/fishing_still_test3.jpg"))


    incorrect_image = np.array(Image.open("resources/incorrect_still.jpg"))
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    sct = mss.mss()
    mon = {'left': 0, 'top': 0, 'width': 1920, 'height': 1080}
    ff = None

    print(FishingFrame.FishingFrame.fishing_frame.shape)

    time.sleep(2)
    while not ff:
        #if cv2.waitKey(0) == ord('q'):
        #    break
        ss = sct.grab(mon)

        image = np.asarray(Image.frombytes(
            'RGB',
            (ss.width, ss.height),
            ss.rgb,
        ))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #cv2.imshow("T", np.array(image))

        ff = FishingFrame.FishingFrame.create_from_image(image)
    print(ff)

    video = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc(*"XVID"), 30, (ff.width, ff.height))

    start_time = time.time()
    ims = []
    while time.time() - start_time < 5:
        print(time.time() - start_time)
        ff.find(test_image)
        ff.find_fish()
        ff.find_fishing_bar()
        # ims.append(ff.get_image())

    for im in ims:
        video.write(im)
    video.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
