import pprint

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
    #test_image = np.array(Image.open("resources/fishing_still_test1.jpg"))
    #test_image = np.array(Image.open("resources/fishing_still_test2.jpg"))
    test_image = np.array(Image.open("resources/fishing_still_test3.jpg"))


    incorrect_image = np.array(Image.open("resources/incorrect_still.jpg"))
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

    ff = FishingFrame.FishingFrame.create_from_image(test_image)
    while True:
        ff.find(test_image)
        ff.find_fish()
        image = ff.get_image()
        image = cv2.rectangle(image, (75, 17), (105, 575), 255, 2)
        print(image.shape)
        bar = image[19:575, 70:110, :]

        #filter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        filter_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])



        filtered = cv2.filter2D(bar, -1, filter_x)

        #filtered = ndimage.convolve(bar, filter_x, mode='same')
        # bar = cv2.cvtColor(bar, v2.)
        #for i in range(3):
        #    # print(filter_x)
        #    filtered[:, :, i] = scipy.signal.convolve2d(bar[:, :, i], filter_x, mode='same')

        print(filtered.shape, bar.shape)

        cv2.imshow("TEST %i2", filtered)




        # print(filtered)
        print(filtered.sum(1).shape)


        """
        Finding bar.
            Perform filter using gaussian derivative
            Using filtered image
            Calculate sum of row in bar
            Find bars that exceed threshold
            If bar is within fish bounds, dont consider it
        """
        for i, num in enumerate(filtered.sum(1).sum(1)):
            if num < 5000:
                continue
            #if ff.fish_position[0] < i < ff.fish_position[0] + ff.fish_image.shape[0]:
            print(ff.fish_position, ff.fish_image.shape, i)
            if ff.fish_position[1] < i < ff.fish_position[1] + ff.fish_image.shape[1]:
                continue
            bar = cv2.line(bar, (0, i), (50, i), 255, 2)
            print(i, num)

        cv2.imshow("TEST %i", bar)



        cv2.imshow("Screenshot2", ff.get_image())
        if cv2.waitKey(0) == ord('q'):
            break



if __name__ == "__main__":
    main()
