import logging
import os.path
import time
import re
import csv
import cv2
import numpy as np
import pynput
import datetime

import FishingFrame
import TackleBox


class DataGatherer:
    def __init__(self):
        self.mouse_is_pressed = False
        self.press_start_time = 0
        self.total_mouse_press_time = 0
        self.frame_rate = 25
        self.frame_rate_s = self.frame_rate / 1000.0
        self.data_dir = os.path.join(os.path.curdir, "training_data_%i" % self.frame_rate)
        self.raw_dir = os.path.join(self.data_dir, "raw")
        self.prep_dir = os.path.join(self.data_dir, "prepared")
        self.memory = 1500  # ms

        self.current_data_file_descriptor = None
        self.current_csv_writer = None
        self.prev_press_time = 0

        self.current_session_data = []
        self.fish_caught = False
        self.ff = None

        self.dict_keys = FishingFrame.FishingFrame.get_dict_keys()

        self.current_buffer = np.zeros((len(self.dict_keys) + 1) * int(self.memory / self.frame_rate))

        if not os.path.isdir(self.data_dir):
            os.mkdir(self.data_dir)

        if not os.path.isdir(self.raw_dir):
            os.mkdir(self.raw_dir)

        if not os.path.isdir(self.prep_dir):
            os.mkdir(self.prep_dir)

    def _mouseCallBack(self, x, y, button, pressed):
        if pressed:
            self.mouse_is_pressed = True
            self.press_start_time = time.time()
        if not pressed:
            self.mouse_is_pressed = False
            self.total_mouse_press_time += (time.time() - self.press_start_time) % self.frame_rate_s

    def _addDataPoint(self, X):
        if self.mouse_is_pressed:
            self.total_mouse_press_time += min(time.time() - self.press_start_time, self.frame_rate_s)

        self.current_buffer = np.roll(self.current_buffer, - (len(self.dict_keys) + 1))
        self.current_buffer[-(len(self.dict_keys) + 1):] = [X[k] for k in self.dict_keys] + [self.prev_press_time]
        self.prev_press_time = self.total_mouse_press_time
        self.total_mouse_press_time = 0

        return self.prev_press_time, self.current_buffer

    def _writeDataPoint(self, X):
        if self.mouse_is_pressed:
            self.total_mouse_press_time += min(time.time() - self.press_start_time, self.frame_rate_s)
        print("X: %s" % str(X))
        print("total_mouse_press_time: %f" % self.total_mouse_press_time)
        print("prest_start_time: %s" % self.press_start_time)
        print("press_time: %f" % (time.time() - self.press_start_time))
        print("frame_rate: %f" % self.frame_rate_s)
        print()

        self.current_session_data.append([X[k] for k in self.dict_keys] + [self.total_mouse_press_time])
        # print(self.current_session_data[-1])
        self.total_mouse_press_time = 0

    def _writePreparedDataPoint(self):
        if self.mouse_is_pressed:
            self.total_mouse_press_time += min(time.time() - self.press_start_time, self.frame_rate_s)

    def getPreparedDataPoint(self):
        return

    def getPreparedLiveData(self):
        pass


    def _prepareCurrentData(self, new_data, press_time):
        self.current_buffer = np.roll(self.current_buffer, - (len(self.dict_keys) + 1))
        self.current_buffer[-(len(self.dict_keys) + 1):] = new_data

        return self.current_buffer

    def _writeDataFile(self):
        """
        Writes the current game session data to the next file
        :return:
        """
        n = len(os.listdir(self.data_dir))
        file_name = "training_data_%s_%i_%i_%i.csv" % (datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                                                       self.frame_rate, self.fish_caught, n + 1)
        path = os.path.join(self.raw_dir, file_name)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            [writer.writerow(row) for row in self.current_session_data]

        self.fish_caught = False
        self.current_session_data = []

    def concatRawDataFile(self, file_name):
        """
        Function will convert the raw data file that is generated during the gatherTrainingData function and modulate
        it so the rows can be read into the neural network. It will concatenate the consecutive raw data points to create
        a single neural network input that contains contextual data for the last self.memory milliseconds. This input is
        accompanied by the optimized metric of progress difference. Which is the change in progress between the first
        and last data read of the window

        :param file_name:
        :return:
        """
        file_path = os.path.join(self.raw_dir, file_name)
        with open(file_path, "r", newline='') as fr:
            with open(os.path.join(self.prep_dir, file_name), "w", newline='') as fw:
                """
                Create a contiguous array that holds the training data from the last X ms
                One raw data point read is the length of the number of keys in the FF data params plus 1 for the mouse press time
                The length of the array is the size of the raw data point times the number of samples within the self.memory time contstraint
                
                
                Ex.
                Our memory is 3 sec (3000 ms)
                We take data reads every .025 sec (25 ms)
                
                Our number FF params is 4
                We have 1 mouse press value as well
                
                Num of vals        Our memory      Our frame rate
                (4 + 1)      *      (3000     /     25) 
                """
                buffer = [0] * (len(self.dict_keys) + 1) * int(self.memory / self.frame_rate)
                reader = csv.reader(fr)
                writer = csv.writer(fw)

                for i in reader:
                    buffer += i[:-1]
                    buffer = buffer[len(self.dict_keys) + 1:]

                    # The difference in game progress between first and last point in the time window
                    # Use this as the Y value
                    diff = float(buffer[-2]) - float(buffer[3])

                    writer.writerow([diff] + buffer)

                    # print(diff, buffer)

    def prepareAllRawData(self):
        files = os.listdir(self.raw_dir)
        for file_name in files:
            if not os.path.exists(os.path.join(self.prep_dir, file_name)):
                self.concatRawDataFile(file_name)
                logging.info("Prepared '%s'" % file_name)

    def liveDataFeed(self):
        return

    def gatherTrainingData(self):
        """
        Creates training data
        First col is if click is made


        :return:
        """
        pynput.mouse.Listener(on_click=self._mouseCallBack).start()

        while True:
            # Wait for the fishing game to appear on screen
            print("Waiting for fishing frame")
            ff = TackleBox.waitForFF()
            n = len(os.listdir(self.data_dir))
            file_name = "training_data_%s_%i_%i_%i.csv" % (datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                                                           self.frame_rate, self.fish_caught, n + 1)
            path = os.path.join(self.prep_dir, file_name)
            self.current_data_file_descriptor = open(path, "w", newline="")

            self.current_csv_writer = csv.writer(self.current_data_file_descriptor)

            prev_params = {k:0 for k in self.dict_keys}

            while True:
                # Make sure the game is still present on screen
                ret = ff.find()

                if ret is False:
                    break

                # Locate all important features of the fishing game
                ff.find_all()


                params = ff.get_params()
                if params["progress"] > 0.95:
                    self.fish_caught = True

                # self._writeDataPoint(params)
                y, X = self._addDataPoint(prev_params)
                ret = np.concatenate((np.array([y]), X))
                self.current_csv_writer.writerow(ret)

                prev_params = params
                if cv2.waitKey(self.frame_rate) == ord('q'):
                    break

            # Once the game has closed, write the game session data to the next file
            self.current_data_file_descriptor.close()
            # self._writeDataFile()
            self.fish_caught = False

    def clickTest(self):
        pynput.mouse.Listener(on_click=self._mouseCallBack).start()
        while True:
            self._writeDataPoint("X")
            if cv2.waitKey(self.frame_rate) == ord('q'):
                break


if __name__ == "__main__":
    dg = DataGatherer()
    # dg.prepareAllRawData()

    dg.current_buffer = [0.24035087719298243, 0.809009009009009, 0.9873873873873874, 0.8558558558558559,
                         0.7596491228070176, 0.025, 0.8126126126126126, 0.990990990990991, 0.8594594594594595,
                         0.7666666666666666, 0.025, 0.8126126126126126, 0.990990990990991, 0.8612612612612612,
                         0.7719298245614035, 0.025, 0.8108108108108109, 0.9891891891891892, 0.8558558558558559,
                         0.7771929824561403, 0.00200018882751464, 0.809009009009009, 0.9873873873873874,
                         0.8522522522522522, 0.7824561403508772, 0, 0.809009009009009, 0.9873873873873874,
                         0.8414414414414414, 0.7894736842105263, 0, 0.8108108108108109, 0.9891891891891892,
                         0.836036036036036, 0.7947368421052632, 0, 0.8162162162162162, 0.9945945945945946,
                         0.8234234234234235, 0.8017543859649123, 0, 0.8846846846846846, 0.9981981981981982,
                         0.8162162162162162, 0.8070175438596492, 0.025, 0.8756756756756757, 0.9981981981981982,
                         0.8072072072072072, 0.8140350877192982, 0.025, 0.872072072072072, 0.9981981981981982,
                         0.8036036036036036, 0.8175438596491228, 0.025, 0.863063063063063, 0.9891891891891892,
                         0.7945945945945946, 0.8140350877192982, 0.025, 0.8540540540540541, 0.9891891891891892,
                         0.7855855855855856, 0.8035087719298246, 0.025, 0.8432432432432433, 0.972972972972973,
                         0.7783783783783784, 0.7947368421052632, 0.025, 0.8396396396396396, 0.9585585585585585,
                         0.772972972972973, 0.7912280701754386, 0.007000398635864241, 0.7621621621621621,
                         0.9405405405405406, 0.7639639639639639, 0.7964912280701755, 0, 0.7477477477477478,
                         0.9261261261261261, 0.7549549549549549, 0.8035087719298246, 0, 0.7387387387387387,
                         0.9171171171171171, 0.7495495495495496, 0.8070175438596492, 0.0020003318786621094,
                         0.7279279279279279, 0.9063063063063063, 0.7441441441441441, 0.8140350877192982, 0.025,
                         0.7135135135135136, 0.8918918918918919, 0.7387387387387387, 0.8192982456140351, 0.025,
                         0.6972972972972973, 0.8756756756756757, 0.7315315315315315, 0.8263157894736842,
                         0.0019999980926513616, 0.6846846846846847, 0.863063063063063, 0.7297297297297297,
                         0.8315789473684211, 0, 0.6882882882882883, 0.8522522522522522, 0.7297297297297297,
                         0.8385964912280701, 0, 0.6828828828828829, 0.8432432432432433, 0.7315315315315315,
                         0.843859649122807, 0, 0.6828828828828829, 0.8378378378378378, 0.7333333333333333,
                         0.8508771929824561, 0, 0.6720720720720721, 0.8324324324324325, 0.736936936936937,
                         0.856140350877193, 0, 0.6720720720720721, 0.8324324324324325, 0.7423423423423423,
                         0.8614035087719298, 0, 0.6720720720720721, 0.8324324324324325, 0.7441441441441441,
                         0.8666666666666667, 0, 0.6720720720720721, 0.8342342342342343, 0.7477477477477478,
                         0.8719298245614036, 0, 0.6828828828828829, 0.8396396396396396, 0.7495495495495496,
                         0.8789473684210526, 0, 0.6720720720720721, 0.8486486486486486, 0.7567567567567568,
                         0.8842105263157894, 0, 0.6828828828828829, 0.8576576576576577, 0.7585585585585586,
                         0.8912280701754386, 0.025, 0.6882882882882883, 0.8666666666666667, 0.7603603603603604,
                         0.8964912280701754, 0.025, 0.6954954954954955, 0.8738738738738738, 0.7657657657657657,
                         0.9035087719298246, 0.005748414993286127, 0.7027027027027027, 0.8810810810810811,
                         0.7675675675675676, 0.9087719298245613, 0, 0.7099099099099099, 0.8882882882882883,
                         0.7711711711711712, 0.9122807017543859, 0.025, 0.7171171171171171, 0.8954954954954955,
                         0.7765765765765765, 0.9192982456140351, 0.025, 0.7243243243243244, 0.9027027027027027,
                         0.7837837837837838, 0.9245614035087719, 0.025, 0.7279279279279279, 0.9063063063063063,
                         0.790990990990991, 0.9315789473684211, 0.025, 0.7279279279279279, 0.9063063063063063,
                         0.7963963963963964, 0.9368421052631579, 0.0009999752044677623, 0.7297297297297297,
                         0.9081081081081082, 0.8018018018018018, 0.9438596491228071, 0, 0.7351351351351352,
                         0.9135135135135135, 0.809009009009009, 0.9491228070175438, 0.025, 0.736936936936937,
                         0.9153153153153153, 0.8126126126126126, 0.9543859649122807, 0.025, 0.7387387387387387,
                         0.9171171171171171, 0.8216216216216217, 0.9596491228070175, 0.003001260757446285,
                         0.7441441441441441, 0.9225225225225225, 0.827027027027027, 0.9666666666666667, 0,
                         0.7513513513513513, 0.9297297297297298, 0.8306306306306306, 0.9719298245614035, 0,
                         0.7567567567567568, 0.9351351351351351, 0.836036036036036, 0.9754385964912281,
                         0.015999555587768555, 0.7657657657657657, 0.9441441441441442, 0.8396396396396396,
                         0.9824561403508771, 0.025, 0.772972972972973, 0.9513513513513514, 0.8432432432432433,
                         0.987719298245614, 0.006002330780029291, 0.7765765765765765, 0.954954954954955,
                         0.8522522522522522, 0.9947368421052631, 0, 0.7837837837837838, 0.9621621621621622,
                         0.8558558558558559, 1.0, 0.025, 0.7891891891891892, 0.9675675675675676, 0.8612612612612612, 1,
                         0.025, 0.7927927927927928, 0.9711711711711711, 0.863063063063063, 1, 0.00700154304504394,
                         0.7963963963963964, 0.9747747747747748, 0.8684684684684685, 1, 0, 0.7945945945945946,
                         0.972972972972973, 0.8666666666666667, 1, 0.025, 0.7981981981981981, 0.9765765765765766,
                         0.8702702702702703, 1, 0.016999673843383786, 0.7945945945945946, 0.972972972972973,
                         0.8666666666666667, 1, 0, 0.7963963963963964, 0.9747747747747748, 0.8684684684684685, 1,
                         0.011000633239746094, 0.7963963963963964, 0.9747747747747748, 0.8684684684684685, 1,
                         1, 1, 1, 0.8702702702702703, 1, 0
                         ]
    dg._prepareCurrentData([0.8072072072072072, 0.9855855855855856, 0.8558558558558559, 0.7543859649122807, 0])
