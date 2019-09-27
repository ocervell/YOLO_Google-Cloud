import logging
import threading
import time
import glob
import numpy as np
import pandas as pd
from pydarknet import Detector
from pydarknet import Image as darknetImage
from data_structures import OutputClassificationData

LOGGER = logging.getLogger(__name__)

class DarknetYOLO(threading.Thread):

    def __init__(self, image_data, yolo_dir, score_thresh=0.5, fps=0.08):
        LOGGER.info("Directory: " + yolo_dir)
        self.createDataFile(yolo_dir)
        data =  glob.glob(yolo_dir + '*.data')[0]
        config =  glob.glob(yolo_dir + '*.cfg')[0]
        weights =  glob.glob(yolo_dir + '*.weights')[0]
        cls_names =  glob.glob(yolo_dir + '*.names')[0]

        LOGGER.info("Yolo data: %s" % data)
        LOGGER.info("Yolo config: %s" % config)
        LOGGER.info("Yolo weights: %s" % weights)
        LOGGER.info("Yolo class names: %s" % cls_names)

        self.createClassNames(yolo_dir, cls_names)
        self.done = False
        threading.Thread.__init__(self)
        self.pause = False
        self.name = "YOLO Predictor Thread"
        self.image_data = image_data

        LOGGER.info("Launching detector ...")
        self.net = net = Detector(
            bytes(config, encoding="utf-8"),
            bytes(weights, encoding="utf-8"),
            0,
            bytes(data, encoding="utf-8"))
        self.results = []
        self.output_data = OutputClassificationData()
        self.output_data.score_thresh = score_thresh
        self.frames_per_ms = fps;
        LOGGER.info("Score thresh: %s" % score_thresh)
        LOGGER.info("FPS: %s" % fps)

    def createDataFile(self, yolo_dir):
        filepath = yolo_dir + yolo_dir.split('/')[-2] + '.data'
        filenames = glob.glob(yolo_dir + '*.names')[0]
        LOGGER.info("Creating data file at %s ..." % filepath)
        num_classes = len(pd.read_csv(filenames,header=None).index.values)

        with open(filepath, "w+") as f:
            LOGGER.info("Writing to data file at %s" % filepath)
            f.write('classes= ' + str(num_classes) + '\n')
            f.write('names= ' + str(filenames) + '\n')
            f.close()

    def createClassNames(self,yolo_dir, cls_names):
        self.__BEGIN_STRING = ''
        self.cls_names = [self.__BEGIN_STRING + str(s)
                            for s in pd.read_csv(cls_names,header=None,names=['LabelName']).LabelName.tolist()]

        # Remove all of the odd characters
        for indx,x in enumerate(self.cls_names):
            if "'" in x:
                self.cls_names[indx] = x.replace("'","")

    def getLabelIndex(self, class_):
        class_ = str(class_.decode("utf-8"))
        # Get the remapped label
        label = class_
        indx = self.cls_names.index(class_) + 1
        return indx

    def predict_once(self, image_np):
        dark_frame = darknetImage(image_np)
        image_height,image_width,_ = image_np.shape
        results = self.net.detect(dark_frame,self.output_data.score_thresh)
        LOGGER.debug("Detection results: %s" % results)
        del dark_frame
        classes = []
        scores = []
        bbs = []
        for class_, score, bounds in results:
            x, y, w, h = bounds
            X = (x - w/2)/image_width
            Y = (y - h/2)/image_height
            X_ = (x + w/2)/image_width
            Y_ = (y + h/2)/image_height
            bbs.append([Y, X,Y_,X_])
            scores.append(score)
            index = self.getLabelIndex(class_)
            classes.append(index)
        scores = np.asarray(scores)
        classes = np.asarray(classes)
        bbs = np.asarray(bbs)

        self.output_data.scores = scores
        self.output_data.classes = classes
        self.output_data.bbs = bbs
        self.output_data.image_data.image_np = image_np
        LOGGER.info("Output: %s" % self.output_data)

        time.sleep(self.frames_per_ms)

    def predict(self,threadName):
        while not self.done:
            image_np = self.getImage()
            if not self.pause:
                LOGGER.info("Run predict_once on image %s" % image_np)
                self.predict_once(image_np)
            else:
                self.output_data.bbs = np.asarray([])
                time.sleep(2.0) # Sleep for 2 seconds

    def run(self):
        LOGGER.info("Starting " + self.name)
        self.predict(self.name)
        print("Exiting " + self.name)

    def pause_predictor(self):
        self.pause = True

    def continue_predictor(self):
        self.pause = False

    def stop(self):
        self.done = True

    def getImage(self):
        '''
        Returns the image that we will use for prediction.
        '''
        self.output_data.image_data.original_image_np = self.image_data.image_np
        self.output_data.image_data.image_np = self.image_data.image_np

        return self.output_data.image_data.image_np
