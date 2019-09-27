import warnings
warnings.filterwarnings('ignore')

import cv2
import logging
import os
import time
import sys
import zmq
from PIL import Image
from PredictorDarknet import DarknetYOLO
from ZeroMQ import ZeroMQDataHandler
from ZeroMQ import ZeroMQImageInput

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S')

LOGGER = logging.getLogger(__name__)
ROOT = os.path.dirname(os.path.realpath(__file__))

LOGGER.info("Finished Loading Imports")
LOGGER.info("Root path: %s" % ROOT)


LOGGER.info("Starting ZeroMQImageInput thread ...")
context = zmq.Context()
thread_image = ZeroMQImageInput(context);
thread_image.start()

LOGGER.info("Starting DarknetYOLO thread ...")
thread_yolo = DarknetYOLO(thread_image.image_data,
                          yolo_dir=ROOT + "/model/",
                          score_thresh=0.5,
                          fps=0.08)
thread_yolo.start()


LOGGER.info("Starting ZeroMQDataHandler thread ...")
thread_zeromqdatahandler = ZeroMQDataHandler(context, thread_yolo)
thread_zeromqdatahandler.start()
