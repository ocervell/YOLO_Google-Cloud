import argparse
import asyncio
import base64
import cv2
import json
import logging
import numpy as np
import os
import random
import threading
import ssl
import sys
import zmq

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder
from av import VideoFrame

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S')

ROOT = os.path.dirname(__file__)
LOGGER = logging.getLogger(__name__)
DEBUG = True
PORT = os.environ.get("YOLO_PORT", 8889)

context = zmq.Context()
data_socket_send = context.socket(zmq.PUB)
data_socket_send.connect('tcp://localhost:5556')
LOGGER.info("Connected to websocket 'tcp://localhost:5556'.")

detectionData = DetectionDataHolder()
detectionData.start()
LOGGER.info("Detection data thread started ...")

class VideoTransformTrack(VideoStreamTrack):
    def __init__(self, track):
        super().__init__()
        self.counter = 0
        self.track = track
        self.footage_socket = context.socket(zmq.PUB)
        self.footage_socket.connect('tcp://localhost:5555')


    async def recv(self):
        frame = await self.track.recv()
        try:
            # Send via MQTT
            img = frame.to_ndarray(format='bgr24')
            encoded, buffer = cv2.imencode('.jpg', img)
            # Send Webcam stream from HTTP Server -> ML Server
            self.footage_socket.send(base64.b64encode(buffer))
            return frame
        except Exception as e:
            print("An error occured sending over MQTT: " + str(e))
            return frame

class DetectionDataHolder(threading.Thread):
    def __init__(self):
        """
        Receives bounding box, class and scores data from ML Server.
        """
        threading.Thread.__init__(self)
        self.name = 'ZeroMQ DataHandler'
        self.done = False
        self.data = "{}"
        LOGGER.info("Connecting to %s" % 'tcp://*:5557 ...')
        self.data_socket_rcv = context.socket(zmq.SUB)
        self.data_socket_rcv.bind('tcp://*:5557')
        self.data_socket_rcv.setsockopt_string(zmq.SUBSCRIBE, str(''))
        LOGGER.info("Connected to %s" % 'tcp://*:5557 ...')

    def run(self):
        LOGGER.info("Starting " + self.name)
        self.update(self.name)
        LOGGER.info("Exiting " + self.name)

    def update(self, threadName):
        while not self.done:
            try:
                self.data = self.data_socket_rcv.recv_string()
                LOGGER.info("Received %s from socket" % self.data)
            except:
                LOGGER.error("Error occured receiving data on ML client")

    def stop(self):
        self.done = True

async def index(request):
    filepath = os.path.join(ROOT + '/public/', 'index.html')
    LOGGER.info("index: new request for %s" % filepath)
    with open(filepath, 'r') as f:
        content = f.read()
    return web.Response(content_type='text/html', text=content)

async def javascript(request):
    filepath = os.path.join(ROOT + '/public/', 'client.js')
    LOGGER.info("javascript: new request for %s" % filepath)
    with open(filepath, 'r') as f:
        content = f.read()
    return web.Response(content_type='application/javascript', text=content)

async def offer(request):
    LOGGER.info("offer: new request")
    params = await request.json()
    LOGGER.debug("offer: request params: %s" % params)
    offer = RTCSessionDescription(
        sdp=params['sdp'],
        type=params['type'])
    pc = RTCPeerConnection()
    pcs.add(pc)

    LOGGER.info("Established RTCPeerConnection %s" % pc)

    # prepare local media
    recorder = MediaBlackhole()

    @pc.on('datachannel')
    def on_datachannel(channel):
        @channel.on('message')
        def on_message(message):
            try:
                # Send data to ML Server. Currently only sends image height and width.
                LOGGER.info("Received message, sending to ML server: %s" % message)
                data_socket_send.send_string(message)
                LOGGER.debug("Message to browser: " + str(detectionData.data))
                channel.send(detectionData.data)
            except:
                LOGGER.error("Failed receiving module data.")
                channel.send("{}")

    @pc.on('iceconnectionstatechange')
    async def on_iceconnectionstatechange():
        print('ICE connection state is %s' % pc.iceConnectionState)
        if pc.iceConnectionState == 'failed':
            await pc.close()
            pcs.discard(pc)

    @pc.on('track')
    def on_track(track):
        # Add the CNN Video
        LOGGER.info('Track %s received' % track.kind)
        if track.kind == 'video':
            local_video = VideoTransformTrack(track)
            pc.addTrack(local_video)
            LOGGER.info("Added local video (cnn).")

        @track.on('ended')
        async def on_ended():
            LOGGER.info('Track %s ended' % track.kind)
            await recorder.stop()

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type='application/json',
        text=json.dumps({
            'sdp': pc.localDescription.sdp,
            'type': pc.localDescription.type
        }))

pcs = set();

async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

def run():
    LOGGER.info("Baking app. ..")
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get('/', index)
    app.router.add_get('/client.js', javascript)
    app.router.add_post('/offer', offer)
    app.router.add_static('/static/', ROOT + '/public/static/', name='static', show_index=True)

    # SSL context
    ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_ctx.load_cert_chain(ROOT + "/ssl/domain.crt",
                            ROOT + "/ssl/domain.key")

    LOGGER.info("Starting app on PORT %s ..." % PORT)
    web.run_app(app, port=PORT, ssl_context=ssl_ctx)

if __name__ == '__main__':
    run()
