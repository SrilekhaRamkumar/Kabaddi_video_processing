# video_stream.py

import cv2
import time
from threading import Thread
from queue import Queue

class VideoStream:
    def __init__(self, path, queue_size=5):
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        self.queue = Queue(maxsize=queue_size)

    def start(self):
        t = Thread(target=self.update)
        t.daemon = True
        t.start()
        return self

    def update(self):
        while not self.stopped:
            if not self.queue.full():
                ret, frame = self.stream.read()
                if not ret:
                    self.stopped = True
                    return
                self.queue.put(frame)
            else:
                time.sleep(0.001)

    def read(self):
        return self.queue.get() if not self.queue.empty() else None

    def running(self):
        return not self.stopped or not self.queue.empty()
