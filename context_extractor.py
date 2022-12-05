import os
import sys
import time
from typing import List, Dict

import cv2

import zita


# Cars format: {frame: [cars: [x, y, w, h, certainty, class]]}
def save(run_id: str, path: str, frames: List, litter: List[zita.LitterEvent], cars: Dict[int, List[List[float]]]):
    # Create directory for run
    video_dir = "context_extractor/out/" + run_id + "/" + path
    os.mkdir(video_dir)

    # Save data to csv
    with open(video_dir + "/litter.csv", "w+") as f:
        f.write("event_id,frame,x1,y1,x2,y2,cert\n")
        for event in litter:
            event_id = litter.index(event)
            for det in event.detections:
                f.write(f"{event_id},{det[0]},{det[1][0]},{det[1][1]},{det[1][2]},{det[1][3]},{det[1][4]}\n")
        f.close()

    with open(video_dir + "/cars.csv", "w+") as f:
        f.write("frame,x1,y1,x2,y2,cert\n")
        for frame, c in cars.items():
            for car in c:
                f.write(f"{frame},{car[0]},{car[1]},{car[2]},{car[3]},{car[4]}\n")
        f.close()

    # Draw rects on and save frames
    for fi, frame in enumerate(frames):
        height, width, _ = frame.shape
        for event in litter:
            for det in event.detections:
                if det[0] == fi:
                    cv2.rectangle(frame, tuple(map(int, [det[1][0] * width, det[1][1] * height,
                                                         (det[1][2] - det[1][0]) * width,
                                                         (det[1][3] - det[1][1]) * height])), (255, 0, 0), 4)

        for car in cars[fi]:
            cv2.rectangle(frame, tuple(map(int, [car[0] * width, car[1] * height,
                                                 (car[2] - car[0]) * width,
                                                 (car[3] - car[1]) * height])), (0, 0, 255), 4)

        cv2.imwrite(f"{video_dir}/{fi}.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        config = zita.parse_args()
    else:
        config = zita.parse_args("zs pl12 --max-move 100 --batch-size-litter 32 --batch-size-cars 32".split(" "))

    litter_detector = zita.load_litter_model(config)
    car_detector = zita.load_car_model(config) if config.car_class is None else None

    run_id = str(int(time.time()))[2:]

    os.mkdir("results/context/" + run_id)

    for path in config.video:
        if not path.startswith("data/"):
            path = "data/" + path

        print("Processing: ", path)

        frames, video_length = zita.load_frames(path, config.fps)

        litter, cars = zita.detect(config, litter_detector, car_detector, frames)

        save(run_id, path[5:], frames, litter, cars)
