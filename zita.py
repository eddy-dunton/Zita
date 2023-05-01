#!/usr/bin/env python3

import argparse
import cv2
import itertools
import math
import os
import sys
import time
from dataclasses import dataclass

import cv2
import openalpr
import torch
import torchvision
import torchvision.transforms


class LitterEvent:
    # Stores the rectangles of this detection in the form:
    #    {frame number: [xyxyn]}
    detections: [(int, [float, float, float, float])]
    # Maximum (relative) size of this item of litter, in either x or y dimension
    max_size: int

    def __init__(self, frame, bbox):
        self.detections = [(frame, bbox)]
        self.max_size = max(*xyxy2wh(bbox))

    # Returns true or false
    # Considers whether a detection could be part of this event
    # If it is then it is added and the max size is potentially updated and True is return
    # Otherwise false is return
    # TODO Make this probabilistic and consider all events? rather than just adding a detection to the first event
    #  that is plausible
    def consider_detection(
            self, config: argparse.Namespace, frame_index: int,
            bbox: [float, float, float, float]) -> bool:
        # Iterate backwards through frames, more likely to find a close match later on
        for i, det in reversed(self.detections):
            # We're going backwards, so i is decreasing, if i is 1 second of frames away then stop
            if frame_index - i > config.fps * config.max_detection_gap:
                break

            if dist(bbox_centre(det), bbox_centre(bbox)) < config.max_movement * self.max_size:
                self.add(frame_index, bbox)
                return True

        # Nothing found, return false
        return False

    # Add a detection to this event, adds to dict and updates the max size
    def add(self, frame_index: int, bbox: [float, float, float, float]):
        if frame_index in self.detections:
            # TODO Not shit the bed?
            print("WARNING: Multiple detections attributed to a single event")

        self.detections.append((frame_index, bbox))

        # Update max size
        self.max_size = max(*xyxy2wh(bbox), self.max_size)

    def get_region(self) -> [float, float, float, float]:
        # TODO I think there's a smarter way to do this with zip, but I can't figure it out
        region = [1, 1, 0, 0]  # in format xyxyn
        for i, det in self.detections:
            region[0] = min(region[0], det[0])
            region[1] = min(region[1], det[1])
            region[2] = max(region[2], det[2])
            region[3] = max(region[3], det[3])
        return region

    # Returns start and end event time
    def get_time(self) -> (float, float):
        s_per_frame = 1.0 / config.fps
        start_seconds = self.detections[0][0] * s_per_frame
        end_seconds = self.detections[-1][0] * s_per_frame
        return start_seconds, end_seconds

    def __str__(self) -> str:
        format_region = ["{:.2f}".format(n) for n in self.get_region()[:4]]
        times = self.get_time()
        return "Detection event: @ {} between {:.2f} and {:.2f} seconds".format(format_region, times[0], times[1])


# Setup and init
# def extract_frames(video_path: str, fps: int) -> [torch.Tensor]:
# 	ms_per_frame = 1000.0 / fps
#
# 	out = []
# 	video = cv2.VideoCapture(video_path)
#
# 	frame = 0
# 	success, image = video.read()
# 	while success:
# 		out.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#
# 		frame += ms_per_frame
# 		video.set(cv2.CAP_PROP_POS_MSEC, frame)
#
# 		success, image = video.read()
#
# 	return out

def load_frames(path: str, fps: int) -> [torch.Tensor]:
    out = []
    skip = int(30 / fps)
    if not os.path.isdir(path):
        return out, -1

    frame_paths = os.listdir(path)

    crop = [0, 0, 1, 1]
    if "crop.txt" in frame_paths:
        frame_paths.remove("crop.txt")
        with open(os.path.join(path, "crop.txt")) as f:
            crop = [float(n) for n in f.read().split(",")]
            f.close()

    frame_paths.sort(key = lambda x: int(x.split(".")[0]))

    for frame_path in frame_paths[::skip]:
        raw = cv2.imread(os.path.join(path, frame_path))
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        # Just in case the images aren't all the same size
        crop_dimensions = [int(crop[0] * rgb.shape[1]), int(crop[1] * rgb.shape[0]),
                           int(crop[2] * rgb.shape[1]), int(crop[3] * rgb.shape[0])]
        cropped = rgb[crop_dimensions[1]:crop_dimensions[3], crop_dimensions[0]:crop_dimensions[2]]
        out.append(cropped)

    return out, len(out) / fps


def load_alpr() -> openalpr.Alpr:
    alpr = openalpr.Alpr("eu", "/etc/openalpr/openalpr.conf", "/usr/share/openalpr/runtime_data/")
    alpr.set_top_n(1)
    return alpr


def load_car_model(weights):
    model = torch.hub.load('ultralytics/yolov5', weights)
    model.classes = [2, 3, 5, 7]  # Filter only cars, motorcycles, buses and trucks
    return model


def load_litter_model(config: argparse.Namespace):
    conf = config.confidence
    if 0 <= conf >= 1:
        raise Exception("Invalid confidence level")

    weights = config.weights

    if not weights.endswith(".pt"):
        weights += ".pt"

    if not weights.startswith("weights/"):
        weights = "weights/" + weights

    # noinspection PyShadowingNames
    model = torch.hub.load('ultralytics/yolov5', 'custom', weights)
    model.conf = conf

    return model


def parse_args(args = None) -> argparse.Namespace:
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(description = "Detects littering motors from video")
    parser.add_argument("weights", help = "Weights to use for litter detection")
    parser.add_argument("video", nargs = "+", help = "Video to process")
    parser.add_argument(
        "-c", dest = "confidence", action = "store", default = 0.5, type = float,
        help = "Confidence threshold for the litter detector")
    parser.add_argument(
        "-f", dest = "fps", action = "store", default = 5, type = int,
        help = "Frames per second to evaluate")
    parser.add_argument("-v", dest = "evaluate", action = "store_true", help = "Evaluate performance")
    parser.add_argument(
        "-s", dest = "save", action = "store_true",
        help = "Save the results (to results/results/<weights>.csv")
    parser.add_argument(
        "-p", dest = "plates", action = "store_true",
        help = "Perform plate detection, please note this requires OpenALPR, which is a pain in the ass to install")
    parser.add_argument("--verbose", dest = "verbose", action = "store_true", help = "Print additional debugging data")
    parser.add_argument(
        "--cross-detection-threshold", dest = "cross_detection_threshold", action = "store",
        type = float,
        default = 0.8,
        help = "IoU threshold required to classify a joint car and litter detection as a car")
    parser.add_argument(
        "--max-litter-size", dest = "max_litter_size", action = "store", type = float,
        default = 0.2,
        help = "Max size (of a single dimension) of a litter detection before it is discard as an "
               "error, given as a proportion of the screen (0.0 - 1.0)")
    parser.add_argument(
        "--max-car-litter-intersection", dest = "max_car_litter_intersection", action = "store",
        type = float,
        default = 0.5,
        help = "Max intersection (across all detections) between litter and a car before it "
               "is considered a part of the car")
    parser.add_argument(
        "--max-detection-gap", dest = "max_detection_gap", action = "store", default = 1000,
        type = int,
        help = "Max time (seconds) between litter detections for them to be considered part of the same "
               "event")
    parser.add_argument(
        "--max-movement", dest = "max_movement", action = "store", default = 3, type = int,
        help = "Max movement (as a multiple of the event's maximum detected width or height) between "
               "litter detections for them to be consider part of the same event")
    parser.add_argument(
        "--motion-threshold", dest = "motion_threshold", action = "store", default = 1000, type = int,
        help = "Threshold for the motion detector, frames below the threshold are not considered")
    parser.add_argument(
        "--car-weights", dest = "car_weights", action = "store", default = "yolov5n6",
        help = "YOLOv5 weights used for car detection")
    parser.add_argument(
        "--tag", dest = "tag", action = "store", default = "",
        help = "Tag saved alongside results, does nothing if --save is not also passed")
    parser.add_argument(
        "--batch-size-litter", dest = "batch_size_litter", action = "store", default = 1,
        type = int,
        help = "Batch size for the litter detector")
    parser.add_argument(
        "--batch-size-cars", dest = "batch_size_car", action = "store", default = 1, type = int,
        help = "Batch size for the car detector")
    parser.add_argument(
        "--car-class", dest = "car_class", action = "store", default = None, type = int,
        help = "Class index of vehicle class if a unified detector is being used")

    return parser.parse_args(args)


# Metrics in format (avg score, avg score event, avg score no event, avg speedup)
def save(
        config: argparse.Namespace, scores_events: {str: (float, float)}, scores_no_events: {str: (float, float)},
        metrics: {str: float}):
    exists = os.path.exists("results/results.csv")

    run_id = str(int(time.time()))[2:]

    # Save general results
    with open("results.csv", "a+") as file:
        if not exists:
            # Write headers
            file.write(
                "Run Id,Video,Litter Model,Weights,Image Size,Score,Score (Events),Score (No Events),Speed,"
                "Frames Excluded,Conf,FPS,X-Det Threshold,Max Size,Max Intersection,Max Det Gap,Max Move,"
                "Motion Threshold,Car Weights,Tag,Version,\n")

        def write(val):
            file.write(str(val) + ",")

        def write_float(f):
            file.write("{:.3f},".format(f))

        write(run_id)
        write(";".join(config.video))
        write(config.litter_model)
        write(config.weights)
        write(config.image_size)

        write_float(metrics["score"])
        write_float(metrics["score_events"])
        write_float(metrics["score_no_events"])
        write_float(metrics["speed"])
        write_float(metrics["frames_excluded"])

        write(config.confidence)
        write(config.fps)
        write(config.cross_detection_threshold)
        write(config.max_litter_size)
        write(config.max_car_litter_intersection)
        write(config.max_detection_gap)
        write(config.max_movement)
        write(config.motion_threshold)

        if config.car_class:
            write("Unified")
        else:
            write(config.car_weights)

        write(config.tag), write(VERSION)
        file.write("\n")

    # Save specifics
    try:
        os.mkdir("results")
    except:
        pass

    with open("results/runs/{}.csv".format(run_id), "w+") as file:
        file.write("Data,Score,Speed,Length\n")
        for data, (score, speed, length) in scores_events.items():
            file.write("{},{},{},{}\n".format(data, score, speed, length))
        for data, (score, speed, length) in scores_no_events.items():
            file.write("{},{},{},{}\n".format(data, score, speed, length))


# Reads the ground truth from the
# noinspection PyTypeChecker
def load_truth(path: str) -> [[str, float, float]]:
    data = []

    with open(path, "r") as file:
        raw = file.readlines()
        file.close()

    for line in raw:
        split = line.split(",")
        split[1] = float(split[1])
        split[2] = float(split[2])
        data.append(split)

    return data


# Helper functions
def bbox_centre(bbox: [float, float, float, float]) -> [float, float]:
    return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]


def dist(a: [float, float], b: [float, float]) -> float:
    return math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2))


def xyxy2wh(bbox: [float, float, float, float]) -> [float, float]:
    return [bbox[2] - bbox[0], bbox[3] - bbox[1]]


def area(bbox: [float, float, float, float]) -> float:
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def intersection(a: [float, float, float, float], b: [float, float, float, float]) -> float:
    intersection_x1 = max(a[0], b[0])
    intersection_y1 = max(a[1], b[1])
    intersection_x2 = min(a[2], b[2])
    intersection_y2 = min(a[3], b[3])

    return max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)


def iou(a: [float, float, float, float], b: [float, float, float, float]) -> float:
    intersection_area = intersection(a, b)

    return intersection_area / float(area(a) + area(b) - intersection_area)


# Litter detection section

# Car detection event, not currently used
"""class CarEvent:
    # Stores the rectangles of this detection in the form:
    #    {frame number: [normalised xywh]}
    detections: [(int, [float])]  # TODO Figure out class
"""


# Determine the difference between frames, used to determine if there is motion in frame and therefore the frame should
# be considered
#
# Converts to grayscale and then applies a blur to reduce noise, then calculates the square difference
def frame_difference(frame1: torch.Tensor, frame2: torch.Tensor) -> float:
    # Convert to greyscale
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

    # Blur to reduce noise
    # TODO maybe tweak values
    frame1 = cv2.GaussianBlur(frame1, (43, 43), 0)
    frame2 = cv2.GaussianBlur(frame2, (43, 43), 0)

    # Return difference
    return int(cv2.norm(frame1, frame2, cv2.NORM_L2SQR))


def motion_filter_frames(config: argparse.Namespace, frames: [torch.Tensor]) -> [torch.Tensor]:
    filtered = []

    for fi in range(len(frames) - 1):
        if frame_difference(frames[fi], frames[fi + 1]) > config.motion_threshold:
            filtered.append(frames[fi])

    return filtered


# Thanks https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
def chunk(i, size) -> list:
    it = iter(i)
    return list(iter(lambda: list(itertools.islice(it, size)), []))


# noinspection PyShadowingNames
def detect(config: argparse.Namespace, litter_detector, car_detector, frames: [torch.Tensor]):
    start = time.time()

    if not config.car_class:
        # Break frames into batches
        car_batches = [frame for frame in chunk(frames, config.batch_size_car)]
        # Perform detection on batches
        car_detections_batches = [car_detector(batch).xyxyn for batch in car_batches]
        # Flatten batches back out into a flat list
        car_detections = list(itertools.chain.from_iterable(car_detections_batches))

    cd = time.time() - start

    # Break frames into batches
    litter_batches = [frame for frame in chunk(frames, config.batch_size_litter)]

    # Perform detection on batches
    # TODO return to old
    # litter_detections_batches = [litter_detector(batch).xyxyn for batch in litter_batches]
    # TODO remove
    # Strange new code, for some reason this doesn't work without this?
    litter_detections_batches = []
    i = 1
    for batch in litter_batches:
        print(f"Complete batch {i} of {len(litter_batches)}")
        i += 1
        litter_detections_batches.append(litter_detector(batch).xyxyn)

    # Flatten batches back out into a flat list
    litter_detections = list(itertools.chain.from_iterable(litter_detections_batches))

    ld = time.time() - start

    if config.car_class:
        car_class = config.car_class
        car_detections = []
        for i, frame in enumerate(litter_detections):
            litter_frame = frame.tolist()
            car_frame = []
            id = -1
            while id < len(litter_frame) - 1:
                id += 1
                det = litter_frame[id]
                if det[5] == car_class:
                    car_frame.append(det)
                    litter_frame.remove(det)
                    id -= 1
            litter_detections[i] = torch.Tensor(litter_frame)
            car_detections.append(torch.Tensor(car_frame))

    litter_events = []
    car_events = {}

    col = time.time() - start

    for i in range(len(frames)):
        # Get detections on the frame
        litter = litter_detections[i].tolist()
        cars = car_detections[i].tolist()
        car_events[i] = []

        # Each detection comes in the format [x1, y1, x2, y2, conf, class index] where xyxy are normalised
        # Here 0 in the index of the frame in the batch (but we only submit frames in a single batch, so 0)
        for litter_detection in litter:
            # Determine if it's a car or not
            car = False

            # Determine if it's actually a car
            for car_detection in cars:
                if iou(car_detection, litter_detection) > config.cross_detection_threshold:
                    # Actually a car
                    car = True
                    break

            if car:
                continue

            # If not a car

            # Heuristic filtering
            # Discard overly large detections, it isn't uncommon for the entire screen to be detected as litter
            if max(
                    litter_detection[2] - litter_detection[0],
                    litter_detection[3] - litter_detection[1]) > config.max_litter_size:
                continue

            collected = False
            for event in litter_events:
                if event.consider_detection(config, i, litter_detection):
                    collected = True
                    break

            # No event has claimed this detection, create a new one
            if not collected:
                litter_events.append(LitterEvent(i, litter_detection))

        # Similar process for cars
        for car_detection in cars:
            # For now cars are not dealt with the same way
            car_events[i].append(car_detection)
            """
            collected = False
            for event in car_events:
                if event.consider_detection(i, car_detection):
                    collected = True
                    break
            if not collected:
                car_events.append(CarEvent(i, car_detection))"""

    return litter_events, car_events


# Links car events and litter events together
def link(
        litter_events: [LitterEvent], car_events: {int: [[float, float, float, float]]}, alpr: openalpr.Alpr,
        frames: [torch.Tensor]) -> [(LitterEvent, str)]:
    out = []

    for event in litter_events:
        # Discard if always inside a cars bbox
        inside_car = True
        for litter_det in event.detections:
            litter_area = area(litter_det[1])

            # No cars in frame
            if not car_events[litter_det[0]]:
                inside_car = False
                break

            # Max intersection with any car over this frame
            inter = max(map(lambda x: intersection(litter_det[1], x) / litter_area, car_events[litter_det[0]]))
            if inter < config.max_car_litter_intersection:
                inside_car = False
                break

        if inside_car:
            continue

        # Find nearest car in first frame of litter
        litter_centre = bbox_centre(event.detections[0][1])
        closest = [1, None, -1]  # Distance, car, frame
        for frame, _ in event.detections:
            for car in car_events[frame]:
                distance = dist(bbox_centre(car), litter_centre)
                if distance < closest[0]:
                    closest[0] = distance
                    closest[1] = car
                    closest[2] = frame

        start = time.time()
        if closest[1] is not None:
            # Crop frame to car region
            frame = frames[closest[2]]
            frame_height, frame_width, _ = frame.shape
            # Denormalize car region
            car_region = [int(closest[1][0] * frame_width), int(closest[1][1] * frame_height),
                          int(closest[1][2] * frame_width), int(closest[1][3] * frame_height)]
            crop = (frame[car_region[1]:car_region[3], car_region[0]:car_region[2]])
            if alpr:
                alpr_results = alpr.recognize_ndarray(crop)["results"]
                if alpr_results:
                    out.append((event, alpr_results[0]["plate"]))

            else:
                out.append((event, "No plate found"))

        else:
            out.append((event, "No car found"))

    return out


@dataclass
class RunData:
    path: str
    attributions: [(LitterEvent, str)]
    detection_time: float
    attribution_time: float
    video_length: float
    frames_excluded: float

    # Convert to human readable format
    def to_readable(self):
        out = []
        for attr in self.attributions:
            out.append("{} linked to {}".format(str(attr[0]), attr[1]))

        out.append(
            "Found {} littering events in {:.2f} seconds of footage in {:.2f} seconds\n\n".format(
                len(self.attributions), self.video_length, self.detection_time + self.attribution_time))

        return out


def run(
        frames_path: str, config: argparse.Namespace, alpr: openalpr.Alpr, litter_detector, car_detector,
        verbose = True) -> [RunData]:
    # Set up p to print or not print depending on verbosity
    if verbose:
        def p(s: str):
            print(s)
    else:
        def p(_):
            pass

    p(frames_path)

    if not frames_path.startswith("data/"):
        frames_path = "data/" + frames_path

    frames, video_length = load_frames(frames_path, config.fps)

    p("Loaded {} frames".format(len(frames)))

    start = time.time()

    filtered_frames = motion_filter_frames(config, frames)

    p("Filtered {} motionless frames".format(len(frames) - len(filtered_frames)))

    litter, cars = detect(config, litter_detector, car_detector, filtered_frames)

    detection_time = time.time() - start

    attributions = link(litter, cars, alpr, filtered_frames)

    attribution_time = time.time() - start - detection_time

    data = RunData(frames_path, attributions, detection_time, attribution_time, video_length, -1.0)

    del frames, filtered_frames

    for line in data.to_readable():
        p(line)

    # if config.save:
    # 	save(config, attributions, (detection_time, attribution_time))

    return data


# Takes a 2 sets of segments
# Returns a the first set of segments - the intersection of the set of segments
def temporal_exclusivity(include: [[float, float]], exclude: [[float, float]]) -> [[float, float]]:
    for exc in exclude:
        # Skip weird incredibly short detections
        if abs(exc[0] - exc[1]) < 0.001:
            continue

        ii = -1
        # Necessary to move things about
        while ii < len(include) - 1:
            ii += 1
            inc = include[ii]

            # Exc overhangs inc start
            if inc[0] <= exc[1] <= inc[1]:
                # Exc entirely inside inc, split inc
                if inc[0] <= exc[0] <= inc[1]:
                    include.remove(inc)
                    ii -= 1  # Move index back as item is removed

                    include.append([inc[0], exc[0]])
                    include.append([exc[1], inc[1]])
                    continue

                # Move the start of the inc back
                inc[0] = exc[1]
                continue

            # Exc overhangs inc end, cannot be entirely inside (would have been caught earlier)
            if inc[0] <= exc[0] <= inc[1]:
                # Push end of inc forwards
                inc[1] = exc[0]
                continue

            # Exc encompasses inc
            if exc[0] <= inc[0] and exc[1] >= inc[1]:
                include.remove(inc)
                ii -= 1  # Move index back as item is removed

    #       else: detection and segment do not overlap

    return include


# Score is a measure of temporal correctness:
# The score is designed to judge how much time is saved, 0 is none, 1 is as much as possible
# All attributions are lumped together for this scoring
# If the video contains no events:
#   An event is detected: score = proportion of video not detected
#   An event is not detected: score = 1

# If the video contains events:
#   detections do not overlap: score = - proportion of video (incorrectly) detected
#   No events are detected:  score = 0
#   Detections do overlap: score = 1 - (proportion of video not including litter which is also included)

def score(truth: [[str, float, float]], run_data: [RunData]) -> (
        {str: (float, float, float)}, {str: (float, float, float)}):
    # Calculates the speed of the operation as ratio of processing time to video length
    def speed(r: RunData):
        return r.video_length / (r.detection_time + r.attribution_time)

    scores_events: {str: (float, float)} = {}
    scores_no_events: {str: (float, float)} = {}
    for run in run_data:
        run_truth = None
        # Find truth
        for t in truth:
            if t[0] == run.path:
                run_truth = t
                break

        if run_truth is None:
            print("Truth not found for {}".format(run.path))
            continue

        if run_truth[1] < 0:  # No events
            if not run.attributions:  # No detection
                scores_no_events[run.path] = (1, speed(run), run.video_length)
            else:  # There is a detection
                # Segments of video with no detections, start with the whole video
                segments_remaining = temporal_exclusivity(
                    [[0, run.video_length]],
                    [event.get_time() for event, _ in run.attributions])

                # Sum up length of remaining segments
                time_remaining = sum(map(lambda s: s[1] - s[0], segments_remaining))
                proportion_remaining = time_remaining / run.video_length
                # Clamp
                proportion_remaining = max(0.0, min(1.0, proportion_remaining))
                scores_no_events[run.path] = (proportion_remaining, speed(run), run.video_length)

            continue

        # There is an event
        if not run.attributions:  # No event detected
            scores_events[run.path] = (0, speed(run), run.video_length)
        else:  # Event detected
            segments_remaining = temporal_exclusivity(
                [[0, run_truth[1]], [run_truth[2], run.video_length]],
                [event.get_time() for event, _ in run.attributions])

            # Sum up length of remaining segments
            time_remaining = sum(map(lambda s: s[1] - s[0], segments_remaining))
            proportion_remaining = time_remaining / (run.video_length - (run_truth[2] - run_truth[1]))
            # Clamp
            proportion_remaining = max(-1.0, min(1.0, proportion_remaining))
            scores_events[run.path] = (proportion_remaining, speed(run), run.video_length)

    return scores_events, scores_no_events


if __name__ == '__main__':
    VERSION = "1.0.0-new-score"

    config = parse_args()

    print("Zita v{}\n".format(VERSION))
    print("Config: ")
    print(config)

    # Set up p to print or not print depending on verbosity
    if config.verbose:
        def p(s: str):
            print(s)
    else:
        def p(_):
            pass

    if config.plates:
        import openalpr

        alpr = load_alpr()
    else:
        alpr = None

    litter_detector = load_litter_model(config)
    car_detector = load_car_model(config.car_weights) if config.car_class is None else None

    print("\n")

    run_data = []

    for path in config.video:
        if os.path.isdir(path):
            run_data.append(run(path, config, alpr, litter_detector, car_detector))
        else:
            p(f"Error path: {path} not a directory: could it be the truth file?")

    if config.evaluate:
        truth = load_truth("data/truth.txt")

        scores_events, scores_no_events = score(truth, run_data)

        # Scores weighted by length (and length itself)
        wgt_score_events, length_events = 0, 0
        wgt_score_no_events, length_no_events = 0, 0
        wgt_speed = 0

        print("Videos with events:")
        for path, (score, speed, length) in scores_events.items():
            wgt_score_events += (score * length)
            wgt_speed += (speed * length)
            length_events += length
            print("{} (length {:.2f} seconds) scored {:.2f} at {:.2f}x speed".format(path, length, score, speed))

        print("\nVideos without events:")
        for path, (score, speed, length) in scores_no_events.items():
            wgt_score_no_events += (score * length)
            wgt_speed += (speed * length)
            length_no_events += length
            print("{} (length {:.2f} seconds) scored {:.2f} at {:.2f}x speed".format(path, length, score, speed))

        avg_score = (wgt_score_events + wgt_score_no_events) / (length_events + length_no_events)
        if scores_events:
            avg_score_events = wgt_score_events / length_events
        else:
            avg_score_events = 0
        if scores_no_events:
            avg_score_no_events = wgt_score_no_events / length_no_events
        else:
            avg_score_no_events = 0
        avg_speed = wgt_speed / (length_events + length_no_events)

        avg_frames_excluded = sum(map(lambda r: r.frames_excluded, run_data)) / len(run_data)

        print(
            "Average score: {:.2f} (events: {:.2f}, no events {:.2f}), average speed {:.2f}"
                .format(avg_score, avg_score_events, avg_score_no_events, avg_speed))

        if config.save:
            save(
                config, scores_events, scores_no_events,
                {
                    "score": avg_score,
                    "score_events": avg_score_events,
                    "score_no_events": avg_score_no_events,
                    "speed": avg_speed, "frames_excluded": avg_frames_excluded})
