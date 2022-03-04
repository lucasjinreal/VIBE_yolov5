from matplotlib.pyplot import box
from wanwu import Det, Backends
from alfred.utils.file_io import ImageSourceIter
import cv2
from wanwu.track.deepsort import DeepSort
from alfred.vis.image.det import (
    visualize_det_cv2_part,
)
import sys
import numpy as np


class MultiObjectTracker:
    def __init__(self) -> None:
        self.det = Det(
            type="yolov5s_coco",
            backend=Backends.AUTO,
            fp16=False,
            timing=True,
        )
        self.deepsort = DeepSort(
            Backends.AUTO, max_dist=0.4, max_iou_distance=0.8, max_age=20
        )

    def prepare_output_tracks(self, trackers):
        people = dict()
        for frame_idx, tracks in enumerate(trackers):
            for d in tracks:
                if d[5] == 0:
                    # select person class
                    person_id = int(d[4])
                    # bbox = np.array([d[0], d[1], d[2] - d[0], d[3] - d[1]]) # x1, y1, w, h

                    w, h = d[2] - d[0], d[3] - d[1]
                    c_x, c_y = d[0] + w / 2, d[1] + h / 2
                    w = h = np.where(w / h > 1, w, h)
                    bbox = np.array([c_x, c_y, w, h])

                    if person_id in people.keys():
                        people[person_id]["bbox"].append(bbox)
                        people[person_id]["frames"].append(frame_idx)
                    else:
                        people[person_id] = {
                            "bbox": [],
                            "frames": [],
                        }
                        people[person_id]["bbox"].append(bbox)
                        people[person_id]["frames"].append(frame_idx)
        for k in people.keys():
            people[k]["bbox"] = np.array(people[k]["bbox"]).reshape(
                (len(people[k]["bbox"]), 4)
            )
            people[k]["frames"] = np.array(people[k]["frames"])
        return people

    def track(self, data_f, show=False):
        """
        data_f can be video or images folder etc.
        """
        iter = ImageSourceIter(data_f, exit_auto=False)
        all_boxes = []

        i = 0
        while iter.ok:
            im = next(iter)
            if not im:
                break
            if isinstance(im, str):
                im = cv2.imread(im)

            inp = self.det.load_inputs(im, normalize_255=True, is_rgb=True)
            boxes, scores, labels = self.det.infer(inp, im.shape[0], im.shape[1])
            boxes, track_ids, labels = self.deepsort.update(boxes, scores, labels, im)

            if show:
                im = visualize_det_cv2_part(
                    im,
                    None,
                    labels,
                    boxes,
                    track_ids=track_ids,
                    font_scale=0.62,
                    transparent=False,
                )
                # only for debug
                cv2.imshow(f"Wanwu YOLOv5 tracker", im)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            # print(boxes)
            if len(boxes) > 0:
                boxes = np.insert(boxes, 4, track_ids, axis=1)
                boxes = np.insert(boxes, 5, labels, axis=1)
            all_boxes.append(boxes)
            i += 1

            print(f"\r{i}/{iter.lens}", end="", flush=True)
        res = self.prepare_output_tracks(all_boxes)
        return res
