"""Detect people, find bib numbers in their crops, and display bib boxes."""
import argparse
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', '/tmp/race-infer-matplotlib')

import cv2
import torch
from torchvision.ops import nms

from bib_detection.detector import BibDetector
from bib_ocr import BibOCRInferer
from person_detection.detector import PersonDetector


def detect_bibs(image, person_detector, bib_detector):
    """Return original-image [x1, y1, x2, y2, confidence] bib detections."""
    height, width = image.shape[:2]
    people = person_detector.predict([image])[0]
    crops, offsets = [], []
    for x1, y1, x2, y2 in people[:, :4].tolist():
        x1, y1 = max(0, math.floor(x1)), max(0, math.floor(y1))
        x2, y2 = min(width, math.ceil(x2)), min(height, math.ceil(y2))
        if x2 > x1 and y2 > y1:
            crops.append(image[y1:y2, x1:x2].copy())
            offsets.append((x1, y1))
    if not crops:
        return torch.empty((0, 5))
    boxes = []
    for result, (x, y) in zip(bib_detector.predict(crops), offsets):
        detections = torch.from_numpy(result)
        detections[:, [0, 2]] += x
        detections[:, [1, 3]] += y
        boxes.append(detections)
    boxes = torch.cat(boxes)
    return boxes[nms(boxes[:, :4], boxes[:, 4], iou_threshold=0.5)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--output', type=Path, help='Optionally save the annotated image')
    parser.add_argument('--no-show', action='store_true', help='Skip the image window')
    args = parser.parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        parser.error(f'Cannot read image: {args.image}')
    person_detector = PersonDetector()
    bib_detector = BibDetector()
    boxes = detect_bibs(image, person_detector, bib_detector)
    bib_crops = []
    for x1, y1, x2, y2, _ in boxes.tolist():
        bib_crops.append(image[max(0, math.floor(y1)):math.ceil(y2),
                               max(0, math.floor(x1)):math.ceil(x2)].copy())
    ocr_inferer = BibOCRInferer() if bib_crops else None
    ocr_results = ocr_inferer.infer(bib_crops) if ocr_inferer else []
    thickness = max(2, round(max(image.shape[:2]) / 1000))
    font_scale = max(0.6, max(image.shape[:2]) / 2400)
    accepted_count = 0
    for (x1, y1, x2, y2, confidence), ocr_result in zip(boxes.tolist(), ocr_results):
        if not ocr_inferer.is_valid(ocr_result):
            continue
        accepted_count += 1
        start, end = (round(x1), round(y1)), (round(x2), round(y2))
        cv2.rectangle(image, start, end, (0, 255, 0), thickness)
        cv2.putText(image, ocr_result.text, (start[0], max(25, start[1]-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
        print(f'Bib number: {ocr_result.text}')
    print(f'Found {accepted_count} bib number(s).')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), image):
            raise OSError(f'Cannot save image: {args.output}')
        print(f'Saved {args.output}')
    if not args.no_show:
        scale = min(1.0, 1600/image.shape[1], 1000/image.shape[0])
        preview = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image
        cv2.imshow('Bib detections - press any key to close', preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
