"""Save original-resolution person crops and COCO bib-number annotations."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]


def contains(person, bib):
    x, y, w, h = bib
    return person[0] <= x and person[1] <= y and person[2] >= x+w and person[3] >= y+h


def create_crops(task):
    image, labels, prediction, source, destination = task
    persons = prediction['persons_xyxy_conf']
    selected, missed = set(), []
    for label in labels:
        candidates = [i for i, p in enumerate(persons) if contains(p, label['bbox'])]
        if candidates:
            best = min(candidates, key=lambda i: ((persons[i][2]-persons[i][0]) * (persons[i][3]-persons[i][1]), -persons[i][4]))
            selected.add(best)
        else:
            missed.append(label['id'])
    if not selected:
        return [], missed
    frame = cv2.imread(str(source / image['file_name']))
    if frame is None:
        raise ValueError(f'Cannot read {source / image["file_name"]}')
    height, width = frame.shape[:2]
    if (width, height) != (image['width'], image['height']):
        raise ValueError(f'Image and annotation dimensions differ: {image["file_name"]}')
    crops = []
    for index in sorted(selected):
        px1, py1, px2, py2, confidence = persons[index]
        x1, y1 = max(0, math.floor(px1)), max(0, math.floor(py1))
        x2, y2 = min(width, math.ceil(px2)), min(height, math.ceil(py2))
        if x2 <= x1 or y2 <= y1:
            raise ValueError('Empty person box')
        crop_labels = []
        for label in labels:
            bx, by, bw, bh = label['bbox']
            left, top = max(x1, bx), max(y1, by)
            right, bottom = min(x2, bx+bw), min(y2, by+bh)
            if right <= left or bottom <= top:
                continue
            clipped = [left-x1, top-y1, right-left, bottom-top]
            crop_labels.append({'category_id': 1, 'bbox': clipped,
                                'area': clipped[2]*clipped[3], 'iscrowd': label.get('iscrowd', 0),
                                'segmentation': [], 'source_annotation_id': label['id'],
                                'truncated': not contains([x1, y1, x2, y2], label['bbox'])})
        filename = f'image_{image["id"]:06d}_person_{index:03d}.png'
        crop = frame[y1:y2, x1:x2]
        if not cv2.imwrite(str(destination / filename), crop, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            raise OSError(f'Failed to write {filename}')
        crops.append(({'file_name': filename, 'width': x2-x1, 'height': y2-y1,
                       'source_image_id': image['id'], 'source_file_name': image['file_name'],
                       'source_crop_xyxy': [x1, y1, x2, y2], 'person_confidence': confidence}, crop_labels))
    return crops, missed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detections', type=Path, default=ROOT / 'person_detection/evaluation_results.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'datasets/person_cropped_images')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    report = json.loads(args.detections.read_text())
    source = Path(report['dataset'])
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError(f'Output must be empty: {args.output}')
    summary = {'source_dataset': str(source), 'detections': str(args.detections.resolve()),
               'weights': report['weights'], 'detection_mode': 'full_image',
               'resizing': False, 'image_format': 'lossless PNG',
               'selection': 'Smallest detected person box fully containing each labeled bib; deduplicate selected person boxes.',
               'partial_labels': 'Other intersecting bib boxes are clipped to crop boundaries and marked truncated.',
               'splits': {}}
    cv2.setNumThreads(1)
    for split in ('train', 'val', 'test'):
        folder = source / split
        if split == 'val' and not folder.is_dir():
            folder = source / 'valid'
        original = json.loads((folder / '_annotations.coco.json').read_text())
        bib_ids = {c['id'] for c in original['categories'] if c['name'] == 'bib-number'}
        if not bib_ids:
            raise ValueError('Expected bib-number category')
        labels = defaultdict(list)
        for annotation in original['annotations']:
            if annotation['category_id'] in bib_ids:
                labels[annotation['image_id']].append(annotation)
        predictions = {p['image_id']: p for p in report['splits'][split]['full_image']['per_image']}
        destination = args.output / split
        destination.mkdir()
        output = {'info': {'description': 'Original-resolution person crops with bib-number boxes'},
                  'licenses': original.get('licenses', []),
                  'categories': [{'id': 1, 'name': 'bib-number', 'supercategory': 'number-words'}],
                  'images': [], 'annotations': []}
        images = sorted(original['images'], key=lambda image: image['id'])
        for image in images:
            if image['id'] not in predictions or predictions[image['id']]['file_name'] != image['file_name']:
                raise ValueError(f'Missing or mismatched predictions: {split}/{image["file_name"]}')
        tasks = ((image, labels[image['id']], predictions[image['id']], folder, destination) for image in images)
        missed_ids, represented_sources = [], set()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for completed, (crops, missed) in enumerate(pool.map(create_crops, tasks), 1):
                missed_ids.extend(missed)
                for image_record, crop_labels in crops:
                    image_record['id'] = len(output['images']) + 1
                    represented_sources.add(image_record['source_image_id'])
                    output['images'].append(image_record)
                    for label in crop_labels:
                        label.update(id=len(output['annotations'])+1, image_id=image_record['id'])
                        output['annotations'].append(label)
                if completed % 100 == 0 or completed == len(images):
                    print(f'{split}: {completed}/{len(images)} source images, {len(output["images"])} crops', flush=True)
        (destination / '_annotations.coco.json').write_text(json.dumps(output, indent=2) + '\n')
        summary['splits'][split] = {
            'source_images': len(images), 'source_images_with_crops': len(represented_sources),
            'person_crops': len(output['images']), 'bib_annotations': len(output['annotations']),
            'truncated_annotations': sum(a['truncated'] for a in output['annotations']),
            'source_bib_numbers': sum(len(v) for v in labels.values()),
            'missed_source_bib_numbers': len(missed_ids), 'missed_source_annotation_ids': missed_ids}
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Saved dataset: {args.output}', flush=True)


if __name__ == '__main__':
    main()
