"""Compare YOLO26 person coverage of bib labels, full images versus 2x2 tiles."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from collections import defaultdict

HERE = Path(__file__).resolve().parent
if (HERE / '.deps').is_dir():
    sys.path.insert(0, str(HERE / '.deps'))
os.environ.setdefault('YOLO_CONFIG_DIR', '/tmp/race-person-yolo')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/race-person-matplotlib')

import cv2
import torch
import yaml
from torchvision.ops import nms
from ultralytics import YOLO
import ultralytics


def tile_bounds(width, height, overlap):
    """Exactly four corner-anchored crops; overlap is a fraction of tile size."""
    tw, th = math.ceil(width / (2 - overlap)), math.ceil(height / (2 - overlap))
    return [(x, y, x + tw, y + th) for y in (0, height - th) for x in (0, width - tw)]


def contains(person, bib):
    x, y, w, h = bib
    return person[0] <= x and person[1] <= y and person[2] >= x+w and person[3] >= y+h


def load_split(root, split, limit):
    folder = root / split
    if split == 'val' and not folder.is_dir():
        folder = root / 'valid'
    data = json.loads((folder / '_annotations.coco.json').read_text())
    images = sorted(data['images'], key=lambda image: image['id'])
    if limit:
        images = images[:limit]
    categories = {c['id'] for c in data['categories'] if c['name'] in ('bib-number', 'bib', 'number')}
    if not categories:
        raise ValueError('No bib-number category in annotations')
    annotations = defaultdict(list)
    for annotation in data['annotations']:
        if annotation['category_id'] in categories:
            annotations[annotation['image_id']].append(annotation)
    return folder, images, annotations


def evaluate(model, folder, images, annotations, config, tiled):
    records, found, total, detections = [], 0, 0, 0
    annotated_images = all_found_images = 0
    group_size = max(1, config['batch'] // (4 if tiled else 1))
    started = time.perf_counter()
    last_progress = 0
    for start in range(0, len(images), group_size):
        group = images[start:start+group_size]
        sources, metadata = [], []
        for image in group:
            path = folder / image['file_name']
            frame = cv2.imread(str(path))
            if frame is None:
                raise ValueError(f'Cannot read image: {path}')
            height, width = frame.shape[:2]
            if (width, height) != (image['width'], image['height']):
                raise ValueError(f'Annotation dimensions differ from image: {path}')
            bounds = tile_bounds(width, height, config['tile_overlap']) if tiled else [(0, 0, width, height)]
            for x1, y1, x2, y2 in bounds:
                sources.append(frame[y1:y2, x1:x2].copy())
                metadata.append((image['id'], x1, y1))
        results = model.predict(sources, imgsz=config['imgsz'], batch=config['batch'],
                                rect=False, device=config['device'], classes=[0],
                                conf=config['confidence'], max_det=config['max_det'], verbose=False)
        per_image = defaultdict(list)
        for result, (image_id, x, y) in zip(results, metadata):
            boxes = result.boxes.data.cpu().clone()
            boxes[:, [0, 2]] += x
            boxes[:, [1, 3]] += y
            per_image[image_id].append(boxes)
        for image in group:
            boxes = torch.cat(per_image[image['id']])
            if tiled:
                boxes = boxes[nms(boxes[:, :4], boxes[:, 4], config['merge_iou'])]
            persons = boxes[:, :4].tolist()
            labels = annotations[image['id']]
            matched = [a['id'] for a in labels if any(contains(p, a['bbox']) for p in persons)]
            missing = [a['id'] for a in labels if a['id'] not in matched]
            found += len(matched)
            total += len(labels)
            detections += len(persons)
            if labels:
                annotated_images += 1
                all_found_images += not missing
            records.append({'image_id': image['id'], 'file_name': image['file_name'],
                            'bib_count': len(labels), 'found_bib_ids': matched, 'missed_bib_ids': missing,
                            'persons_xyxy_conf': boxes[:, :5].tolist()})
        completed = start + len(group)
        if completed - last_progress >= 100 or completed == len(images):
            print(f'{"tiles" if tiled else "full"}: {completed}/{len(images)} ({completed/(time.perf_counter()-started):.1f} images/s)', flush=True)
            last_progress = completed
    elapsed = time.perf_counter() - started
    return {'images': len(images), 'bib_numbers': total, 'found': found, 'missed': total-found,
            'bib_coverage_recall': found/total if total else None,
            'images_with_bibs': annotated_images, 'images_with_all_bibs_found': all_found_images,
            'all_bibs_image_rate': all_found_images/annotated_images if annotated_images else None,
            'person_detections': detections, 'elapsed_seconds': elapsed,
            'images_per_second': len(images)/elapsed, 'per_image': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE / 'config.yaml')
    parser.add_argument('--splits', nargs='+', choices=['train', 'val', 'test'])
    parser.add_argument('--max-images', type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.max_images is not None and args.max_images < 1:
        parser.error('--max-images must be positive')
    config = yaml.safe_load(args.config.read_text())
    if not 0 <= config['tile_overlap'] < 1:
        raise ValueError('tile_overlap must be in [0, 1)')
    base = args.config.resolve().parent
    dataset = (base / config['dataset']).resolve()
    weights = (base / config['weights']).resolve()
    output = args.output or (base / config['output'])
    weights.parent.mkdir(parents=True, exist_ok=True)
    if str(config['device']) != 'cpu' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    model = YOLO(str(weights))
    if model.names[0] != 'person':
        raise ValueError('Expected COCO pretrained model with class 0 person')
    report = {'weights': str(weights), 'dataset': str(dataset), 'ultralytics': ultralytics.__version__,
              'config': config, 'criterion': 'entire annotated bib-number box contained in any detected person box',
              'limitation': 'Bib coverage only: person precision cannot be measured without person annotations.',
              'splits': {}}
    print(f'Weights: {weights}\nDevice: {config["device"]}\nImage size: {config["imgsz"]}', flush=True)
    for split in args.splits or config['splits']:
        folder, images, annotations = load_split(dataset, split, args.max_images)
        report['splits'][split] = {}
        for mode, tiled in [('full_image', False), ('tiles_2x2_overlap20', True)]:
            print(f'\n{split}: {mode}', flush=True)
            stats = evaluate(model, folder, images, annotations, config, tiled)
            report['splits'][split][mode] = stats
            output.parent.mkdir(parents=True, exist_ok=True)
            temp = output.with_suffix(output.suffix + '.tmp')
            temp.write_text(json.dumps(report, indent=2) + '\n')
            temp.replace(output)
            print(f'Found {stats["found"]}/{stats["bib_numbers"]} = {stats["bib_coverage_recall"]:.2%}', flush=True)
    print(f'\nSaved {output}', flush=True)


if __name__ == '__main__':
    main()
