"""Evaluate a trained RF-DETR bib-number detector on cropped validation and test images."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from model_utils import HERE, SPLITS, load_config, prepare_dataset

sys.path.insert(0, str(HERE.parent))


def iou(box, boxes):
    """IoU of one xyxy box against an (N, 4) xyxy array."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area + areas - inter, 1e-9)


def confidence_sweep(detector, split_dir, iou_threshold=0.5):
    """Greedy IoU matching at a low threshold, then best-F1 confidence."""
    coco = json.loads((split_dir / '_annotations.coco.json').read_text())
    truths = {image['id']: [] for image in coco['images']}
    for annotation in coco['annotations']:
        x, y, w, h = annotation['bbox']
        truths[annotation['image_id']].append([x, y, x + w, y + h])
    scores, matches, total = [], [], sum(len(boxes) for boxes in truths.values())
    images = coco['images']
    for start in range(0, len(images), detector.batch_size):
        batch = images[start:start + detector.batch_size]
        crops = [cv2.imread(str(split_dir / image['file_name'])) for image in batch]
        for image, predictions in zip(batch, detector.predict(crops, confidence=0.01)):
            gt = np.asarray(truths[image['id']], dtype=np.float32).reshape(-1, 4)
            used = np.zeros(len(gt), dtype=bool)
            for box in predictions:  # already sorted by confidence
                scores.append(float(box[4]))
                if len(gt):
                    overlaps = iou(box[:4], gt)
                    overlaps[used] = 0
                    best = int(overlaps.argmax())
                    if overlaps[best] >= iou_threshold:
                        used[best] = True
                        matches.append(True)
                        continue
                matches.append(False)
    order = np.argsort(-np.asarray(scores))
    scores, matches = np.asarray(scores)[order], np.asarray(matches)[order]
    tp = np.cumsum(matches)
    precision = tp / np.arange(1, len(tp) + 1)
    recall = tp / max(total, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    best = int(f1.argmax())
    return {'based_on': 'val', 'iou': iou_threshold, 'confidence': float(scores[best]),
            'precision': float(precision[best]), 'recall': float(recall[best]),
            'f1': float(f1[best])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE/'config.yaml')
    parser.add_argument('--weights', type=Path, help='Override trained weights')
    parser.add_argument('--limit', type=int, help='Use only the first N images per split (quick check)')
    parser.add_argument('--output', type=Path, help='Report path (default: evaluation_stats.json)')
    args = parser.parse_args()
    config = load_config(args.config)
    weights = args.weights.resolve() if args.weights else Path(config['weights'])
    if not weights.is_file():
        raise FileNotFoundError(f'Trained weights not found: {weights}. Run train_model.py first.')

    import yaml
    from bib_detection.detector import BibDetector

    # BibDetector resolves `weights` relative to the config file, so give it
    # an absolute override.
    raw = yaml.safe_load(args.config.read_text())
    raw['weights'] = str(weights)
    override = HERE / '.evaluate_config.yaml'
    override.write_text(yaml.safe_dump(raw))
    try:
        detector = BibDetector(override)
    finally:
        override.unlink()
    if detector.model.class_names != ['bib-number']:
        raise ValueError(f'Expected single-class bib-number weights, got {detector.model.class_names}')

    data = prepare_dataset(config['dataset'], args.config.resolve().parent/'.rfdetr_dataset', args.limit)
    report = {'weights': str(weights), 'dataset': config['dataset'], 'model': config['model'],
              'resolution': detector.model.model_config.resolution, 'splits': {}}
    output = args.output or args.config.resolve().parent/'evaluation_stats.json'
    for split in ('val', 'test'):
        metrics = detector.model.evaluate(
            split=split, dataset_dir=str(data), device=detector.device,
            batch_size=config['train']['batch_size'], num_workers=config['train']['num_workers'],
            output_dir=str(Path(config['train']['output_dir']) / f'evaluate_{split}'),
        )
        report['splits'][split] = {key.split('/', 1)[-1]: float(value) for key, value in metrics.items()}
        if split == 'val':
            report['recommended_confidence'] = confidence_sweep(detector, data / SPLITS['val'])
        output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    print(f'Saved {output}')


if __name__ == '__main__':
    main()
