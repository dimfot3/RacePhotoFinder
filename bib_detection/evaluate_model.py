"""Evaluate a trained bib-number detector on cropped validation and test images."""
import argparse
import json
from pathlib import Path
from model_utils import HERE, load_config, prepare_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE/'config.yaml')
    parser.add_argument('--weights', type=Path, help='Override trained weights')
    args = parser.parse_args()
    config = load_config(args.config)
    weights = args.weights.resolve() if args.weights else Path(config['weights'])
    if not weights.is_file():
        raise FileNotFoundError(f'Trained weights not found: {weights}. Run train_model.py first.')
    from ultralytics import YOLO
    model = YOLO(str(weights))
    if model.names != {0: 'bib-number'}:
        raise ValueError('Expected trained single-class bib-number weights')
    data = prepare_dataset(config['dataset'], args.config.resolve().parent/'.yolo_dataset')
    report = {'weights': str(weights), 'dataset': config['dataset'], 'imgsz': config['imgsz'], 'splits': {}}
    output = args.config.resolve().parent/'evaluation_stats.json'
    for split in ('val', 'test'):
        metrics = model.val(data=str(data), split=split, imgsz=config['imgsz'], batch=config['batch'],
                            device=config['device'], workers=config['workers'], conf=0.001,
                            rect=False, plots=True, project=config['train']['project'],
                            name=f'evaluate_{split}')
        report['splits'][split] = {key: float(value) for key, value in metrics.results_dict.items()}
        if split == 'val':
            curves = getattr(metrics.box, 'f1_curve', None)
            report['recommended_confidence'] = None
            if curves is not None and curves.size and curves.max() > 0:
                index = int(curves.mean(axis=0).argmax())
                report['recommended_confidence'] = {
                    'based_on': 'val', 'confidence': float(metrics.box.px[index]),
                    'precision': float(metrics.box.p_curve[0, index]),
                    'recall': float(metrics.box.r_curve[0, index]),
                    'f1': float(curves[0, index])}
        output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    print(f'Saved {output}')


if __name__ == '__main__':
    main()
