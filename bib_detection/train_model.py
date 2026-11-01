"""Fine-tune COCO-pretrained YOLO26s on person-cropped bib numbers."""
import argparse
from pathlib import Path
from model_utils import HERE, load_config, prepare_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE/'config.yaml')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    data = prepare_dataset(config['dataset'], args.config.resolve().parent/'.yolo_dataset')
    if args.prepare_only:
        return
    from ultralytics import YOLO
    Path(config['pretrained']).parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(config['pretrained'])
    model.train(data=str(data), imgsz=config['imgsz'], batch=config['batch'], device=config['device'],
                workers=config['workers'], cache=False, seed=0, **config['train'])
    print(f'Best weights: {model.trainer.best}')
    print('Run evaluate_model.py to evaluate val and test with these weights.')


if __name__ == '__main__':
    main()
