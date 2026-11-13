"""Fine-tune COCO-pretrained RF-DETR on person-cropped bib numbers."""
import argparse
from pathlib import Path

from model_utils import HERE, load_config, prepare_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE/'config.yaml')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--limit', type=int, help='Use only the first N images per split (quick check)')
    parser.add_argument('--epochs', type=int, help='Override train.epochs')
    parser.add_argument('--output-dir', type=Path, help='Override train.output_dir')
    args = parser.parse_args()
    config = load_config(args.config)
    data = prepare_dataset(config['dataset'], args.config.resolve().parent/'.rfdetr_dataset', args.limit)
    if args.prepare_only:
        return

    import rfdetr
    from detector import APACHE_MODELS

    size = str(config['model']).lower()
    if size not in APACHE_MODELS:
        raise ValueError(f'Unsupported RF-DETR model {size!r}; choose one of {sorted(APACHE_MODELS)}')
    train = dict(config['train'])
    if args.epochs:
        train['epochs'] = args.epochs
    if args.output_dir:
        train['output_dir'] = str(args.output_dir.resolve())
    Path(config['pretrained']).parent.mkdir(parents=True, exist_ok=True)
    # The COCO checkpoint downloads to `pretrained` if missing; the detection
    # head is adapted to the single bib-number class automatically.
    model = getattr(rfdetr, APACHE_MODELS[size])(pretrain_weights=config['pretrained'])
    model.train(
        dataset_dir=str(data),
        device='cpu' if str(config['device']) == 'cpu' else 'cuda',
        class_names=['bib-number'],
        seed=0,
        **train,
    )
    print(f'Checkpoints saved in {train["output_dir"]}')
    print('Run evaluate_model.py to evaluate val and test with the best checkpoint.')


if __name__ == '__main__':
    main()
