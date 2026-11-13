"""Config loading and an RF-DETR (Roboflow COCO layout) view of the crop dataset."""
import json
import os
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', '/tmp/race-bib-matplotlib')

import yaml

# Our dataset uses `val`; RF-DETR's Roboflow COCO layout expects `valid`.
SPLITS = {'train': 'train', 'val': 'valid', 'test': 'test'}


def load_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    for key in ('dataset', 'pretrained', 'weights'):
        config[key] = str((path.parent / config[key]).resolve())
    config['train']['output_dir'] = str((path.parent / config['train']['output_dir']).resolve())
    return config


def prepare_dataset(dataset, output, limit=None):
    """Validate COCO labels and link original images without copying or resizing.

    `limit` keeps only the first N images per split, for quick pipeline checks.
    """
    dataset, output = Path(dataset), Path(output)
    if output.exists():
        shutil.rmtree(output)
    for split, target in SPLITS.items():
        source = dataset / split
        coco = json.loads((source / '_annotations.coco.json').read_text())
        if len(coco['categories']) != 1 or coco['categories'][0]['name'] != 'bib-number':
            raise ValueError(f'Expected one bib-number category in {source}')
        category_id = coco['categories'][0]['id']
        images = sorted(coco['images'], key=lambda image: image['id'])
        if limit:
            images = images[:limit]
        sizes = {image['id']: (image['width'], image['height']) for image in images}
        annotations = []
        for annotation in coco['annotations']:
            if annotation['category_id'] != category_id:
                raise ValueError('Unexpected annotation category')
            if annotation['image_id'] not in sizes:
                continue
            iw, ih = sizes[annotation['image_id']]
            x, y, w, h = annotation['bbox']
            if not (iw > 0 and ih > 0 and w > 0 and h > 0 and x >= 0 and y >= 0
                    and x+w <= iw+1e-6 and y+h <= ih+1e-6):
                raise ValueError(f'Invalid bib box {annotation["id"]} in {source}')
            annotations.append(annotation)

        split_dir = output / target
        split_dir.mkdir(parents=True)
        names = set()
        for image in images:
            filename = Path(image['file_name'])
            if filename.name != str(filename) or filename.name in names:
                raise ValueError(f'Expected unique image basenames: {filename}')
            names.add(filename.name)
            image_path = source / filename
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            (split_dir / filename).symlink_to(image_path.resolve())
        (split_dir / '_annotations.coco.json').write_text(json.dumps({
            **{key: coco[key] for key in ('info', 'licenses') if key in coco},
            'categories': coco['categories'], 'images': images, 'annotations': annotations,
        }))
        print(f'{split}: {len(images):,} crops, {len(annotations):,} bib boxes', flush=True)
    return output
