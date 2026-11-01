"""Config loading and a generated YOLO label view of the COCO crop dataset."""
import json
import os
from collections import defaultdict
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
if (HERE / '.deps').is_dir():
    sys.path.insert(0, str(HERE / '.deps'))
os.environ.setdefault('YOLO_CONFIG_DIR', '/tmp/race-bib-yolo26')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/race-bib-matplotlib')

import yaml


def load_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    for key in ('dataset', 'pretrained', 'weights'):
        config[key] = str((path.parent / config[key]).resolve())
    config['train']['project'] = str((path.parent / config['train']['project']).resolve())
    return config


def prepare_dataset(dataset, output):
    """Generate YOLO labels; link original images without copying or resizing."""
    dataset, output = Path(dataset), Path(output)
    if output.exists():
        shutil.rmtree(output)
    for split in ('train', 'val', 'test'):
        source = dataset / split
        coco = json.loads((source / '_annotations.coco.json').read_text())
        if len(coco['categories']) != 1 or coco['categories'][0]['name'] != 'bib-number':
            raise ValueError(f'Expected one bib-number category in {source}')
        category_id = coco['categories'][0]['id']
        annotations = defaultdict(list)
        for annotation in coco['annotations']:
            if annotation['category_id'] != category_id:
                raise ValueError('Unexpected annotation category')
            annotations[annotation['image_id']].append(annotation)
        images_dir, labels_dir = output/'images'/split, output/'labels'/split
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        names = set()
        for image in coco['images']:
            filename = Path(image['file_name'])
            if filename.name != str(filename) or filename.stem in names:
                raise ValueError(f'Expected unique image basenames: {filename}')
            names.add(filename.stem)
            image_path = source / filename
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            (images_dir/filename).symlink_to(image_path.resolve())
            iw, ih = image['width'], image['height']
            lines = []
            for annotation in annotations[image['id']]:
                x, y, w, h = annotation['bbox']
                if not (iw > 0 and ih > 0 and w > 0 and h > 0 and x >= 0 and y >= 0 and x+w <= iw+1e-6 and y+h <= ih+1e-6):
                    raise ValueError(f'Invalid bib box in {image_path}')
                lines.append(f'0 {(x+w/2)/iw:.8f} {(y+h/2)/ih:.8f} {w/iw:.8f} {h/ih:.8f}')
            (labels_dir/filename.with_suffix('.txt')).write_text('\n'.join(lines)+'\n')
        print(f'{split}: {len(coco["images"]):,} crops, {len(coco["annotations"]):,} bib boxes', flush=True)
    data_path = output/'data.yaml'
    data_path.write_text(yaml.safe_dump({'path': str(output.resolve()), 'train': 'images/train',
                                        'val': 'images/val', 'test': 'images/test', 'names': {0: 'bib-number'}}))
    return data_path
