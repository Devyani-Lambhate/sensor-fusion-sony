import _init_path
import numpy as np
np.bool = np.bool_
np.int = np.int_
import argparse
from pathlib import Path


try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='NuScenes Demo for OpenPCDet')
    parser.add_argument('--cfg_file', type=str, default='cfgs/nuscenes_models/bevfusion.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='../data/nuscenes/v1.0-mini',
                        help='specify the root path to NuScenes dataset')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--save_dir', type=str, default=None, 
                        help='directory to save visualization screenshots (optional)')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------NuScenes Demo of OpenPCDet-------------------------')
    demo_dataset = NuScenesDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=args.data_path, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    
    # Create save directory if specified
    save_dir = None
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f'Visualization screenshots will be saved to: {save_dir}')
    
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            print(pred_dicts)

            fig = V.draw_scenes(
                points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )
            
            # Save screenshot if save_dir is specified
            if save_dir is not None and OPEN3D_FLAG is False:
                # For Mayavi, use mlab.savefig
                save_path = save_dir / f'sample_{idx:04d}.png'
                mlab.savefig(str(save_path))
                logger.info(f'Saved visualization to: {save_path}')

            if not OPEN3D_FLAG:
                mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
