#!/usr/bin/env python3
"""
Script to count trainable vs total parameters in the model.
"""
import _init_path
import argparse
from pathlib import Path

import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils


def freeze_layers(model, freeze_cfg, logger):
    """Freeze specific layers of the model based on configuration."""
    layer_map = {
        'FREEZE_BACKBONE_3D': ['backbone_3d'],
        'FREEZE_MAP_TO_BEV': ['map_to_bev'],
        'FREEZE_IMAGE_BACKBONE': ['image_backbone'],
        'FREEZE_NECK': ['neck'],
        'FREEZE_VTRANSFORM': ['vtransform'],
        'FREEZE_FUSER': ['fuser'],
        'FREEZE_BACKBONE_2D': ['backbone_2d'],
        'FREEZE_VFE': ['vfe']
    }
    
    frozen_modules = set()
    for freeze_key, module_names in layer_map.items():
        if freeze_cfg.get(freeze_key, False):
            for module_name in module_names:
                for name, param in model.named_parameters():
                    if module_name in name:
                        param.requires_grad = False
                        frozen_modules.add(module_name)
    
    if frozen_modules:
        logger.info(f"Frozen layers: {', '.join(sorted(frozen_modules))}")
        logger.info("Only DENSE_HEAD will be trained")
    
    return frozen_modules


def count_parameters(model):
    """Count trainable and total parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params
    }


def breakdown_by_layer(model):
    """Show parameter breakdown by major layer."""
    breakdown = {}
    for name, param in model.named_parameters():
        # Extract the top-level module name
        top_module = name.split('.')[0]
        if top_module not in breakdown:
            breakdown[top_module] = {'total': 0, 'trainable': 0}
        
        num_params = param.numel()
        breakdown[top_module]['total'] += num_params
        if param.requires_grad:
            breakdown[top_module]['trainable'] += num_params
    
    return breakdown


def main():
    parser = argparse.ArgumentParser(description='Count model parameters')
    parser.add_argument('--cfg_file', type=str, default='cfgs/nuscenes_models/bevfusion.yaml',
                        help='specify the config for the model')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to load')
    parser.add_argument('--freeze', action='store_true', default=False,
                        help='apply layer freezing based on config')
    
    args = parser.parse_args()
    
    # Load config
    cfg_from_yaml_file(args.cfg_file, cfg)
    logger = common_utils.create_logger()
    
    # Build a dummy dataset for model initialization
    train_set, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False, workers=0,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=False,
        total_epochs=1,
    )
    
    # Build model
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    model.cuda()
    model.eval()
    
    # Load checkpoint if provided
    if args.ckpt is not None:
        model.load_params_from_file(filename=args.ckpt, to_cpu=False, logger=logger)
    
    # Freeze layers if requested or if specified in config
    if args.freeze or hasattr(cfg.MODEL, 'FREEZE_LAYERS'):
        logger.info("Applying layer freezing...")
        freeze_cfg = cfg.MODEL.FREEZE_LAYERS if hasattr(cfg.MODEL, 'FREEZE_LAYERS') else {}
        freeze_layers(model, freeze_cfg, logger)
    
    # Count parameters
    params = count_parameters(model)
    
    print("\n" + "="*70)
    print("MODEL PARAMETER SUMMARY")
    print("="*70)
    print(f"Total Parameters:      {params['total']:>15,}")
    print(f"Trainable Parameters:  {params['trainable']:>15,}")
    print(f"Frozen Parameters:     {params['frozen']:>15,}")
    print(f"Trainable %:           {100.0 * params['trainable'] / params['total']:>14.2f}%")
    print("="*70)
    
    # Breakdown by layer
    breakdown = breakdown_by_layer(model)
    
    print("\nPARAMETER BREAKDOWN BY LAYER:")
    print("-"*70)
    print(f"{'Layer':<30} {'Total':>15} {'Trainable':>15}")
    print("-"*70)
    
    for layer_name in sorted(breakdown.keys()):
        stats = breakdown[layer_name]
        print(f"{layer_name:<30} {stats['total']:>15,} {stats['trainable']:>15,}")
    
    print("-"*70)
    print("\nLayers with all parameters trainable (will be updated during training):")
    for layer_name, stats in sorted(breakdown.items()):
        if stats['trainable'] == stats['total'] and stats['total'] > 0:
            print(f"  - {layer_name}: {stats['total']:,} parameters")
    
    print("\nLayers with all parameters frozen (will NOT be updated during training):")
    for layer_name, stats in sorted(breakdown.items()):
        if stats['trainable'] == 0 and stats['total'] > 0:
            print(f"  - {layer_name}: {stats['total']:,} parameters")


if __name__ == '__main__':
    main()
