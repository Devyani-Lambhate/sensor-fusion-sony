import os
import torch
import copy
import numpy as np
from pcdet.utils import common_utils
from tqdm import tqdm
from pathlib import Path
from train_utils.train_utils import train_model  
from pcdet.models import load_data_to_gpu

def create_moving_blocks(dataset, block_length=8):
    """Create overlapping temporal blocks from time-series dataset"""
    blocks = []
    num_frames = len(dataset)
    for start in range(0, num_frames - block_length + 1):
        block = list(range(start, start + block_length))
        blocks.append(block)
    return blocks

def sample_blocks_with_replacement(blocks, M):
    """Sample M blocks with replacement"""
    indices = np.random.choice(len(blocks), size=M, replace=True)
    sampled = [blocks[i] for i in indices]
    return sampled

def create_bootstrap_dataset(original_dataset, sampled_blocks):
    """
    Create a bootstrap dataset containing only frames from the sampled blocks.
    
    Args:
        original_dataset: The original dataset object (e.g. NuScenesDataset, V2XSimDataset, etc.)
        sampled_blocks: List of lists, where each inner list contains frame indices in that block.
    
    Returns:
        A new dataset instance with filtered indices.
    """
    # Collect all unique frame indices from sampled blocks
    selected_indices = []
    for block in sampled_blocks:
        selected_indices.extend(block)
    
    selected_indices = sorted(set(selected_indices))  # remove duplicates, keep order
    
    # Create a new dataset with only selected frames
    bootstrap_dataset = copy.deepcopy(original_dataset)
    
    # Most OpenPCDet-style datasets have these attributes:
    if hasattr(bootstrap_dataset, 'frame_ids') or hasattr(bootstrap_dataset, 'infos'):
        # Filter infos / frame list
        if hasattr(bootstrap_dataset, 'infos'):
            bootstrap_dataset.infos = [bootstrap_dataset.infos[i] for i in selected_indices]
        
        if hasattr(bootstrap_dataset, 'frame_ids'):
            bootstrap_dataset.frame_ids = [bootstrap_dataset.frame_ids[i] for i in selected_indices]
            
        # Update length
        bootstrap_dataset.__len__ = lambda: len(selected_indices)
        
    elif hasattr(bootstrap_dataset, 'data_infos'):  # some datasets use this
        bootstrap_dataset.data_infos = [bootstrap_dataset.data_infos[i] for i in selected_indices]
    
    else:
        # Fallback: Try to override __getitem__ and __len__
        bootstrap_dataset.original_dataset = original_dataset
        bootstrap_dataset.selected_indices = selected_indices
        
        def new_getitem(self, idx):
            original_idx = self.selected_indices[idx]
            return self.original_dataset[original_idx]
        
        import types
        bootstrap_dataset.__getitem__ = types.MethodType(new_getitem, bootstrap_dataset)
        bootstrap_dataset.__len__ = lambda self: len(self.selected_indices)
    
    return bootstrap_dataset

def compute_average_aleatoric(aleatoric_list):
    """
    Compute average aleatoric uncertainty (Σ_a) across bootstrap runs.
    """
    if not aleatoric_list:
        return {}

    heads = aleatoric_list[0].keys()
    sigma_a_dict = {}

    for head in heads:
        values = [d[head] for d in aleatoric_list if head in d and isinstance(d[head], (int, float))]
        if values:
            mean_val = float(np.mean(values))
            sigma_a_dict[head] = mean_val
            sigma_a_dict[head + '_aleatoric_avg'] = mean_val
        else:
            sigma_a_dict[head] = 0.0

    print("\nAverage Aleatoric Scales (Σ_a):")
    for head, val in sigma_a_dict.items():
        if not head.endswith('_aleatoric_avg'):
            print(f"  {head:>8s}: {val:.4f}")

    return sigma_a_dict

# def moving_block_bootstrap_training(cfg, model, optimizer, train_loader, val_loader, 
#                                   model_func, lr_scheduler, optim_cfg, logger, 
#                                   N=15, l=8, device='cuda'):
#     """
#     Double-M Quantification: Moving Block Bootstrap
#     """
#     logger.info("=== Starting Double-M Bootstrap Training ===")
    
#     # 1. Initial training (already done usually)
#     model0 = model
    
#     blocks = create_moving_blocks(train_loader.dataset, block_length=l)
#     logger.info(f"Created {len(blocks)} overlapping blocks (l={l})")
    
#     residuals_list = []
#     aleatoric_list = []
    
#     for n in range(N):
#         logger.info(f"\n--- Bootstrap Iteration {n+1}/{N} ---")
        
#         # Sample blocks
#         sampled_blocks = sample_blocks_with_replacement(blocks, M=len(blocks)//l + 1)
        
#         # Create bootstrap dataset (you may need custom DataLoader logic)
#         bootstrap_dataset = create_bootstrap_dataset(train_loader.dataset, sampled_blocks)
#         bootstrap_loader = torch.utils.data.DataLoader(
#             bootstrap_dataset, batch_size=train_loader.batch_size,
#             shuffle=True, num_workers=train_loader.num_workers,
#             collate_fn=train_loader.collate_fn
#         )
        
#         # Train on bootstrap sample (continue from previous or reset?)
#         model_n = copy.deepcopy(model0) if n == 0 else model_n
#         optimizer_n = torch.optim.Adam(model_n.parameters(), lr=optim_cfg.LR)
        
#         # Train for fewer epochs (e.g. 5-10) to save time
#         train_model(
#             model=model_n,
#             optimizer=optimizer_n,
#             train_loader=bootstrap_loader,
#             model_func=model_func,
#             # ... other args
#             total_epochs=5,   # short training
#             logger=logger
#         )
        
#         # Evaluate on validation set
#         residuals, avg_aleatoric = evaluate_for_doublem(model_n, val_loader, device)
        
#         residuals_list.append(residuals)
#         aleatoric_list.append(avg_aleatoric)
        
#         torch.cuda.empty_cache()
    
#     # === Compute Σ_e (Epistemic) ===
#     sigma_e_dict = compute_epistemic_uncertainty(residuals_list)
    
#     # === Compute Σ_a (Average Aleatoric) ===
#     sigma_a_dict = compute_average_aleatoric(aleatoric_list)
    
#     # Save
#     save_path = cfg.ROOT_DIR / 'doublem_uncertainty.pth'
#     torch.save({
#         'sigma_e': sigma_e_dict,
#         'sigma_a': sigma_a_dict,
#         'N': N,
#         'block_length': l,
#         'residuals_stats': {k: v.mean().item() for k,v in residuals_list[0].items()}
#     }, save_path)
    
#     logger.info(f"Double-M uncertainties saved to {save_path}")
#     return sigma_e_dict, sigma_a_dict


def moving_block_bootstrap_training(cfg, model, optimizer, train_loader, val_loader, 
                                  model_func, lr_scheduler, optim_cfg, logger, 
                                  N=12, l=8, device='cuda'):
    """
    Full Double-M Quantification using Moving Block Bootstrap
    """
    logger.info("="*70)
    logger.info("Starting Double-M Quantification (Moving Block Bootstrap)")
    logger.info(f"Parameters: N={N}, block_length={l}")
    logger.info("="*70)

    model0 = model
    if hasattr(model, 'module'):  # DDP case
        model0 = model.module

    blocks = create_moving_blocks(train_loader.dataset, block_length=l)
    logger.info(f"Created {len(blocks)} temporal blocks")

    residuals_list = []
    aleatoric_list = []

    for n in range(N):
        logger.info(f"\n--- Bootstrap Run {n+1}/{N} ---")
        
        # Sample blocks
        sampled_blocks = sample_blocks_with_replacement(blocks, M=len(blocks)//l + 2)
        
        # Create bootstrap dataset
        bootstrap_dataset = create_bootstrap_dataset(train_loader.dataset, sampled_blocks)
        bootstrap_loader = torch.utils.data.DataLoader(
            bootstrap_dataset,
            batch_size=train_loader.batch_size,
            shuffle=True,
            num_workers=train_loader.num_workers,
            collate_fn=train_loader.collate_fn,
            pin_memory=True,
            drop_last=False
        )

        # Create fresh model copy from final trained model
        model_n = copy.deepcopy(model0)
        model_n.cuda()
        if hasattr(model, 'module'):
            model_n = torch.nn.parallel.DistributedDataParallel(model_n, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
        
        optimizer_n = torch.optim.Adam(model_n.parameters(), lr=optim_cfg.LR * 0.5)  # lower LR

        # Short training on bootstrap sample
        logger.info(f"Training bootstrap model for 3 epochs...")
        train_model(
            model=model_n,
            optimizer=optimizer_n,
            train_loader=bootstrap_loader,
            model_func=model_func,
            lr_scheduler=lr_scheduler,
            optim_cfg=optim_cfg,
            start_epoch=0,
            total_epochs=3,
            start_iter=0,
            rank=cfg.LOCAL_RANK if hasattr(cfg, 'LOCAL_RANK') else 0,
            tb_log=None,
            ckpt_save_dir=Path('/tmp/bootstrap_ckpts'),
            ckpt_save_interval=100,
            max_ckpt_save_num=1,
            logger=logger,
            logger_iter_interval=50,           # ← Critical fix
            ckpt_save_time_interval=300,
            use_logger_to_record=True,
            show_gpu_stat=False,
            use_amp=False,
            cfg=cfg,
            statistic=None,
        )

        # Evaluate for Double-M
        logger.info("Evaluating bootstrap model on validation set...")
        residuals, avg_aleatoric = evaluate_for_doublem(model_n, val_loader, device)
        
        residuals_list.append(residuals)
        aleatoric_list.append(avg_aleatoric)

        torch.cuda.empty_cache()

    # === Compute Σ_e and Σ_a ===
    sigma_e_dict = compute_epistemic_uncertainty(residuals_list)
    sigma_a_dict = compute_average_aleatoric(aleatoric_list)

    # Save
    save_path = Path(cfg.ROOT_DIR) / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / 'doublem_uncertainty.pth'
    print(f"Saving Double-M uncertainties to: {save_path}")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'sigma_e': sigma_e_dict,
        'sigma_a': sigma_a_dict,
        'N': N,
        'block_length': l,
    }, save_path)

    logger.info(f"Double-M uncertainties saved to: {save_path}")
    return sigma_e_dict, sigma_a_dict

def evaluate_for_doublem(model, val_loader, device=None):
    """
    Robust evaluation for Double-M Quantification.
    """
    model.eval()
    
    residuals = {name: [] for name in ['center', 'height', 'dim', 'rot', 'vel']}
    aleatoric_stats = {name: [] for name in ['center', 'height', 'dim', 'rot', 'vel']}
    
    total_samples = 0
    failed_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Double-M Evaluation")):
            
            load_data_to_gpu(batch)
            
            try:
                # === Safer Forward Pass ===
                if hasattr(model, 'predict'):                    # TransFusionHead has predict
                    pred_dicts = model.predict(batch['spatial_features_2d'])
                else:
                    # Standard model forward
                    output = model(batch)
                    if isinstance(output, dict):
                        pred_dicts = output
                    elif isinstance(output, (list, tuple)):
                        if len(output) >= 2:
                            pred_dicts = output[1] if isinstance(output[1], dict) else output[0]
                        else:
                            pred_dicts = output[0] if isinstance(output[0], dict) else None
                    else:
                        pred_dicts = getattr(model, 'forward_ret_dict', None)

                if pred_dicts is None or not isinstance(pred_dicts, dict):
                    failed_batches += 1
                    continue

                # Get final predictions if available
                if 'final_box_dicts' in pred_dicts:
                    pred_dicts = pred_dicts['final_box_dicts'][0] if isinstance(pred_dicts['final_box_dicts'], list) else pred_dicts['final_box_dicts']

            except Exception as e:
                print(f"Warning: Forward pass failed in batch {batch_idx}: {e}")
                failed_batches += 1
                continue

            # Get ground truth
            gt_boxes = batch.get('gt_boxes', None)
            if gt_boxes is None or gt_boxes.numel() == 0:
                continue

            # Get model head
            head = model.module if hasattr(model, 'module') else model

            if not hasattr(head, 'get_targets'):
                print("Warning: Model has no get_targets method.")
                continue

            try:
                gt_bboxes_3d = gt_boxes[..., :-1]
                gt_labels_3d = gt_boxes[..., -1].long() - 1
                
                # Get matched targets using Hungarian assigner
                labels, label_weights, bbox_targets, bbox_weights, \
                num_pos, matched_ious, heatmap = head.get_targets(
                    gt_bboxes_3d, gt_labels_3d, pred_dicts
                )
                
                # Build predictions tensor
                head_list = ['center', 'height', 'dim', 'rot', 'vel']
                preds_list = []
                for h in head_list:
                    if h in pred_dicts:
                        preds_list.append(pred_dicts[h].permute(0, 2, 1))
                
                if not preds_list:
                    continue
                    
                preds = torch.cat(preds_list, dim=-1)
                
                # Positive samples only
                pos_mask = (bbox_weights.sum(dim=-1) > 0).view(-1)
                if pos_mask.sum() == 0:
                    continue
                    
                matched_preds = preds.view(-1, preds.shape[-1])[pos_mask]
                matched_targets = bbox_targets.view(-1, bbox_targets.shape[-1])[pos_mask]
                
                # Process each head
                slices = [(0,2), (2,3), (3,6), (6,8), (8,10)]
                head_names = ['center', 'height', 'dim', 'rot', 'vel']
                
                for (start, end), hname in zip(slices, head_names):
                    if start >= matched_preds.shape[-1]:
                        continue
                    end = min(end, matched_preds.shape[-1])
                    
                    pred_slice = matched_preds[..., start:end]
                    target_slice = matched_targets[..., start:end]
                    
                    # Residuals for epistemic uncertainty
                    res = pred_slice - target_slice
                    residuals[hname].append(res.cpu())
                    
                    # Aleatoric uncertainty
                    scale_key = hname + '_scale'
                    if scale_key in pred_dicts:
                        scale = pred_dicts[scale_key].permute(0, 2, 1)
                        matched_scale = scale.reshape(-1, scale.shape[-1])[pos_mask][..., :(end-start)]
                        if matched_scale.numel() > 0:
                            aleatoric_stats[hname].append(matched_scale.mean(dim=0).cpu())
                
                total_samples += pos_mask.sum().item()
                
            except Exception as e:
                print(f"Warning: Target matching failed in batch {batch_idx}: {e}")
                failed_batches += 1
                continue

    # === Aggregate Results ===
    final_residuals = {}
    final_aleatoric = {}
    
    for head in residuals:
        if residuals[head]:
            final_residuals[head] = torch.cat(residuals[head], dim=0)
        else:
            c = 3 if head == 'center' else 1 if head == 'height' else 2
            final_residuals[head] = torch.zeros(0, c)
        
        if aleatoric_stats[head]:
            final_aleatoric[head] = torch.stack(aleatoric_stats[head]).mean(dim=0).item()
        else:
            final_aleatoric[head] = 0.0

    print(f"Double-M Evaluation completed. Total matched samples: {total_samples} | Failed batches: {failed_batches}")
    
    return final_residuals, final_aleatoric


def compute_epistemic_uncertainty(residuals_list):
    """
    Compute Epistemic Uncertainty (Σ_e) from residuals across all bootstrap runs.
    
    Args:
        residuals_list: List of dicts. Each dict contains residuals for each head.
                        Example: [{'center': tensor(...), 'height': tensor(...), ...}, ...]
    
    Returns:
        sigma_e_dict: Dict with covariance matrices (or variances) per head.
    """
    if not residuals_list:
        print("Warning: No residuals collected for epistemic uncertainty!")
        return {}

    sigma_e_dict = {}
    heads = residuals_list[0].keys()

    for head_name in heads:
        # Stack all residuals from different bootstraps: (N_bootstraps, N_samples, C)
        all_res_list = [res[head_name] for res in residuals_list if res[head_name].numel() > 0]
        
        if len(all_res_list) == 0:
            sigma_e_dict[head_name] = torch.zeros(1)  # fallback
            continue

        # Concatenate along sample dimension
        all_residuals = torch.cat(all_res_list, dim=0)  # (Total_samples, C)
        
        # Remove any NaNs or Infs
        all_residuals = torch.nan_to_num(all_residuals, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Compute covariance matrix (C x C)
        if all_residuals.shape[0] > 1:
            # Use torch.cov (requires PyTorch >= 1.9)
            cov_matrix = torch.cov(all_residuals.T)   # shape: (C, C)
        else:
            # Fallback: diagonal variance
            cov_matrix = torch.diag(torch.var(all_residuals, dim=0) + 1e-6)
        
        sigma_e_dict[head_name] = cov_matrix
        
        # Also store simple standard deviation for quick inspection
        std_dev = torch.sqrt(torch.diag(cov_matrix)).mean().item()
        print(f"  {head_name:>8s} → Epistemic std: {std_dev:.4f} | Cov shape: {cov_matrix.shape}")

    return sigma_e_dict
