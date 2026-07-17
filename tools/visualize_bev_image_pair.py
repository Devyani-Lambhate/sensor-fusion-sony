"""
BEVFusion Visualization — Camera 2D + LiDAR BEV (FULL DATASET SAVING)
FIXED: Now accumulates ALL images across the entire evaluation.
- clear_dir=False by default (no more deleting previous images)
- Cleaner filenames (timestamp only)
- Works perfectly even when called every batch
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import os

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], np.float32)

CLASS_COLORS = {
    "car":          "#00FFFF",
    "vehicle":      "#00FFFF",
    "pedestrian":   "#FF3333",
    "cyclist":      "#FFD700",
    "bicycle":      "#FFD700",
    "motorcycle":   "#FF8800",
    "truck":        "#AA44FF",
    "bus":          "#FF44FF",
    "traffic_cone": "#00FF88",
    "barrier":      "#888888",
}
DEFAULT_COLOR = "#FFFFFF"

def _class_color(name):
    return CLASS_COLORS.get(name.lower(), DEFAULT_COLOR)


def _recover_image(img_chw):
    img = img_chw.transpose(1,2,0).astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx > 10.0:
        return np.clip(img, 0, 255).astype(np.uint8)
    if mn >= -0.1 and mx <= 1.01:
        return (np.clip(img,0,1)*255).astype(np.uint8)
    denorm = img * _IMAGENET_STD[None,None,:] + _IMAGENET_MEAN[None,None,:]
    if denorm.min() > -0.3 and denorm.max() < 1.3:
        return (np.clip(denorm,0,1)*255).astype(np.uint8)
    out = np.zeros_like(img)
    for c in range(3):
        ch=img[:,:,c]; lo,hi=ch.min(),ch.max()
        out[:,:,c]=(ch-lo)/(hi-lo+1e-6)
    return (out*255).astype(np.uint8)


def _boxes_to_corners_3d(boxes):
    N=len(boxes)
    x,y,z   =boxes[:,0],boxes[:,1],boxes[:,2]
    dx,dy,dz=boxes[:,3],boxes[:,4],boxes[:,5]
    yaw     =boxes[:,6]
    ux=np.array([ 1, 1,-1,-1, 1, 1,-1,-1],np.float32)*0.5
    uy=np.array([ 1,-1,-1, 1, 1,-1,-1, 1],np.float32)*0.5
    uz=np.array([-1,-1,-1,-1, 1, 1, 1, 1],np.float32)*0.5
    corners=np.stack([ux[None]*dx[:,None],
                      uy[None]*dy[:,None],
                      uz[None]*dz[:,None]],axis=-1)
    c,s=np.cos(yaw),np.sin(yaw)
    R=np.zeros((N,3,3),np.float32)
    R[:,0,0]=c; R[:,0,1]=-s; R[:,1,0]=s; R[:,1,1]=c; R[:,2,2]=1.
    corners=np.einsum('nij,nkj->nki',R,corners)
    corners[:,:,0]+=x[:,None]
    corners[:,:,1]+=y[:,None]
    corners[:,:,2]+=z[:,None]
    return corners


def _corners_to_homo(corners3d):
    N=corners3d.shape[0]
    ones=np.ones((N,8,1),np.float32)
    return np.concatenate([corners3d,ones],axis=-1)


def _draw_bev_box(ax, box7, color, label=None):
    x,y,_,dx,dy,_,yaw=box7
    corners=np.array([[-dx/2,-dy/2],[dx/2,-dy/2],[dx/2,dy/2],[-dx/2,dy/2]])
    c,s=np.cos(yaw),np.sin(yaw)
    corners=corners@np.array([[c,-s],[s,c]]).T+[x,y]
    ax.add_patch(plt.Polygon(corners,closed=True,
                             edgecolor=color,facecolor='none',
                             linewidth=2.0,alpha=0.9))
    mid=(corners[0]+corners[1])/2
    ax.plot([x,mid[0]],[y,mid[1]],color=color,linewidth=1.2,alpha=0.8)
    if label:
        ax.text(x,y,label,color=color,fontsize=6,ha='center',va='center',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15',fc='black',alpha=0.55,lw=0))


# ─────────────────────────────────────────────────────────────────────────────
def visualize_detections(
    batch_dict,
    pred_dicts,
    class_names,
    sample_idx:   int   = 0,
    cam_idx:      int   = 0,
    score_thresh: float = 0.3,
    save_path           = None,
    show:         bool  = True,
    verbose:      bool  = False,
):
    img_t = batch_dict['camera_imgs'][sample_idx][cam_idx].cpu().numpy()
    img   = _recover_image(img_t)
    img_h, img_w = img.shape[:2]

    sp    = pred_dicts[sample_idx] if isinstance(pred_dicts,list) else pred_dicts
    boxes = sp['pred_boxes'].cpu().numpy()
    scores= sp['pred_scores'].cpu().numpy()
    labels= sp['pred_labels'].cpu().numpy()
    keep  = scores >= score_thresh
    boxes,scores,labels = boxes[keep],scores[keep],labels[keep]

    def get_name(l):
        idx=int(l)-1
        return class_names[idx] if 0<=idx<len(class_names) else f"cls{l}"

    meta=[dict(name=get_name(l), color=_class_color(get_name(l)), score=float(s))
          for l,s in zip(labels,scores)]

    LIDAR_AUG = batch_dict['lidar_aug_matrix'][sample_idx].cpu().numpy()
    L2I       = batch_dict['lidar2image'][sample_idx][cam_idx].cpu().numpy()
    IMG_AUG   = batch_dict['img_aug_matrix'][sample_idx][cam_idx].cpu().numpy()

    # ── project boxes (TWO-STEP) ─────────────────────────────────────────────
    bboxes_2d = []
    if len(boxes):
        corners3d = _boxes_to_corners_3d(boxes[:,:7])
        corners_h = _corners_to_homo(corners3d)

        proj_orig = (L2I @ LIDAR_AUG @ corners_h.reshape(-1,4).T).T
        depth = proj_orig[:,2].reshape(len(boxes), 8)
        px_orig = (proj_orig[:,0] / (proj_orig[:,2] + 1e-9)).reshape(len(boxes), 8)
        py_orig = (proj_orig[:,1] / (proj_orig[:,2] + 1e-9)).reshape(len(boxes), 8)

        sx, sy = IMG_AUG[0, 0], IMG_AUG[1, 1]
        tx, ty = IMG_AUG[0, 3], IMG_AUG[1, 3]
        px = sx * px_orig + tx
        py = sy * py_orig + ty

        for i in range(len(boxes)):
            vis = depth[i] > 0.1
            if not vis.any():
                bboxes_2d.append(None)
                continue
            x1 = float(np.clip(px[i][vis].min(), 0, img_w))
            x2 = float(np.clip(px[i][vis].max(), 0, img_w))
            y1 = float(np.clip(py[i][vis].min(), 0, img_h))
            y2 = float(np.clip(py[i][vis].max(), 0, img_h))
            if x2 - x1 < 2 or y2 - y1 < 2:
                bboxes_2d.append(None)
            else:
                bboxes_2d.append((x1, y1, x2, y2))

    visible = sum(1 for b in bboxes_2d if b is not None)

    # ── LiDAR points ─────────────────────────────────────────────────────────
    pts_all = batch_dict['points'].cpu().numpy()
    pts     = pts_all[pts_all[:,0]==sample_idx][:,1:]

    # ── figure ────────────────────────────────────────────────────────────────
    fig=plt.figure(figsize=(22,13),facecolor='#0a0a0a')
    ax1=fig.add_subplot(2,1,1)
    ax1.imshow(img)
    ax1.set_facecolor('#0a0a0a')

    if len(pts)>0:
        rng=np.random.default_rng(0)
        idx=rng.choice(len(pts), min(3000,len(pts)), replace=False)
        spts=pts[idx,:3]
        ones=np.ones((len(spts),1),np.float32)
        spts_h=np.concatenate([spts,ones], axis=1)
        proj_orig = (L2I @ LIDAR_AUG @ spts_h.T).T
        dm = proj_orig[:, 2] > 0.1
        if dm.any():
            px_orig = proj_orig[dm, 0] / proj_orig[dm, 2]
            py_orig = proj_orig[dm, 1] / proj_orig[dm, 2]
            sx, sy = IMG_AUG[0, 0], IMG_AUG[1, 1]
            tx, ty = IMG_AUG[0, 3], IMG_AUG[1, 3]
            pxp = sx * px_orig + tx
            pyp = sy * py_orig + ty
            in_im = (pxp >= 0) & (pxp <= img_w) & (pyp >= 0) & (pyp <= img_h)
            ax1.scatter(pxp[in_im], pyp[in_im], s=1.5, c='yellow', alpha=0.35, zorder=3)

    for i,box2d in enumerate(bboxes_2d):
        if box2d is None: continue
        x1,y1,x2,y2=box2d
        color=meta[i]['color']
        ax1.add_patch(mpatches.Rectangle((x1,y1),x2-x1,y2-y1,
            linewidth=2.2,edgecolor=color,facecolor='none',zorder=4))
        ax1.text(x1,max(y1-4,0),
                 f"{meta[i]['name']} {meta[i]['score']:.2f}",
                 color=color,fontsize=8,fontweight='bold',
                 va='bottom',ha='left',zorder=5,
                 bbox=dict(boxstyle='round,pad=0.2',fc='black',alpha=0.6,lw=0))

    ax1.set_title(f"Front Camera — Projected 2D Detections  ({visible} visible)",
                  fontsize=13,color='white',pad=6)
    ax1.axis('off')

    ax2=fig.add_subplot(2,1,2)
    ax2.set_facecolor('#111111')
    if len(pts):
        z=pts[:,2]
        sc=ax2.scatter(pts[:,0],pts[:,1],s=0.4,
                       c=np.clip(z,z.mean()-2*z.std(),z.mean()+2*z.std()),
                       cmap='plasma',alpha=0.55)
        plt.colorbar(sc,ax=ax2,label='Height (m)',fraction=0.02,pad=0.01)

    for i,box in enumerate(boxes):
        _draw_bev_box(ax2,box[:7],meta[i]['color'],label=meta[i]['name'][:3])

    ax2.plot(0,0,'w^',markersize=9,zorder=5)
    seen={}
    for m in meta:
        if m['name'] not in seen:
            seen[m['name']]=mpatches.Patch(color=m['color'],label=m['name'])
    if seen:
        ax2.legend(handles=list(seen.values()),loc='upper right',
                   facecolor='#1a1a1a',edgecolor='#555',labelcolor='white',fontsize=8)
    ax2.set_xlabel('X — forward (m)',color='#aaa')
    ax2.set_ylabel('Y — lateral (m)',color='#aaa')
    ax2.set_title(f"LiDAR BEV — 3D Predicted Boxes  ({len(boxes)} total)",
                  fontsize=13,color='white',pad=6)
    ax2.tick_params(colors='#777')
    for sp_ in ax2.spines.values(): sp_.set_edgecolor('#333')
    ax2.grid(True,color='#2a2a2a',linewidth=0.8)
    ax2.set_aspect('equal')

    plt.tight_layout(rect=[0,0,1,0.97])
    fig.suptitle(f"BEVFusion Detections  |  sample_idx={sample_idx}  |  cam={cam_idx}",
                 color='#cccccc',fontsize=14,y=0.99)

    if save_path:
        p=Path(save_path)
        p.parent.mkdir(parents=True,exist_ok=True)
        plt.savefig(p,dpi=300,bbox_inches='tight',facecolor=fig.get_facecolor())
        print(f"✅ Saved → {p.name}")
    elif show:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# FIXED: Save ALL samples across the entire evaluation (no more deleting)
# ─────────────────────────────────────────────────────────────────────────────
def visualize_all_samples(
    batch_dict,
    pred_dicts,
    class_names=None,
    save_dir: str = "vis",
    score_thresh: float = 0.3,
    cam_idx: int = 0,
    verbose: bool = False,
    clear_dir: bool = False,          # ← CHANGED TO FALSE BY DEFAULT
):
    """Save visualization for EVERY sample using unique frame_id"""
    if class_names is None:
        class_names = ['Car', 'Pedestrian', 'Cyclist']

    os.makedirs(save_dir, exist_ok=True)

    if clear_dir:
        import shutil
        if Path(save_dir).exists():
            shutil.rmtree(save_dir)
            os.makedirs(save_dir, exist_ok=True)
        print(f"🧹 Cleared old files in {save_dir}")

    batch_size = batch_dict.get('batch_size', len(batch_dict['camera_imgs']))
    frame_ids = batch_dict.get('frame_id', None)

    if frame_ids is not None:
        frame_ids = [str(f) for f in frame_ids]
    else:
        frame_ids = [f"sample_{i:06d}" for i in range(batch_size)]

    print(f"\n🔄 Saving {batch_size} samples to {save_dir}/")

    saved_count = 0
    for sample_idx in range(batch_size):
        # Cleaner filename: take the timestamp part only
        fid = frame_ids[sample_idx]
        if '__LIDAR_TOP__' in fid:
            clean_name = fid.split('__LIDAR_TOP__')[-1].replace('.pcd', '')
        else:
            clean_name = fid.replace('.pcd', '').replace('/', '_').replace(':', '_')
        
        save_path = f"{save_dir}/unc_sample_{clean_name}.png"
        
        print(f"   → {clean_name}")
        visualize_detections(
            batch_dict=batch_dict,
            pred_dicts=pred_dicts,
            class_names=class_names,
            sample_idx=sample_idx,
            cam_idx=cam_idx,
            score_thresh=score_thresh,
            save_path=save_path,
            show=False,
            verbose=verbose,
        )
        saved_count += 1

    print(f"✅ Finished batch! Saved {saved_count} images (total in folder keeps growing)\n")


# Backward compatibility
def visualize_image_bev_pair_with_2d_boxes(
    batch_dict, pred_dicts, class_names=None, sample_idx=0, save_path=None
):
    if class_names is None:
        class_names=['Car','Pedestrian','Cyclist']
    visualize_detections(batch_dict, pred_dicts, class_names,
                         sample_idx=sample_idx, save_path=save_path,
                         show=(save_path is None))


def debug_batch_keys(batch_dict):
    import torch
    print("\n── batch_dict keys ──────────────────────────────")
    for k,v in batch_dict.items():
        if isinstance(v,torch.Tensor):
            print(f"  {k:40s} Tensor {tuple(v.shape)}  {v.dtype}")
        elif isinstance(v,list):
            print(f"  {k:40s} list  len={len(v)}")
        else:
            print(f"  {k:40s} {type(v).__name__}")
    print("─────────────────────────────────────────────────\n")


if __name__=="__main__":
    print("Ready. Use visualize_all_samples()")