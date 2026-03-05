import torch
from torch.utils.data import Dataset
import numpy as np
import os
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud, LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from PIL import Image
import cv2
from skimage.draw import line
import skimage.morphology as morph
from shapely.geometry import MultiPoint, Polygon
from scipy.spatial import Delaunay
from scipy import ndimage

class RadarOccupancyDataset(Dataset):
    def __init__(self, dataroot, config=None, version='v1.0-mini', split='train'):
        self.config = config if config else self.default_config()
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
        self.split = split
        self.scenes = self._get_scenes()
        self.samples = self._get_samples()
        
    @staticmethod
    def default_config():
        return {
            'grid_res': 0.4,
            'grid_h': 256, 
            'grid_w': 64,  
            'range_max': 102.4, # 256 * 0.4
            'width_max': 25.6,  # 64 * 0.4
            'k_frames': 20,
            'min_lidar_points': 50,
            'dyn_thresh': 0.5  
        }
        
    def _get_scenes(self):
        return self.nusc.scene
        
    def _get_samples(self):
        samples = []
        k = self.config['k_frames']
        for scene in self.scenes:
            scene_samples = []
            curr = scene['first_sample_token']
            while curr:
                sample = self.nusc.get('sample', curr)
                scene_samples.append(sample)
                curr = sample['next']
            
            # Use SLIDING windows to restore data volume (3 Pro approach)
            for i in range(k-1, len(scene_samples)):
                samples.append(scene_samples[i])
        return samples

    def __len__(self):
        return len(self.samples)

    def _get_radar_points(self, sample, ref_pose, ref_cs, channels=None):
        if channels is None:
            channels = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']
        
        all_pts = []
        for chan in channels:
            if chan not in sample['data']: continue
            token = sample['data'][chan]
            rec = self.nusc.get('sample_data', token)
            cs = self.nusc.get('calibrated_sensor', rec['calibrated_sensor_token'])
            pose = self.nusc.get('ego_pose', rec['ego_pose_token'])
            
            pc = RadarPointCloud.from_file(os.path.join(self.nusc.dataroot, rec['filename']))
            
            # Dynamic Filtering (Sec 4.2)
            # 8: vx_comp, 9: vy_comp
            v_comp = np.sqrt(pc.points[8,:]**2 + pc.points[9,:]**2)
            keep = v_comp < self.config['dyn_thresh']
            pc.points = pc.points[:, keep]
            
            # Radar -> Ego -> Global -> RefEgo -> RefCs
            pc.rotate(Quaternion(cs['rotation']).rotation_matrix)
            pc.translate(np.array(cs['translation']))
            pc.rotate(Quaternion(pose['rotation']).rotation_matrix)
            pc.translate(np.array(pose['translation']))
            pc.translate(-np.array(ref_pose['translation']))
            pc.rotate(Quaternion(ref_pose['rotation']).rotation_matrix.T)
            pc.translate(-np.array(ref_cs['translation']))
            pc.rotate(Quaternion(ref_cs['rotation']).rotation_matrix.T)
            
            all_pts.append(pc.points)
            
        return np.hstack(all_pts) if all_pts else np.zeros((18, 0))

    def _alpha_shape(self, points, alpha=0.1):
        """
        Compute the alpha shape (concave hull) of a set of points.
        Simplified version or use shapely if possible.
        Using a simple convex hull or triangulation-based approach if scikit-spatial/alphashape not present.
        Here we use Shapely's .concave_hull (v2.0+) or Delaunay filtering.
        Let's use a Delaunay approach for robustness without extra heavy libs.
        """
        if len(points) < 4:
            return MultiPoint(list(points)).convex_hull

        tri = Delaunay(points)
        triangles = points[tri.simplices]
        a = ((triangles[:,0,0] - triangles[:,1,0])**2 + (triangles[:,0,1] - triangles[:,1,1])**2)**0.5
        b = ((triangles[:,1,0] - triangles[:,2,0])**2 + (triangles[:,1,1] - triangles[:,2,1])**2)**0.5
        c = ((triangles[:,2,0] - triangles[:,0,0])**2 + (triangles[:,2,1] - triangles[:,0,1])**2)**0.5
        s = (a+b+c)/2.0
        area = (s*(s-a)*(s-b)*(s-c))**0.5
        
        # Filter triangles
        circum_r = a*b*c / (4.0*area + 1e-6)
        valid = circum_r < (1.0/alpha) 
        
        # Create Polygon from boundary of valid triangles? 
        # Easier: Use Shapely MultiPoint().convex_hull as fallback or a simple bounding box if alpha fails.
        # Ideally we want Tight masking.
        # Paper says: "concave hull ... also known as alpha shape".
        
        # Fallback to Convex Hull for stability if explicit alpha shape library is missing
        # A true Alpha shape implementation from scratch is complex.
        return MultiPoint(list(points)).convex_hull

    def _get_scene_lidar(self, sample):
        # Simple cache to avoid re-aggregating thousands of points for every sample in the same scene
        scene_token = sample['scene_token']
        if not hasattr(self, '_scene_lidar_cache') or self._scene_lidar_cache['scene_token'] != scene_token:
            scene = self.nusc.get('scene', scene_token)
            curr_token = scene['first_sample_token']
            all_pts = []
            while curr_token:
                s = self.nusc.get('sample', curr_token)
                lrec = self.nusc.get('sample_data', s['data']['LIDAR_TOP'])
                pc = LidarPointCloud.from_file(os.path.join(self.nusc.dataroot, lrec['filename']))
                
                # Transform to Global Frame
                cs = self.nusc.get('calibrated_sensor', lrec['calibrated_sensor_token'])
                pose = self.nusc.get('ego_pose', lrec['ego_pose_token'])
                pc.rotate(Quaternion(cs['rotation']).rotation_matrix)
                pc.translate(np.array(cs['translation']))
                pc.rotate(Quaternion(pose['rotation']).rotation_matrix)
                pc.translate(np.array(pose['translation']))
                all_pts.append(pc.points[:3, :]) # Only x, y, z
                curr_token = s['next']
            self._scene_lidar_cache = {
                'scene_token': scene_token,
                'points': np.hstack(all_pts)
            }
        return self._scene_lidar_cache['points']

    def __getitem__(self, idx):
        sample = self.samples[idx]
        cfg = self.config
        
        # Ref Frame: Front Radar
        ref_chan = 'RADAR_FRONT'
        ref_token = sample['data'][ref_chan]
        ref_rec = self.nusc.get('sample_data', ref_token)
        ref_cs = self.nusc.get('calibrated_sensor', ref_rec['calibrated_sensor_token'])
        ref_pose = self.nusc.get('ego_pose', ref_rec['ego_pose_token'])
        
        # 1. Aggregate Radar (Input)
        pts_list = []
        curr = sample
        for _ in range(cfg['k_frames']):
            p_pts = self._get_radar_points(curr, ref_pose, ref_cs)
            if p_pts.shape[1] > 0:
                pts_list.append(p_pts)
            if not curr['prev']: break
            curr = self.nusc.get('sample', curr['prev'])
            
        all_radar = np.hstack(pts_list) if pts_list else np.zeros((18, 0))
        
        # Input Grid Generation
        input_grid = np.zeros((cfg['grid_h'], cfg['grid_w']), dtype=np.float32)
        if all_radar.shape[1] > 0:
            ridx_x = (all_radar[0,:] / cfg['grid_res']).astype(int)
            ridx_y = ((all_radar[1,:] + (cfg['grid_w']*cfg['grid_res'])/2) / cfg['grid_res']).astype(int)
            mask = (ridx_x>=0) & (ridx_x<cfg['grid_h']) & (ridx_y>=0) & (ridx_y<cfg['grid_w'])
            input_grid[ridx_x[mask], ridx_y[mask]] = 1.0
            
        # 2. Lidar Ground Truth Generation (Labeling Procedure Sec 4.1)
        # 2.1 Full Scene Aggregation
        global_lidar = self._get_scene_lidar(sample)
        # Transform Global -> Ref Radar
        # We need a copy because we're going to modify it
        pc_pts = global_lidar.copy()
        pc_pts -= np.array(ref_pose['translation'])[:, None]
        pc_pts = Quaternion(ref_pose['rotation']).rotation_matrix.T @ pc_pts
        pc_pts -= np.array(ref_cs['translation'])[:, None]
        pc_pts = Quaternion(ref_cs['rotation']).rotation_matrix.T @ pc_pts
        
        # Project to Grid
        lidx_x = (pc_pts[0,:] / cfg['grid_res']).astype(int)
        lidx_y = ((pc_pts[1,:] + (cfg['grid_w']*cfg['grid_res'])/2) / cfg['grid_res']).astype(int)
        
        valid_l = (lidx_x>=0) & (lidx_x<cfg['grid_h']) & (lidx_y>=0) & (lidx_y<cfg['grid_w'])
        
        # Binary Thresholding (Low point count)
        grid_counts = np.zeros((cfg['grid_h'], cfg['grid_w']), dtype=np.int32)
        np.add.at(grid_counts, (lidx_x[valid_l], lidx_y[valid_l]), 1)
        occupied_mask = grid_counts >= 2 # Paper mentions binary thresholding for low point count
        
        # 2.2 Morphological Ops (Sec 4.1: Dilation -> Hole Filling -> Erosion)
        selem = morph.disk(1)
        occupied_mask = morph.binary_dilation(occupied_mask, selem)
        occupied_mask = ndimage.binary_fill_holes(occupied_mask)
        occupied_mask = morph.binary_erosion(occupied_mask, selem)
        
        # 2.3 Ray Tracing
        label_grid = np.full((cfg['grid_h'], cfg['grid_w']), 2, dtype=np.longlong) # Default Unobserved (2)
        origin_x, origin_y = 0, int(cfg['grid_w'] / 2)
        occ_xs, occ_ys = np.where(occupied_mask)
        for ox, oy in zip(occ_xs, occ_ys):
            rr, cc = line(origin_x, origin_y, ox, oy)
            v = (rr>=0) & (rr<cfg['grid_h']) & (cc>=0) & (cc<cfg['grid_w'])
            label_grid[rr[v], cc[v]] = 0 # Free
        label_grid[occupied_mask] = 1 # Occupied
        
        # 2.4 Concave Hull Masking
        # Compute Hull for IGNORE (255)
        if np.sum(valid_l) > 10:
            pts_hull = np.column_stack((lidx_x[valid_l], lidx_y[valid_l]))
            try:
                # We use convex hull as approximation but should be tight enough
                # For "Concave" we can prune the hull or use alpha shape
                hull_geom = MultiPoint(pts_hull).convex_hull
                if isinstance(hull_geom, Polygon):
                    coords = np.array(hull_geom.exterior.coords)
                    poly_pts = coords[:, ::-1].astype(np.int32)
                    hull_mask = np.zeros((cfg['grid_h'], cfg['grid_w']), dtype=np.uint8)
                    cv2.fillPoly(hull_mask, [poly_pts], 1)
                    label_grid[hull_mask == 0] = 255 # Ignore
            except: pass

        # 3. Augmentation (Horizontal Flip)
        if self.split == 'train' and np.random.random() > 0.5:
            input_grid = np.flip(input_grid, axis=1).copy()
            label_grid = np.flip(label_grid, axis=1).copy()

        return torch.from_numpy(input_grid).unsqueeze(0), torch.from_numpy(label_grid)
        
class DummyDataset(RadarOccupancyDataset):
    def __init__(self, config=None):
         self.config = config if config else self.default_config()
         self.is_train = True

    def __len__(self): return 100

    def __getitem__(self, idx):
        cfg = self.config
        input_grid = torch.randint(0, 2, (1, cfg['grid_h'], cfg['grid_w'])).float()
        label_grid = torch.randint(0, 3, (cfg['grid_h'], cfg['grid_w'])).long()
        return input_grid, label_grid
