import numpy as np
import math

class OccupancyGridMap:
    def __init__(self, config):
        self.cfg = config
        self.grid_h = config['grid_h']
        self.grid_w = config['grid_w']
        self.res = config['grid_res']
        # Log odds map
        # Prior = 0.5 -> log(0.5/0.5) = 0
        self.log_odds = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        
        # Probabilities
        self.p_occ = 0.7  # P(m=occ | z=hit)
        self.p_free = 0.4 # P(m=occ | z=miss) - Paper says "low probability"
        self.p_prior = 0.5
        
        # Log odds increments
        self.l_occ = math.log(self.p_occ / (1 - self.p_occ))
        self.l_free = math.log(self.p_free / (1 - self.p_free))
        self.l_prior = math.log(self.p_prior / (1 - self.p_prior)) # 0

    def update(self, radar_points, method='delta'):
        """
        Update grid with new radar observation
        radar_points: (2, N) array of (x, y) in Grid Frame
        """
        # Create ISM for this observation
        if method == 'delta':
            inv_model_log_odds = self._delta_ism(radar_points)
        elif method == 'gaussian':
            inv_model_log_odds = self._gaussian_ism(radar_points)
        else:
            raise ValueError(f"Unknown method {method}")
            
        # Bayesian Update: l_t = l_{t-1} + l_inv - l_0
        self.log_odds += inv_model_log_odds - self.l_prior
        
        # Clamping to avoid overflow (optional, but good practice)
        self.log_odds = np.clip(self.log_odds, -100, 100)

    def get_map(self):
        # Convert log odds to probability
        # p = 1 - 1 / (1 + exp(l))
        exp_l = np.exp(self.log_odds)
        p = exp_l / (1 + exp_l)
        
        # Thresholding
        # Occupied > 0.6 ? 
        # Paper: "best thresholds used to determine... occupied, free and unobserved"
        # Let's pick standard thresholds
        res = np.zeros_like(p, dtype=int)
        res[p > 0.6] = 1 # Occupied
        res[p < 0.4] = 0 # Free
        res[(p >= 0.4) & (p <= 0.6)] = 2 # Unobserved
        return res

    def _delta_ism(self, points):
        """
        Delta ISM:
        - Hit cells = l_occ
        - Ray cells = l_free
        - Else = l_prior (0)
        """
        update_grid = np.zeros_like(self.log_odds) # Default 0 (prior)
        
        if points.shape[1] == 0:
            return update_grid
            
        # 1. Map points to grid indices
        x_idxs = (points[0, :] / self.res).astype(int)
        y_idxs = ((points[1, :] + (self.grid_w * self.res) / 2) / self.res).astype(int)
        
        # Filter valid
        valid = (x_idxs >= 0) & (x_idxs < self.grid_h) & (y_idxs >= 0) & (y_idxs < self.grid_w)
        x_idxs = x_idxs[valid]
        y_idxs = y_idxs[valid]
        
        # 2. Mark Hits
        # Note: In Delta function, checking ray tracing for free space is implied?
        # Paper says: "We use two ISM functions... Delta... Gaussian".
        # Usually ISM includes the free space carving.
        # "Ray tracing" section is separate.
        # But Bayesian filtering NEEDS free space model, otherwise it only accumulates occupancy.
        # Let's implement simplified Ray Tracing for the Free space part of ISM.
        # (Similar to what we did in dataset.py, but on the update_grid)
        
        # Ideally we use Bresenham again.
        # Origin
        ox, oy = 0, int(self.grid_w / 2)
        
        # Mark Free (using simple line drawing)
        # This is slow in Python. For a baseline, it's acceptable.
        from skimage.draw import line
        
        for tx, ty in zip(x_idxs, y_idxs):
             rr, cc = line(ox, oy, tx, ty)
             # Clip
             v = (rr >= 0) & (rr < self.grid_h) & (cc >= 0) & (cc < self.grid_w)
             rr, cc = rr[v], cc[v]
             update_grid[rr, cc] = self.l_free
             
        # Mark Occupied (overwrites free at the hit)
        update_grid[x_idxs, y_idxs] = self.l_occ
        
        return update_grid
        
    def _gaussian_ism(self, points):
        """
        Gaussian ISM:
        - Free space logic similar to Delta (Ray tracing).
        - Hit occupancy probability decays spatially as a Gaussian around the detection point.
        """
        if points.shape[1] == 0:
            return np.zeros_like(self.log_odds)
            
        # 1. Delta logic for Free space
        update_grid = np.zeros_like(self.log_odds)
        x_idxs = (points[0, :] / self.res).astype(int)
        y_idxs = ((points[1, :] + (self.grid_w * self.res) / 2) / self.res).astype(int)
        valid = (x_idxs >= 0) & (x_idxs < self.grid_h) & (y_idxs >= 0) & (y_idxs < self.grid_w)
        x_idxs, y_idxs = x_idxs[valid], y_idxs[valid]
        
        from skimage.draw import line
        ox, oy = 0, int(self.grid_w / 2)
        for tx, ty in zip(x_idxs, y_idxs):
             rr, cc = line(ox, oy, tx, ty)
             v = (rr >= 0) & (rr < self.grid_h) & (cc >= 0) & (cc < self.grid_w)
             update_grid[rr[v], cc[v]] = self.l_free
             
        # 2. Gaussian spread for Hits
        hit_grid = np.zeros_like(self.log_odds)
        hit_grid[x_idxs, y_idxs] = 1.0
        
        from scipy.ndimage import gaussian_filter
        # Sigma of 1.0 cell = 0.4m which is reasonable for radar uncertainty
        blurred_hits = gaussian_filter(hit_grid, sigma=1.0)
        
        # Scale to match l_occ at peaks
        # Max of blurred hits will be < 1.0, let's normalize
        if blurred_hits.max() > 0:
            blurred_hits = blurred_hits / blurred_hits.max()
            
        # Add to update_grid
        # Where blurred hits are significant, override free space
        hit_mask = blurred_hits > 0.1
        update_grid[hit_mask] = blurred_hits[hit_mask] * self.l_occ
        
        return update_grid
