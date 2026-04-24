import cv2
import numpy as np
from scipy.interpolate import Rbf

class SpatialAligner:
    """Handles rigid and non-rigid registration of spatial biological data."""
    
    def compute_tps_warp(self, source_pts, target_pts, grid_size, smoothness=0.0, pin_corners=True):
        """
        Computes the inverse mapping (Target -> Source) for TPS warping.
        Exposes 'smoothness' for researcher-level control over tissue flexibility.
        """
        x_src, y_src = source_pts[:, 0], source_pts[:, 1]
        x_tar, y_tar = target_pts[:, 0], target_pts[:, 1]
        
        if pin_corners:
            h, w = grid_size
            corners = np.array([[0,0], [0,w-1], [h-1,0], [h-1,w-1]])
            x_src = np.concatenate([x_src, corners[:, 1]])
            y_src = np.concatenate([y_src, corners[:, 0]])
            x_tar = np.concatenate([x_tar, corners[:, 1]])
            y_tar = np.concatenate([y_tar, corners[:, 0]])

        # epsilon=smoothness in Rbf
        rbf_x = Rbf(x_tar, y_tar, x_src, function='thin_plate', smooth=smoothness)
        rbf_y = Rbf(x_tar, y_tar, y_src, function='thin_plate', smooth=smoothness)
        
        return rbf_x, rbf_y

    def warp_image(self, image, rbf_x, rbf_y):
        h, w = image.shape[:2]
        yi, xi = np.mgrid[0:h, 0:w]
        xi_warped = rbf_x(xi, yi)
        yi_warped = rbf_y(xi, yi)
        return cv2.remap(image, xi_warped.astype(np.float32), yi_warped.astype(np.float32), cv2.INTER_LINEAR)
