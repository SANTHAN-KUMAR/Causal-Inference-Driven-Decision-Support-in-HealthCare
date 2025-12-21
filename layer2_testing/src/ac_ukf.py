"""
AEGIS 3.0 Layer 2 - Adaptive Constrained Unscented Kalman Filter (AC-UKF)

Implements AC-UKF with:
1. Innovation-based covariance adaptation
2. Constraint projection for physiological bounds
"""

import numpy as np
from typing import Tuple, Optional, Callable
from dataclasses import dataclass

try:
    from .config import CONFIG
except ImportError:
    from config import CONFIG


class AdaptiveConstrainedUKF:
    """
    Adaptive Constrained Unscented Kalman Filter.
    
    Features:
    1. Innovation-based covariance adaptation (Q inflates when residuals large)
    2. Constraint projection (sigma points stay within bounds)
    """
    
    def __init__(self,
                 dim_x: int = 3,  # State dimension [G, X, I]
                 dim_z: int = 1,  # Measurement dimension (glucose only)
                 dt: float = 5.0,
                 fx: Callable = None,  # State transition function
                 hx: Callable = None,  # Measurement function
                 alpha: float = None,
                 beta: float = None,
                 kappa: float = None,
                 bounds: Tuple[np.ndarray, np.ndarray] = None):
        """
        Initialize AC-UKF.
        
        Args:
            dim_x: State dimension
            dim_z: Measurement dimension
            dt: Timestep
            fx: State transition f(x, dt) -> x_next
            hx: Measurement h(x) -> z
            alpha: Spread of sigma points
            beta: Prior knowledge (2 for Gaussian)
            kappa: Secondary scaling
            bounds: (lower_bounds, upper_bounds) for states
        """
        cfg = CONFIG.ac_ukf
        
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dt = dt
        self.fx = fx
        self.hx = hx or (lambda x: x[:1])  # Default: observe only glucose
        
        # UKF scaling parameters
        self.alpha = alpha or cfg.alpha
        self.beta = beta or cfg.beta
        self.kappa = kappa or cfg.kappa
        
        # Compute lambda
        self.lambda_ = self.alpha**2 * (dim_x + self.kappa) - dim_x
        
        # State estimate and covariance
        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x) * 10.0  # Initial uncertainty
        
        # Process and measurement noise
        self.Q = np.eye(dim_x) * 1.0  # Process noise
        self.R = np.eye(dim_z) * 5.0  # Measurement noise
        
        # Adaptation parameters
        self.adaptation_rate = cfg.adaptation_rate
        self.Q_baseline = self.Q.copy()
        
        # Bounds for constraint projection
        if bounds is None:
            cfg_b = CONFIG.bergman
            self.lower_bounds = np.array([cfg_b.glucose_min, 0, cfg_b.insulin_min])
            self.upper_bounds = np.array([cfg_b.glucose_max, 10, cfg_b.insulin_max])
        else:
            self.lower_bounds, self.upper_bounds = bounds
        
        # Residual history for adaptation
        self.residual_history = []
        self.residual_window = cfg.residual_window
        
        # Compute sigma point weights
        self._compute_weights()
    
    def _compute_weights(self):
        """Compute sigma point weights."""
        n = self.dim_x
        
        # Weights for mean
        self.Wm = np.zeros(2 * n + 1)
        self.Wm[0] = self.lambda_ / (n + self.lambda_)
        self.Wm[1:] = 1 / (2 * (n + self.lambda_))
        
        # Weights for covariance
        self.Wc = np.zeros(2 * n + 1)
        self.Wc[0] = self.Wm[0] + (1 - self.alpha**2 + self.beta)
        self.Wc[1:] = self.Wm[1:]
    
    def _generate_sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Generate sigma points around mean x with covariance P."""
        n = self.dim_x
        sigma_points = np.zeros((2 * n + 1, n))
        
        # Matrix square root
        try:
            sqrt_P = np.linalg.cholesky((n + self.lambda_) * P)
        except np.linalg.LinAlgError:
            # If not positive definite, use eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(P)
            eigvals = np.maximum(eigvals, 1e-8)
            sqrt_P = eigvecs @ np.diag(np.sqrt((n + self.lambda_) * eigvals))
        
        # Generate sigma points
        sigma_points[0] = x
        for i in range(n):
            sigma_points[i + 1] = x + sqrt_P[:, i]
            sigma_points[n + i + 1] = x - sqrt_P[:, i]
        
        return sigma_points
    
    def _project_to_constraints(self, sigma_points: np.ndarray) -> np.ndarray:
        """Project sigma points to satisfy physiological constraints."""
        projected = sigma_points.copy()
        
        for i in range(sigma_points.shape[0]):
            projected[i] = np.clip(projected[i], self.lower_bounds, self.upper_bounds)
        
        return projected
    
    def predict(self, u: np.ndarray = None):
        """
        Predict step: propagate sigma points through dynamics.
        
        Args:
            u: Control input (optional)
        """
        # Generate sigma points
        sigma_points = self._generate_sigma_points(self.x, self.P)
        
        # Project to constraints
        sigma_points = self._project_to_constraints(sigma_points)
        
        # Propagate through dynamics
        if self.fx is not None:
            sigma_points_pred = np.array([
                self.fx(sp, self.dt, u) for sp in sigma_points
            ])
        else:
            sigma_points_pred = sigma_points
        
        # Project predictions to constraints
        sigma_points_pred = self._project_to_constraints(sigma_points_pred)
        
        # Compute predicted mean
        self.x = np.sum(self.Wm[:, np.newaxis] * sigma_points_pred, axis=0)
        
        # Compute predicted covariance
        diff = sigma_points_pred - self.x
        self.P = np.zeros((self.dim_x, self.dim_x))
        for i in range(len(sigma_points_pred)):
            self.P += self.Wc[i] * np.outer(diff[i], diff[i])
        self.P += self.Q
        
        # Store sigma points for update step
        self._sigma_points_pred = sigma_points_pred
    
    def update(self, z: np.ndarray) -> np.ndarray:
        """
        Update step: incorporate measurement.
        
        Args:
            z: Measurement vector
            
        Returns:
            Innovation (residual)
        """
        sigma_points_pred = self._sigma_points_pred
        
        # Transform sigma points through measurement function
        z_sigma = np.array([self.hx(sp) for sp in sigma_points_pred])
        
        # Predicted measurement
        z_pred = np.sum(self.Wm[:, np.newaxis] * z_sigma, axis=0)
        
        # Innovation (residual)
        innovation = z - z_pred
        
        # Measurement covariance
        diff_z = z_sigma - z_pred
        S = np.zeros((self.dim_z, self.dim_z))
        for i in range(len(z_sigma)):
            S += self.Wc[i] * np.outer(diff_z[i], diff_z[i])
        S += self.R
        
        # Cross covariance
        diff_x = sigma_points_pred - self.x
        Pxz = np.zeros((self.dim_x, self.dim_z))
        for i in range(len(sigma_points_pred)):
            Pxz += self.Wc[i] * np.outer(diff_x[i], diff_z[i])
        
        # Kalman gain
        try:
            K = Pxz @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = Pxz @ np.linalg.pinv(S)
        
        # Update state and covariance
        self.x = self.x + K @ innovation
        self.P = self.P - K @ S @ K.T
        
        # Ensure positive definiteness
        self.P = 0.5 * (self.P + self.P.T)
        eigvals = np.linalg.eigvalsh(self.P)
        if np.min(eigvals) < 0:
            self.P += np.eye(self.dim_x) * (abs(np.min(eigvals)) + 1e-6)
        
        # Project state to constraints
        self.x = np.clip(self.x, self.lower_bounds, self.upper_bounds)
        
        # Adapt covariance based on innovation
        self._adapt_covariance(innovation, S, K)
        
        return innovation
    
    def _adapt_covariance(self, innovation: np.ndarray, S: np.ndarray, K: np.ndarray):
        """
        Innovation-based covariance adaptation.
        
        If empirical residual variance exceeds prediction, inflate Q.
        Q_{k+1} = Q_k + α * K * (ε*ε' - S) * K'
        """
        # Store innovation
        self.residual_history.append(innovation.copy())
        if len(self.residual_history) > self.residual_window:
            self.residual_history.pop(0)
        
        if len(self.residual_history) >= 3:
            # Compute empirical residual covariance
            residuals = np.array(self.residual_history)
            
            # Handle 1D case (single measurement dimension)
            if residuals.ndim == 1 or (residuals.ndim == 2 and residuals.shape[1] == 1):
                # Flatten and compute variance
                flat_residuals = residuals.flatten()
                empirical_S = np.array([[np.var(flat_residuals)]])
            else:
                empirical_S = np.cov(residuals.T)
            
            if np.isscalar(empirical_S):
                empirical_S = np.array([[empirical_S]])
            
            # Ensure S has proper shape for trace
            if empirical_S.ndim == 0:
                empirical_S = np.array([[float(empirical_S)]])
            
            # If empirical > predicted, inflate Q
            if np.trace(empirical_S) > np.trace(S):
                delta = K @ (empirical_S - S) @ K.T
                self.Q = self.Q + self.adaptation_rate * delta
                
                # Keep Q positive definite
                self.Q = 0.5 * (self.Q + self.Q.T)
                eigvals = np.linalg.eigvalsh(self.Q)
                if np.min(eigvals) < 0:
                    self.Q += np.eye(self.dim_x) * (abs(np.min(eigvals)) + 1e-6)
    
    def get_q_ratio(self) -> float:
        """Return ratio of current Q to baseline Q."""
        return np.trace(self.Q) / np.trace(self.Q_baseline)
    
    def reset_q(self):
        """Reset Q to baseline."""
        self.Q = self.Q_baseline.copy()
        self.residual_history = []
    
    def get_state_estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return current state estimate and covariance."""
        return self.x.copy(), self.P.copy()
