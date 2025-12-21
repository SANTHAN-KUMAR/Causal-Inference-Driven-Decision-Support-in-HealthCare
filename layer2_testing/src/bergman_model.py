"""
AEGIS 3.0 Layer 2 - Bergman Minimal Model

Implements the classic Bergman Minimal Model for glucose-insulin dynamics.
This serves as the mechanistic component (f_mech) of the Universal Differential Equation.

References:
- Bergman et al. (1979) "Quantitative Estimation of Insulin Sensitivity"
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

try:
    from .config import CONFIG
except ImportError:
    from config import CONFIG


@dataclass
class BergmanState:
    """State variables for Bergman Minimal Model."""
    G: float   # Glucose concentration (mg/dL)
    X: float   # Remote insulin (insulin action) (min^-1)
    I: float   # Plasma insulin (μU/mL)
    
    def to_array(self) -> np.ndarray:
        return np.array([self.G, self.X, self.I])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'BergmanState':
        return cls(G=arr[0], X=arr[1], I=arr[2])


class BergmanMinimalModel:
    """
    Bergman Minimal Model for glucose-insulin dynamics.
    
    Equations:
    dG/dt = -p1 * (G - Gb) - X * G + D(t)
    dX/dt = -p2 * X + p3 * (I - Ib)
    dI/dt = -n * (I - Ib) + γ * max(G - Gb, 0) * t + u(t)
    
    Where:
    - G: Glucose concentration (mg/dL)
    - X: Remote insulin action (min^-1)
    - I: Plasma insulin (μU/mL)
    - D(t): Glucose disturbance (meal intake)
    - u(t): Insulin injection
    """
    
    def __init__(self, 
                 p1: float = None,
                 p2: float = None,
                 p3: float = None,
                 n: float = None,
                 gamma: float = None,
                 Gb: float = None,
                 Ib: float = None):
        """Initialize with population or custom parameters."""
        cfg = CONFIG.bergman
        
        self.p1 = p1 if p1 is not None else cfg.p1
        self.p2 = p2 if p2 is not None else cfg.p2
        self.p3 = p3 if p3 is not None else cfg.p3
        self.n = n if n is not None else cfg.n
        self.gamma = gamma if gamma is not None else cfg.gamma
        self.Gb = Gb if Gb is not None else cfg.Gb
        self.Ib = Ib if Ib is not None else cfg.Ib
        
        # Bounds
        self.G_min = cfg.glucose_min
        self.G_max = cfg.glucose_max
        self.I_min = cfg.insulin_min
        self.I_max = cfg.insulin_max
    
    def dynamics(self, 
                 state: np.ndarray, 
                 t: float,
                 D: float = 0.0,
                 u: float = 0.0) -> np.ndarray:
        """
        Compute state derivatives.
        
        Args:
            state: [G, X, I] current state
            t: Current time (minutes)
            D: Glucose disturbance (mg/dL/min) - from meals
            u: Insulin input (μU/mL/min) - from injections
            
        Returns:
            [dG/dt, dX/dt, dI/dt] state derivatives
        """
        G, X, I = state
        
        # Glucose dynamics
        dG = -self.p1 * (G - self.Gb) - X * G + D
        
        # Remote insulin (insulin action) dynamics
        dX = -self.p2 * X + self.p3 * (I - self.Ib)
        
        # Plasma insulin dynamics
        # Pancreatic response only when G > Gb
        pancreatic = self.gamma * max(0, G - self.Gb)
        dI = -self.n * (I - self.Ib) + pancreatic + u
        
        return np.array([dG, dX, dI])
    
    def simulate(self,
                 initial_state: np.ndarray,
                 t_span: Tuple[float, float],
                 dt: float = 5.0,
                 D_func: callable = None,
                 u_func: callable = None,
                 noise_std: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate glucose-insulin trajectory using RK4.
        
        Args:
            initial_state: [G0, X0, I0]
            t_span: (t_start, t_end) in minutes
            dt: Timestep in minutes
            D_func: Glucose disturbance as function of time
            u_func: Insulin input as function of time
            noise_std: Process noise standard deviation
            
        Returns:
            times: Time points
            states: State trajectory [T x 3]
        """
        if D_func is None:
            D_func = lambda t: 0.0
        if u_func is None:
            u_func = lambda t: 0.0
        
        t_start, t_end = t_span
        times = np.arange(t_start, t_end + dt, dt)
        n_steps = len(times)
        
        states = np.zeros((n_steps, 3))
        states[0] = initial_state
        
        for i in range(1, n_steps):
            t = times[i-1]
            state = states[i-1]
            D = D_func(t)
            u = u_func(t)
            
            # RK4 integration
            k1 = self.dynamics(state, t, D, u)
            k2 = self.dynamics(state + 0.5*dt*k1, t + 0.5*dt, D, u)
            k3 = self.dynamics(state + 0.5*dt*k2, t + 0.5*dt, D, u)
            k4 = self.dynamics(state + dt*k3, t + dt, D, u)
            
            new_state = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            
            # Add process noise
            if noise_std > 0:
                new_state += np.random.randn(3) * noise_std
            
            # Apply physiological constraints
            new_state[0] = np.clip(new_state[0], self.G_min, self.G_max)
            new_state[1] = max(0, new_state[1])  # X >= 0
            new_state[2] = np.clip(new_state[2], self.I_min, self.I_max)
            
            states[i] = new_state
        
        return times, states
    
    def get_equilibrium(self) -> np.ndarray:
        """Return equilibrium state [Gb, 0, Ib]."""
        return np.array([self.Gb, 0.0, self.Ib])


def create_meal_disturbance(
    meal_times: list,
    meal_sizes: list,
    absorption_rate: float = 0.03
) -> callable:
    """
    Create glucose disturbance function from meals.
    
    Args:
        meal_times: List of meal times (minutes)
        meal_sizes: List of carb amounts (grams)
        absorption_rate: Gut absorption rate (min^-1)
        
    Returns:
        Function D(t) returning glucose input rate
    """
    def D_func(t: float) -> float:
        total = 0.0
        for meal_t, carbs in zip(meal_times, meal_sizes):
            if t >= meal_t:
                # Simplified glucose appearance from meal
                # Uses exponential decay model
                time_since = t - meal_t
                # ~3 mg/dL per gram of carbs, decaying over time
                glucose_rate = 3.0 * carbs * absorption_rate * np.exp(-absorption_rate * time_since)
                total += glucose_rate
        return total
    
    return D_func


def create_insulin_input(
    injection_times: list,
    doses: list,
    absorption_rate: float = 0.02
) -> callable:
    """
    Create insulin input function from injections.
    
    Args:
        injection_times: List of injection times (minutes)
        doses: List of insulin doses (units)
        absorption_rate: Subcutaneous absorption rate (min^-1)
        
    Returns:
        Function u(t) returning insulin appearance rate
    """
    def u_func(t: float) -> float:
        total = 0.0
        for inj_t, dose in zip(injection_times, doses):
            if t >= inj_t:
                time_since = t - inj_t
                # Insulin appearance rate (simplified)
                insulin_rate = dose * absorption_rate * np.exp(-absorption_rate * time_since)
                total += insulin_rate
        return total
    
    return u_func
