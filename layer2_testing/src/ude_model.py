"""
AEGIS 3.0 Layer 2 - Universal Differential Equation (UDE)

Implements the UDE model: dx/dt = f_mech(x,u;θ_fixed) + f_NN(x,u;θ_learned)

The mechanistic component is the Bergman Minimal Model.
The neural component learns patient-specific residuals.
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass

try:
    from .config import CONFIG
    from .bergman_model import BergmanMinimalModel
except ImportError:
    from config import CONFIG
    from bergman_model import BergmanMinimalModel


class NeuralResidual:
    """
    Simple neural network for learning residuals.
    
    Implements a multi-layer perceptron with tanh activation.
    Uses basic gradient descent with stability enhancements.
    """
    
    def __init__(self, 
                 input_dim: int = 4,  # [G, X, I, u]
                 output_dim: int = 3,  # [dG, dX, dI] residuals
                 hidden_dim: int = None,
                 num_layers: int = None):
        """Initialize neural network with random weights."""
        cfg = CONFIG.ude
        hidden_dim = hidden_dim or cfg.hidden_dim
        num_layers = num_layers or cfg.num_layers
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input normalization parameters (will be set during training)
        self.input_mean = np.zeros(input_dim)
        self.input_std = np.ones(input_dim)
        self.output_scale = 0.1  # Small output scale for residuals
        
        # Initialize weights with small values for stability
        np.random.seed(CONFIG.random_seed)
        self.weights = []
        self.biases = []
        
        # Use smaller initialization
        scale = 0.1
        
        # Input layer
        self.weights.append(np.random.randn(input_dim, hidden_dim) * scale)
        self.biases.append(np.zeros(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.weights.append(np.random.randn(hidden_dim, hidden_dim) * scale)
            self.biases.append(np.zeros(hidden_dim))
        
        # Output layer (very small to start)
        self.weights.append(np.random.randn(hidden_dim, output_dim) * 0.01)
        self.biases.append(np.zeros(output_dim))
    
    def _normalize_input(self, x: np.ndarray) -> np.ndarray:
        """Normalize inputs to zero mean, unit variance."""
        return (x - self.input_mean) / (self.input_std + 1e-8)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through network."""
        # Normalize input
        h = self._normalize_input(x)
        
        # Hidden layers with tanh activation
        for i in range(len(self.weights) - 1):
            h = np.tanh(h @ self.weights[i] + self.biases[i])
        
        # Output layer (linear) scaled down for small residuals
        out = h @ self.weights[-1] + self.biases[-1]
        
        return out * self.output_scale
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
    
    def set_normalization(self, inputs: np.ndarray):
        """Set normalization parameters from training data."""
        self.input_mean = np.mean(inputs, axis=0)
        self.input_std = np.std(inputs, axis=0) + 1e-8
    
    def train_step(self, 
                   inputs: np.ndarray, 
                   targets: np.ndarray,
                   lr: float = None) -> float:
        """
        Single training step using simple gradient descent.
        
        Uses finite differences for gradient approximation with stability.
        """
        lr = lr or CONFIG.ude.learning_rate
        epsilon = 1e-4
        
        # Set normalization from inputs
        self.set_normalization(inputs)
        
        # Scale targets to match output scale
        scaled_targets = targets / self.output_scale
        
        # Current loss
        predictions = np.array([self.forward(x) / self.output_scale for x in inputs])
        loss = np.mean((predictions - scaled_targets) ** 2)
        
        if np.isnan(loss) or np.isinf(loss):
            return float('nan')
        
        # Update weights with gradient clipping
        max_grad = 1.0
        
        for layer_idx in range(len(self.weights)):
            # Simplified update: random direction with finite difference
            # (Much faster than full gradient computation)
            for _ in range(min(3, self.weights[layer_idx].size)):
                i = np.random.randint(0, self.weights[layer_idx].shape[0])
                j = np.random.randint(0, self.weights[layer_idx].shape[1])
                
                orig_val = self.weights[layer_idx][i, j]
                self.weights[layer_idx][i, j] = orig_val + epsilon
                
                predictions_plus = np.array([self.forward(x) / self.output_scale for x in inputs])
                loss_plus = np.mean((predictions_plus - scaled_targets) ** 2)
                
                if np.isnan(loss_plus):
                    self.weights[layer_idx][i, j] = orig_val
                    continue
                
                grad = (loss_plus - loss) / epsilon
                grad = np.clip(grad, -max_grad, max_grad)
                
                self.weights[layer_idx][i, j] = orig_val - lr * grad
        
        return loss


class UniversalDifferentialEquation:
    """
    Universal Differential Equation combining mechanistic and neural components.
    
    dx/dt = f_mech(x, u; θ_fixed) + f_NN(x, u; θ_learned)
    """
    
    def __init__(self,
                 mechanistic_model: BergmanMinimalModel = None,
                 neural_residual: NeuralResidual = None):
        """Initialize UDE with mechanistic model and neural residual."""
        self.mechanistic = mechanistic_model or BergmanMinimalModel()
        self.neural = neural_residual or NeuralResidual()
        
        # Track training history
        self.training_losses = []
    
    def dynamics(self,
                 state: np.ndarray,
                 t: float,
                 D: float = 0.0,
                 u: float = 0.0) -> np.ndarray:
        """
        Compute combined dynamics.
        
        Args:
            state: [G, X, I] current state
            t: Current time
            D: Glucose disturbance (meal)
            u: Insulin input
            
        Returns:
            dx/dt = f_mech + f_NN
        """
        # Mechanistic component
        mech_deriv = self.mechanistic.dynamics(state, t, D, u)
        
        # Neural component input: [G, X, I, u]
        nn_input = np.concatenate([state, [u]])
        neural_deriv = self.neural(nn_input)
        
        return mech_deriv + neural_deriv
    
    def simulate(self,
                 initial_state: np.ndarray,
                 t_span: Tuple[float, float],
                 dt: float = 5.0,
                 D_func: callable = None,
                 u_func: callable = None,
                 noise_std: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate trajectory using RK4 with combined dynamics.
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
            
            # RK4 with combined dynamics
            k1 = self.dynamics(state, t, D, u)
            k2 = self.dynamics(state + 0.5*dt*k1, t + 0.5*dt, D, u)
            k3 = self.dynamics(state + 0.5*dt*k2, t + 0.5*dt, D, u)
            k4 = self.dynamics(state + dt*k3, t + dt, D, u)
            
            new_state = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            
            if noise_std > 0:
                new_state += np.random.randn(3) * noise_std
            
            # Constraints
            new_state[0] = np.clip(new_state[0], 
                                   self.mechanistic.G_min, 
                                   self.mechanistic.G_max)
            new_state[1] = max(0, new_state[1])
            new_state[2] = np.clip(new_state[2],
                                   self.mechanistic.I_min,
                                   self.mechanistic.I_max)
            
            states[i] = new_state
        
        return times, states
    
    def train_neural_component(self,
                               observations: np.ndarray,
                               times: np.ndarray,
                               inputs: np.ndarray,
                               max_epochs: int = None,
                               lr: float = None) -> List[float]:
        """
        Train neural residual to capture patient-specific deviations.
        
        Args:
            observations: Observed states [T x 3]
            times: Time points
            inputs: Insulin inputs at each time
            max_epochs: Maximum training epochs
            lr: Learning rate
            
        Returns:
            Training loss history
        """
        max_epochs = max_epochs or CONFIG.ude.max_epochs
        lr = lr or CONFIG.ude.learning_rate
        
        # Compute target residuals
        # Target = observed derivative - mechanistic prediction
        dt = times[1] - times[0] if len(times) > 1 else 5.0
        
        # Estimate observed derivatives using finite differences
        obs_derivs = np.diff(observations, axis=0) / dt
        
        # Compute mechanistic predictions
        mech_derivs = np.array([
            self.mechanistic.dynamics(observations[i], times[i], 0, inputs[i])
            for i in range(len(observations) - 1)
        ])
        
        # Target residuals = observed - mechanistic
        target_residuals = obs_derivs - mech_derivs
        
        # Prepare training data
        nn_inputs = np.array([
            np.concatenate([observations[i], [inputs[i]]])
            for i in range(len(observations) - 1)
        ])
        
        # Training loop
        losses = []
        for epoch in range(max_epochs):
            loss = self.neural.train_step(nn_inputs, target_residuals, lr)
            losses.append(loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.6f}")
        
        self.training_losses = losses
        return losses
    
    def get_rmse(self,
                 true_states: np.ndarray,
                 times: np.ndarray,
                 initial_state: np.ndarray,
                 D_func: callable = None,
                 u_func: callable = None) -> float:
        """Compute RMSE between predicted and true trajectory."""
        _, predicted = self.simulate(
            initial_state, 
            (times[0], times[-1]),
            dt=times[1]-times[0] if len(times) > 1 else 5.0,
            D_func=D_func,
            u_func=u_func
        )
        
        # Only compare glucose (most important)
        rmse = np.sqrt(np.mean((predicted[:len(true_states), 0] - true_states[:, 0]) ** 2))
        return rmse
