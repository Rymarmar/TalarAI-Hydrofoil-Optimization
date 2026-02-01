import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate, add
from tensorflow.keras import optimizers, initializers
from keras import backend as K
import joblib
from typing import List, Tuple, Callable, Dict, Optional

class ConstrainedNNOptimizer:
    """
    A generalized framework for optimizing design parameters using a pre-trained neural network
    with custom constraints and activation functions.
    """
    
    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        param_bounds: List[List[float]],
        n_samples: int = 20
    ):
        """
        Initialize the optimizer.
        
        Args:
            model_path: Path to the pre-trained model (.h5 file)
            scaler_path: Path to the scaler object (.out file)
            param_bounds: List of [min_values, max_values] where each is a list of bounds
                         e.g., [[0.02, 2, 0.0075], [0.08, 14, 0.014]]
            n_samples: Number of samples for initialization
        """
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.scaler = joblib.load(scaler_path)
        self.param_bounds = param_bounds
        self.n_params = len(param_bounds[0])
        self.n_samples = n_samples
        
        # Make model non-trainable
        self.model.trainable = False
        
        # Generate scaled parameter bounds
        self._compute_scaled_bounds()
        
    def _compute_scaled_bounds(self):
        """Compute scaled parameter bounds."""
        # param_bounds is [[min_vals], [max_vals]]
        bounds_array = np.array(self.param_bounds)
        self.scaled_bounds = self.scaler.transform(bounds_array)
        
    def generate_test_samples(self) -> np.ndarray:
        """Generate uniformly distributed test samples for each parameter."""
        test_samples = []
        min_vals = self.param_bounds[0]
        max_vals = self.param_bounds[1]
        
        for i in range(self.n_params):
            samples = np.linspace(min_vals[i], max_vals[i], self.n_samples)
            test_samples.append(samples)
        
        # Stack and scale
        test_array = np.column_stack(test_samples)
        scaled_samples = self.scaler.transform(test_array)
        
        return scaled_samples
    
    def create_constraint_activation(
        self,
        param_idx: int,
        penalty_weight: float = 100.0
    ) -> Callable:
        """
        Create a constraint activation function for a parameter.
        
        Args:
            param_idx: Index of the parameter
            penalty_weight: Weight for constraint violation penalty
            
        Returns:
            Constraint activation function
        """
        min_bound = self.scaled_bounds[0, param_idx]
        max_bound = self.scaled_bounds[1, param_idx]
        
        def constraint_activation(x):
            return penalty_weight * (
                K.relu(-x + min_bound) + K.relu(x - max_bound)
            )
        
        return constraint_activation
    
    def create_output_constraint(
        self,
        output_idx: int,
        threshold: float,
        constraint_type: str = 'max',
        penalty_weight: float = 50.0
    ) -> Callable:
        """
        Create a constraint on model outputs.
        
        Args:
            output_idx: Index of the output to constrain
            threshold: Threshold value for the constraint
            constraint_type: 'max' for upper bound, 'min' for lower bound
            penalty_weight: Weight for constraint violation penalty
            
        Returns:
            Output constraint activation function
        """
        def output_constraint(x):
            if constraint_type == 'max':
                return penalty_weight * K.relu(x - threshold)
            elif constraint_type == 'min':
                return penalty_weight * K.relu(threshold - x)
            else:
                raise ValueError(f"Unknown constraint type: {constraint_type}")
        
        return output_constraint
    
    def build_optimization_model(
        self,
        objective_fn: Callable,
        output_constraints: Optional[List[Dict]] = None,
        penalty_weights: Optional[List[float]] = None,
        debug: bool = False
    ) -> Model:
        """
        Build the optimization model with constraints.
        
        Args:
            objective_fn: Function that takes model output and returns cost
            output_constraints: List of dicts with 'idx', 'threshold', 'type', 'weight'
            penalty_weights: Custom penalty weights for each parameter constraint
            debug: If True, print detailed model building information
            
        Returns:
            Compiled optimization model
        """
        if debug:
            print("\n" + "="*50)
            print("Building Optimization Model")
            print("="*50)
            print(f"Number of parameters: {self.n_params}")
            print(f"Scaled bounds: {self.scaled_bounds}")
            print(f"\nPre-trained model summary:")
            self.model.summary()
        
        # Default penalty weights if not provided
        if penalty_weights is None:
            penalty_weights = [100.0, 100.0, 500.0]
        
        # Create input layers for each parameter
        input_layers = [Input(shape=(1,)) for _ in range(self.n_params)]
        if debug:
            print(f"\nCreated {len(input_layers)} input layers")
        
        # Create dense layers for each input
        dense_inputs = [Dense(1)(inp) for inp in input_layers]
        if debug:
            print(f"Created {len(dense_inputs)} trainable Dense(1) layers (6 params total)")
        
        # Create constraint layers for input parameters
        constraint_layers = []
        for i in range(self.n_params):
            activation = self.create_constraint_activation(i, penalty_weights[i])
            constraint = Dense(
                1,
                kernel_initializer=initializers.ones(),
                activation=activation,
                trainable=False
            )(dense_inputs[i])
            constraint_layers.append(constraint)
        
        if debug:
            print(f"Created {len(constraint_layers)} constraint layers for inputs (6 params)")
        
        # Concatenate inputs and pass through pre-trained model
        concatenated = Concatenate(axis=1)(dense_inputs)
        
        # Replicate model architecture
        hidden_layer = Dense(
            self.model.layers[1].get_config()['units'],
            activation=self.model.layers[1].get_config()['activation'],
            trainable=False
        )(concatenated)
        
        output_layer = Dense(
            self.model.layers[2].get_config()['units'],
            activation=self.model.layers[2].get_config()['activation'],
            trainable=False
        )(hidden_layer)
        
        if debug:
            print(f"\nReplicated pre-trained model architecture (703 params):")
            print(f"  Hidden layer: {self.model.layers[1].get_config()['units']} units")
            print(f"  Output layer: {self.model.layers[2].get_config()['units']} units")
        
        # Add output constraints if specified
        output_constraint_layers = []
        if output_constraints:
            for constraint_spec in output_constraints:
                idx = constraint_spec['idx']
                threshold = constraint_spec['threshold']
                c_type = constraint_spec.get('type', 'max')
                weight = constraint_spec.get('weight', 50.0)
                
                activation = self.create_output_constraint(idx, threshold, c_type, weight)
                neuron_output = tf.expand_dims(output_layer[:, idx], axis=-1)
                constraint = Dense(
                    1,
                    kernel_initializer=initializers.ones(),
                    activation=activation,
                    trainable=False
                )(neuron_output)
                output_constraint_layers.append(constraint)
            
            if debug:
                print(f"Created {len(output_constraint_layers)} output constraint layers (4 params)")
        
        # Combine all constraints
        all_constraints = constraint_layers + output_constraint_layers
        final_output = add([output_layer] + all_constraints)
        
        # Build model
        opt_model = Model(inputs=input_layers, outputs=final_output)
        
        # Set pre-trained weights
        opt_model.layers[self.n_params * 2 + 1].set_weights(
            self.model.layers[1].get_weights()
        )
        opt_model.layers[self.n_params * 2 + 2].set_weights(
            self.model.layers[2].get_weights()
        )
        
        if debug:
            print(f"\nTransferred weights from pre-trained model")
            print(f"\nFinal optimization model summary:")
            opt_model.summary()
            print(f"\nExpected: 719 total params (6 trainable, 713 non-trainable)")
        
        # Add loss function
        loss = objective_fn(final_output)
        opt_model.add_loss(loss)
        
        return opt_model
    
    def optimize(
        self,
        objective_fn: Callable,
        output_constraints: Optional[List[Dict]] = None,
        penalty_weights: Optional[List[float]] = None,
        n_iterations: int = 100,
        epochs: int = 1000,
        learning_rate: float = 0.0001,
        valid_threshold: Optional[Dict[int, float]] = None,
        verbose: int = 0
    ) -> Dict:
        """
        Run optimization iterations.
        
        Args:
            objective_fn: Objective function for optimization
            output_constraints: List of output constraints
            penalty_weights: Custom penalty weights for parameter constraints
            n_iterations: Number of random initializations
            epochs: Training epochs per iteration
            learning_rate: Learning rate for optimizer
            valid_threshold: Dict mapping output index to max valid value
            verbose: Verbosity level (0=quiet, 1=progress, 2=detailed)
            
        Returns:
            Dictionary with best results
        """
        test_samples = self.generate_test_samples()
        
        costs = []
        optimal_params = []
        predictions = []
        
        # Create a single model for extracting parameters
        param_input = Input(shape=(1,))
        param_output = Dense(1)(param_input)
        param_extractor = Model(inputs=param_input, outputs=param_output)
        param_extractor.compile(optimizer='adam', loss='mse')
        
        for iteration in range(n_iterations):
            # Build optimization model
            opt_model = self.build_optimization_model(
                objective_fn,
                output_constraints,
                penalty_weights
            )
            
            # Compile model
            optimizer_adam = optimizers.Adam(learning_rate=learning_rate)
            opt_model.compile(optimizer=optimizer_adam, metrics=['mse'])
            
            # Random initialization
            init_params = [
                np.array([test_samples[np.random.randint(0, self.n_samples), i]])
                for i in range(self.n_params)
            ]
            
            # Train
            history = opt_model.fit(
                init_params,
                verbose=0,
                epochs=epochs,
                batch_size=1
            )
            
            final_cost = history.history['loss'][-1]
            
            if verbose >= 1:
                print(f"Iteration {iteration+1}/{n_iterations} - Final Loss: {final_cost:.4f}, Cost: {-final_cost:.4f}")
            
            # Extract optimized parameters by getting weights from dense layers
            optimized_params = []
            for i in range(self.n_params):
                # The dense layer for parameter i is at index self.n_params + i
                layer_idx = self.n_params + i
                weights = opt_model.layers[layer_idx].get_weights()
                
                # Set weights in extractor and predict
                param_extractor.layers[1].set_weights(weights)
                param_value = param_extractor.predict(init_params[i], verbose=0)
                
                # Clip to scaled bounds to handle any numerical issues
                param_value_clipped = np.clip(
                    param_value[0, 0],
                    self.scaled_bounds[0, i],
                    self.scaled_bounds[1, i]
                )
                optimized_params.append(param_value_clipped)
            
            # Inverse transform to original scale
            original_params = self.scaler.inverse_transform([optimized_params])[0]
            
            # Double-check bounds after inverse transform (shouldn't be needed with clipping, but just in case)
            params_in_bounds = True
            for i in range(self.n_params):
                if (original_params[i] < self.param_bounds[0][i] - 1e-6 or 
                    original_params[i] > self.param_bounds[1][i] + 1e-6):
                    params_in_bounds = False
                    if verbose >= 2:
                        print(f"  Parameter {i} out of bounds: {original_params[i]:.6f} not in [{self.param_bounds[0][i]}, {self.param_bounds[1][i]}]")
                    break
            
            if not params_in_bounds:
                if verbose >= 1:
                    print("  Parameters out of bounds - skipping")
                    print("-" * 50)
                continue
            
            # Predict outputs
            prediction = self.model.predict(
                np.array([optimized_params]),
                verbose=0
            )[0]
            
            # Check validity using output constraints
            is_valid = True
            if valid_threshold:
                for idx, max_val in valid_threshold.items():
                    if prediction[idx] >= max_val:
                        is_valid = False
                        if verbose >= 2:
                            print(f"  Invalid: output[{idx}] = {prediction[idx]:.4f} >= {max_val}")
                        break
            
            if is_valid:
                costs.append(-final_cost)  # Negative because we minimize in training
                optimal_params.append(original_params)
                predictions.append(prediction)
                if verbose >= 2:
                    print(f"  Valid solution found!")
                    print(f"  Parameters: {original_params}")
                    print(f"  Outputs: {prediction}")
                    print(f"  Cost (negative of loss): {-final_cost:.4f}")
            
            if verbose >= 1:
                print("-" * 50)
        
        # Find best result
        if len(costs) > 0:
            best_idx = np.argmax(costs)
            return {
                'best_cost': costs[best_idx],
                'best_params': optimal_params[best_idx],
                'best_prediction': predictions[best_idx],
                'all_costs': costs,
                'all_params': optimal_params,
                'all_predictions': predictions,
                'n_valid': len(costs),
                'n_total': n_iterations
            }
        else:
            return {
                'best_cost': None,
                'best_params': None,
                'best_prediction': None,
                'all_costs': [],
                'all_params': [],
                'all_predictions': [],
                'n_valid': 0,
                'n_total': n_iterations
            }
    
    def post_process_results(self, results: Dict, scale_outputs: bool = True) -> Dict:
        """
        Post-process results to convert outputs to physical units.
        
        Args:
            results: Results dictionary from optimize()
            scale_outputs: If True, apply inverse transformations to outputs
            
        Returns:
            Dictionary with processed results
        """
        if results['best_prediction'] is None:
            return results
        
        processed = results.copy()
        
        if scale_outputs:
            # Based on original code transformations:
            # power *= -1e4 -> power = output[0] / -1e4 (but in microwatts: * -1e2)
            # PD *= -1 -> PD = output[1] / -1
            # dwdx_max *= 1e2 -> dwdx = output[2] / 1e2
            
            processed['best_prediction_physical'] = np.array([
                results['best_prediction'][0] * -1e2,  # Power in microwatts
                results['best_prediction'][1] / -1,     # PD (positive)
                results['best_prediction'][2] / 1e2     # dwdx_max
            ])
            
            processed['all_predictions_physical'] = []
            for pred in results['all_predictions']:
                processed['all_predictions_physical'].append(np.array([
                    pred[0] * -1e2,
                    pred[1] / -1,
                    pred[2] / 1e2
                ]))
        
        return processed
    
    # Example usage
if __name__ == "__main__":
    # NOTE: This example assumes you have the model and scaler files
    # Adjust paths and parameters according to your actual setup
    
    # Define parameter bounds as [[min_values], [max_values]]
    param_bounds = [
        [0.02, 2.0, 0.0075],      # Min values for L, MR, Do
        [0.08, 14.0, 0.014]       # Max values for L, MR, Do
    ]
    
    # Initialize optimizer
    optimizer = ConstrainedNNOptimizer(
        model_path='model_acc_1_final.h5',
        scaler_path='minmax_scaler_acc_1_final.out',
        param_bounds=param_bounds,
        n_samples=20
    )
    
    # Define objective function (maximize PD, which is output[1])
    # The original code minimizes output[:, 1] which is already negative PD
    # So this correctly maximizes PD
    def objective(output):
        return output[:, 1]
    
    # Define output constraints
    # Based on your original code:
    # - dwdx_max (output[2]) should be less than 0.17 * 100
    # - power (output[0]) should be greater than -200 (after multiplication by -1e4)
    output_constraints = [
        {'idx': 2, 'threshold': 0.17 * 100, 'type': 'max', 'weight': 50.0},
        {'idx': 0, 'threshold': -200, 'type': 'min', 'weight': 0.1}
    ]
    
    # Test with debug mode first
    print("Testing model building with debug mode...")
    opt_model = optimizer.build_optimization_model(
        objective_fn=objective,
        output_constraints=output_constraints,
        penalty_weights=[100.0, 100.0, 500.0],  # Match original
        debug=True
    )
    
    print("\n\nNow running optimization...")
    print("="*50)
    
    # Run optimization
    results = optimizer.optimize(
        objective_fn=objective,
        output_constraints=output_constraints,
        penalty_weights=[100.0, 100.0, 500.0],
        n_iterations=100,
        epochs=1000,
        learning_rate=0.001,  # Match original code
        valid_threshold={2: 0.17 * 100},  # dwdx constraint
        verbose=1
    )
    
    # Print results
    if results['best_params'] is not None:
        # Post-process to get physical units
        processed = optimizer.post_process_results(results, scale_outputs=True)
        
        print("\n" + "="*50)
        print("OPTIMIZATION RESULTS")
        print("="*50)
        print(f"Best Cost: {results['best_cost']:.4f}")
        print(f"\nBest Parameters:")
        print(f"  L:  {results['best_params'][0]:.6f}")
        print(f"  MR: {results['best_params'][1]:.6f}")
        print(f"  Do: {results['best_params'][2]:.6f}")
        print(f"\nBest Prediction (raw model outputs):")
        print(f"  Output[0]: {results['best_prediction'][0]:.4f}")
        print(f"  Output[1]: {results['best_prediction'][1]:.4f}")
        print(f"  Output[2]: {results['best_prediction'][2]:.4f}")
        print(f"\nBest Prediction (physical units):")
        print(f"  Power:     {processed['best_prediction_physical'][0]:.4f} µW")
        print(f"  PD:        {processed['best_prediction_physical'][1]:.4f}")
        print(f"  dwdx_max:  {processed['best_prediction_physical'][2]:.4f}")
        print(f"\nValid Solutions: {results['n_valid']}/{results['n_total']}")
    else:
        print("\nNo valid solutions found!")