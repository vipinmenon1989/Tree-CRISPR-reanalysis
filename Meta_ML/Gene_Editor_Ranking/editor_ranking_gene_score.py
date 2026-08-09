import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, root_scalar
import logging
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

@dataclass
class EditorResult:
    editor_name: str
    ess: float
    rank: int
    weights: List[float]
    n_eff: float

@dataclass
class RecommendationOutput:
    recommended_editor: str
    optimal_beta: float
    target_n_eff: float
    editor_rankings: List[EditorResult]

class EditorRecommendationEngine:
    """
    Framework for recommending the optimal CRISPR editor for a target gene
    by calculating the Editor Suitability Score (ESS).
    """

    def __init__(self, target_effective_guides: float = 6.0, optimizer: str = "minimize_scalar"):
        """
        Initialize the recommendation engine.

        Parameters
        ----------
        target_effective_guides : float
            The desired effective guide diversity (N_eff_target).
        optimizer : str
            The optimization method to use: 'minimize_scalar' or 'root_scalar'.
        """
        if target_effective_guides < 1.0:
            raise ValueError("target_effective_guides must be >= 1.0")
            
        self.target_n_eff = target_effective_guides
        self.optimizer = optimizer
        self.beta_star: Optional[float] = None

    def _clean_scores(self, scores: np.ndarray) -> np.ndarray:
        """Removes NaN and Inf values from the score array."""
        clean = scores[np.isfinite(scores)]
        if len(clean) == 0:
            raise ValueError("Score array contains no valid numbers after dropping NaN/Inf.")
        return clean

    def softmax_weights(self, scores: np.ndarray, beta: float) -> np.ndarray:
        """
        Computes numerically stable softmax weights for candidate guides.
        
        Math: w_i = exp(beta * s_i) / sum_j exp(beta * s_j)
        Stability is achieved by subtracting the max value before exponentiation.
        """
        if len(scores) == 0:
            return np.array([])
        
        # Numerical stability: shift by max
        scaled_scores = beta * scores
        shifted_scores = scaled_scores - np.max(scaled_scores)
        exp_scores = np.exp(shifted_scores)
        
        weights = exp_scores / np.sum(exp_scores)
        return weights

    def effective_guides(self, weights: np.ndarray) -> float:
        """
        Computes the effective number of contributing guides (N_eff).
        
        Math: N_eff = 1 / sum(w_i^2)
        """
        if len(weights) == 0:
            return 0.0
        sum_sq_weights = np.sum(weights ** 2)
        if sum_sq_weights == 0:
            return 0.0
        return 1.0 / sum_sq_weights

    def _objective_function(self, beta: float, scores: np.ndarray) -> float:
        """
        Objective function for calibration.
        Returns the absolute difference between current N_eff and target N_eff.
        """
        weights = self.softmax_weights(scores, beta)
        current_n_eff = self.effective_guides(weights)
        return abs(current_n_eff - self.target_n_eff)

    def fit_beta(self, calibration_scores: Union[List[float], np.ndarray]) -> float:
        """
        Determines the optimal global guide emphasis parameter (beta*).
        
        This calibration is performed once using a representative distribution 
        of candidate guides to approach N_eff_target.
        """
        scores = self._clean_scores(np.array(calibration_scores, dtype=float))
        n_guides = len(scores)

        if n_guides < self.target_n_eff:
            logging.warning(
                f"Target N_eff ({self.target_n_eff}) exceeds total guides ({n_guides}). "
                "Optimization will default beta to approach 0 (equal weighting)."
            )

        if self.optimizer == "minimize_scalar":
            result = minimize_scalar(
                self._objective_function, 
                args=(scores,), 
                bounds=(-10.0, 50.0), 
                method='bounded'
            )
            self.beta_star = result.x
            
        elif self.optimizer == "root_scalar":
            # root_scalar requires a bracket where the function changes sign.
            # Using minimize_scalar is mathematically more robust for absolute differences.
            def root_func(b):
                w = self.softmax_weights(scores, b)
                return self.effective_guides(w) - self.target_n_eff
                
            try:
                result = root_scalar(root_func, bracket=[0, 50], method='brentq')
                self.beta_star = result.root
            except ValueError:
                logging.warning("root_scalar failed to find a bracket. Falling back to minimize_scalar.")
                self.optimizer = "minimize_scalar"
                return self.fit_beta(calibration_scores)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

        logging.info(f"Calibration complete. beta* = {self.beta_star:.4f}")
        return self.beta_star

    def compute_ess(self, scores: np.ndarray, beta: float) -> Tuple[float, np.ndarray, float]:
        """
        Computes the Editor Suitability Score (ESS).
        
        Math: ESS = sum_i (w_i * s_i)
        
        Returns:
            ESS value, calculated weights, and N_eff for this specific gene.
        """
        weights = self.softmax_weights(scores, beta)
        ess = np.sum(weights * scores)
        n_eff = self.effective_guides(weights)
        return float(ess), weights, float(n_eff)

    def rank_editors(self, editor_scores: Dict[str, List[float]]) -> List[EditorResult]:
        """
        Processes a dictionary of editors and their candidate guide scores.
        Sorts the editors by ESS in descending order.
        """
        if self.beta_star is None:
            raise RuntimeError("Engine not calibrated. Call fit_beta() first.")

        results = []
        for editor, scores_list in editor_scores.items():
            scores = self._clean_scores(np.array(scores_list, dtype=float))
            
            if len(scores) == 0:
                logging.warning(f"Editor {editor} has no valid scores. Skipping.")
                continue
                
            ess, weights, n_eff = self.compute_ess(scores, self.beta_star)
            
            results.append({
                "editor_name": editor,
                "ess": ess,
                "weights": weights.tolist(),
                "n_eff": n_eff
            })

        # Sort descending by ESS
        results.sort(key=lambda x: x["ess"], reverse=True)
        
        # Package into Dataclasses with assigned ranks
        ranked_results = []
        for rank, res in enumerate(results, start=1):
            ranked_results.append(
                EditorResult(
                    editor_name=res["editor_name"],
                    ess=res["ess"],
                    rank=rank,
                    weights=res["weights"],
                    n_eff=res["n_eff"]
                )
            )
            
        return ranked_results

    def recommend(self, editor_scores: Dict[str, List[float]]) -> RecommendationOutput:
        """
        Generates the final recommendation output.
        """
        ranked_editors = self.rank_editors(editor_scores)
        
        if not ranked_editors:
            raise ValueError("No valid editors could be evaluated.")

        best_editor = ranked_editors[0].editor_name

        return RecommendationOutput(
            recommended_editor=best_editor,
            optimal_beta=self.beta_star,
            target_n_eff=self.target_n_eff,
            editor_rankings=ranked_editors
        )

    def save(self, filepath: str):
        """Serializes the engine state to JSON."""
        state = {
            "target_n_eff": self.target_n_eff,
            "optimizer": self.optimizer,
            "beta_star": self.beta_star
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=4)
        logging.info(f"Engine state saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "EditorRecommendationEngine":
        """Deserializes the engine state from JSON."""
        with open(filepath, 'r') as f:
            state = json.load(f)
            
        engine = cls(
            target_effective_guides=state.get("target_n_eff", 6.0),
            optimizer=state.get("optimizer", "minimize_scalar")
        )
        engine.beta_star = state.get("beta_star")
        logging.info(f"Engine state loaded from {filepath}")
        return engine
