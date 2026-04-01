from typing import Dict, List, Optional
import argparse
import json
from pathlib import Path

class ParameterSystem:
    """
    Centralized parameter management system for the election model.
    
    Handles command-line argument parsing, parameter validation, and provides
    organized access to model parameters through property groups.
    """
    
    def __init__(self):
        """Initialize the parameter system with default values."""
        self.args = None
        self._parser = self._build_parser()
    
    def _build_parser(self) -> argparse.ArgumentParser:
        """Build and configure the argument parser."""
        p = argparse.ArgumentParser(description="Agent-based model for Pelotas council elections")
        
        # Basic run control
        self._add_basic_args(p)
        
        # Sizes and scales
        self._add_size_args(p)
        
        # Social network parameters
        self._add_network_args(p)
        
        # Input data arguments
        self._add_data_args(p)
        
        # Party generation parameters
        self._add_party_args(p)
        
        # Candidate generation parameters
        self._add_candidate_args(p)
        
        # Voter generation parameters
        self._add_voter_args(p)
        
        # Resource weights
        self._add_resource_args(p)
        
        # Visibility dynamics parameters
        self._add_visibility_args(p)
        
        # Territorial field parameters
        self._add_territorial_args(p)
        
        # Utility parameters
        self._add_utility_args(p)
        
        # Strategic voting parameters
        self._add_strategic_args(p)
        
        # Choice noise and attention parameters
        self._add_choice_args(p)
        
        # Output options
        self._add_output_args(p)
        
        return p
    
    def _add_basic_args(self, p: argparse.ArgumentParser) -> None:
        """Add basic run control arguments."""
        basic = p.add_argument_group("Basic Run Control")
        basic.add_argument("--output-dir", type=str, required=True, 
                          help="directory for outputs")
        basic.add_argument("--seed", type=int, default=12345, 
                          help="random seed for reproducibility")
        basic.add_argument("--cycle", type=int, default=2024, 
                          choices=[2016, 2020, 2024], 
                          help="election cycle (affects corporate donation rules)")
    
    def _add_size_args(self, p: argparse.ArgumentParser) -> None:
        """Add size and scale arguments."""
        size = p.add_argument_group("Sizes and Scales")
        size.add_argument("--n-voters", type=int, default=20000, 
                         help="number of voters in simulation")
        size.add_argument("--n-candidates", type=int, default=300, 
                         help="number of candidates")
        size.add_argument("--n-parties", type=int, default=12, 
                         help="number of political parties")
        size.add_argument("--n-bairros", type=int, default=12, 
                         help="number of neighborhoods")
        size.add_argument("--n-seats", type=int, default=21, 
                         help="number of council seats")
        size.add_argument("--n-steps", type=int, default=12, 
                         help="number of campaign steps to simulate")
    
    def _add_network_args(self, p: argparse.ArgumentParser) -> None:
        """Add social network parameters."""
        net = p.add_argument_group("Social Network")
        net.add_argument("--k-neighbors", type=int, default=10, 
                        help="number of neighbors per voter in social network (must be even)")
        net.add_argument("--rewire-prob", type=float, default=0.08, 
                        help="probability of rewiring in small-world network")
    
    def _add_data_args(self, p: argparse.ArgumentParser) -> None:
        """Add empirical input data arguments."""
        data = p.add_argument_group("Empirical Inputs")
        data.add_argument("--party-csv", type=str, default="", 
                         help="path to CSV with party data (generates synthetic if not provided)")
        data.add_argument("--candidate-csv", type=str, default="", 
                         help="path to CSV with candidate data (generates synthetic if not provided)")
    
    def _add_party_args(self, p: argparse.ArgumentParser) -> None:
        """Add party generation parameters."""
        party = p.add_argument_group("Party Generation")
        party.add_argument("--party-budget-mean", type=float, default=200000.0, 
                          help="mean party budget (lognormal distribution)")
        party.add_argument("--party-budget-sigma", type=float, default=0.7, 
                          help="sigma for party budget distribution")
        party.add_argument("--party-concentration-mean", type=float, default=1.2, 
                          help="mean strategic concentration (higher = more focus on top candidates)")
    
    def _add_candidate_args(self, p: argparse.ArgumentParser) -> None:
        """Add candidate generation parameters."""
        cand = p.add_argument_group("Candidate Generation")
        cand.add_argument("--candidate-ideology-sigma", type=float, default=0.18, 
                         help="standard deviation of candidate ideology around party ideology")
        cand.add_argument("--incumbency-prob", type=float, default=0.08, 
                         help="probability a candidate is incumbent")
        cand.add_argument("--initial-capital-mean", type=float, default=12000.0, 
                         help="mean initial capital")
        cand.add_argument("--cpf-mean", type=float, default=4000.0, 
                         help="mean individual donations")
        cand.add_argument("--cpf-sigma", type=float, default=1.0, 
                         help="sigma for individual donations")
        cand.add_argument("--cnpj-mean", type=float, default=2500.0, 
                         help="mean corporate donations")
        cand.add_argument("--cnpj-sigma", type=float, default=1.0, 
                         help="sigma for corporate donations")
        cand.add_argument("--non-original-mean", type=float, default=18000.0, 
                         help="mean other donations")
        cand.add_argument("--non-original-sigma", type=float, default=1.0, 
                         help="sigma for other donations")
    
    def _add_voter_args(self, p: argparse.ArgumentParser) -> None:
        """Add voter generation parameters."""
        voter = p.add_argument_group("Voter Generation")
        voter.add_argument("--voter-ideology-sigma", type=float, default=0.25, 
                          help="standard deviation of voter ideology around preferred party")
    
    def _add_resource_args(self, p: argparse.ArgumentParser) -> None:
        """Add resource weight parameters."""
        res = p.add_argument_group("Resource Weights")
        res.add_argument("--weight-cpf", type=float, default=1.0, 
                        help="weight for individual donations")
        res.add_argument("--weight-cnpj", type=float, default=1.0, 
                        help="weight for corporate donations")
        res.add_argument("--weight-non-original", type=float, default=1.1, 
                        help="weight for other donations")
        res.add_argument("--weight-party-transfer", type=float, default=1.2, 
                        help="weight for party transfers")
        res.add_argument("--weight-initial-capital", type=float, default=0.5, 
                        help="weight for initial capital")
    
    def _add_visibility_args(self, p: argparse.ArgumentParser) -> None:
        """Add visibility dynamics parameters."""
        vis = p.add_argument_group("Visibility Dynamics")
        vis.add_argument("--visibility-decay", type=float, default=0.25, 
                        help="decay rate of visibility per step")
        vis.add_argument("--a-resource", type=float, default=0.09, 
                        help="resource contribution to visibility")
        vis.add_argument("--b-quality", type=float, default=0.20, 
                        help="quality contribution to visibility")
        vis.add_argument("--c-party-org", type=float, default=0.16, 
                        help="party organization contribution to visibility")
        vis.add_argument("--d-local", type=float, default=0.12, 
                        help="local presence contribution to visibility")
    
    def _add_territorial_args(self, p: argparse.ArgumentParser) -> None:
        """Add territorial field parameters."""
        terr = p.add_argument_group("Territorial Field")
        terr.add_argument("--territorial-background", type=float, default=0.05, 
                         help="baseline territorial strength")
        terr.add_argument("--territorial-peak", type=float, default=0.90, 
                         help="peak territorial strength at home base")
        terr.add_argument("--territorial-decay", type=float, default=1.5, 
                         help="decay rate of territorial strength with distance")
    
    def _add_utility_args(self, p: argparse.ArgumentParser) -> None:
        """Add utility parameters (voter decision factors)."""
        util = p.add_argument_group("Utility Parameters")
        util.add_argument("--alpha-ideology", type=float, default=1.2, 
                         help="weight for ideological alignment")
        util.add_argument("--beta-visibility", type=float, default=1.0, 
                         help="weight for campaign visibility")
        util.add_argument("--gamma-territorial", type=float, default=1.1, 
                         help="weight for territorial affinity")
        util.add_argument("--delta-social", type=float, default=0.85, 
                         help="weight for social influence")
        util.add_argument("--eta-party", type=float, default=0.95, 
                         help="weight for party identification")
        util.add_argument("--mu-incumbency", type=float, default=0.5, 
                         help="weight for incumbency advantage")
        util.add_argument("--kappa-viability", type=float, default=0.7, 
                         help="weight for strategic voting (viability consideration)")
    
    def _add_strategic_args(self, p: argparse.ArgumentParser) -> None:
        """Add strategic voting parameters."""
        strat = p.add_argument_group("Strategic Voting")
        strat.add_argument("--f-strategic", type=float, default=0.35, 
                          help="fraction of voters engaging in strategic voting")
        strat.add_argument("--rho1", type=float, default=0.45, 
                          help="weight for current vote share in viability calculation")
        strat.add_argument("--rho2", type=float, default=0.35, 
                          help="weight for social support in viability calculation")
        strat.add_argument("--rho3", type=float, default=0.20, 
                          help="weight for party strength in viability calculation")
    
    def _add_choice_args(self, p: argparse.ArgumentParser) -> None:
        """Add choice noise and attention parameters."""
        choice = p.add_argument_group("Choice Parameters")
        choice.add_argument("--tau", type=float, default=0.7, 
                           help="temperature for softmax choice (lower = more deterministic)")
        choice.add_argument("--noise-sigma", type=float, default=0.03, 
                           help="standard deviation of random noise in utility")
        choice.add_argument("--shortlist-size", type=int, default=35, 
                           help="number of candidates voters consider (0 = all)")
    
    def _add_output_args(self, p: argparse.ArgumentParser) -> None:
        """Add output options."""
        out = p.add_argument_group("Output Options")
        out.add_argument("--save-voters", action="store_true", 
                        help="save full voter results (may be large)")
        out.add_argument("--voter-sample-size", type=int, default=2000, 
                        help="sample size for voter results when not saving all")
    
    def parse_args(self, args: list = None) -> argparse.Namespace:
        """
        Parse command-line arguments.
        
        Args:
            args: Command-line arguments (defaults to sys.argv[1:])
            
        Returns:
            Parsed arguments namespace
        """
        self.args = self._parser.parse_args(args)
        self._validate()
        return self.args
    
    def _validate(self) -> None:
        """Validate parameter consistency."""
        if self.args is None:
            return
        
        # Check that k_neighbors is even
        if self.args.k_neighbors % 2 != 0:
            raise ValueError(f"k_neighbors must be even, got {self.args.k_neighbors}")
        
        # Check that shortlist_size is valid
        if self.args.shortlist_size < 0:
            raise ValueError(f"shortlist_size must be >= 0, got {self.args.shortlist_size}")
        
        # Check that probabilities are between 0 and 1
        for param in ['rewire_prob', 'f_strategic', 'rho1', 'rho2', 'rho3']:
            value = getattr(self.args, param)
            if not 0 <= value <= 1:
                raise ValueError(f"{param} must be between 0 and 1, got {value}")
        
        # Check that weights are non-negative
        for param in ['weight_cpf', 'weight_cnpj', 'weight_non_original', 
                      'weight_party_transfer', 'weight_initial_capital']:
            if getattr(self.args, param) < 0:
                raise ValueError(f"{param} must be non-negative")
    
    # Property methods for convenient access to parameter groups
    @property
    def basic(self) -> Dict[str, object]:
        """Get basic run control parameters."""
        if self.args is None:
            return {}
        return {
            'output_dir': self.args.output_dir,
            'seed': self.args.seed,
            'cycle': self.args.cycle
        }
    
    @property
    def sizes(self) -> Dict[str, int]:
        """Get size-related parameters."""
        if self.args is None:
            return {}
        return {
            'n_voters': self.args.n_voters,
            'n_candidates': self.args.n_candidates,
            'n_parties': self.args.n_parties,
            'n_bairros': self.args.n_bairros,
            'n_seats': self.args.n_seats,
            'n_steps': self.args.n_steps
        }
    
    @property
    def network(self) -> Dict[str, object]:
        """Get social network parameters."""
        if self.args is None:
            return {}
        return {
            'k_neighbors': self.args.k_neighbors,
            'rewire_prob': self.args.rewire_prob
        }
    
    @property
    def utility(self) -> Dict[str, float]:
        """Get utility weights."""
        if self.args is None:
            return {}
        return {
            'alpha_ideology': self.args.alpha_ideology,
            'beta_visibility': self.args.beta_visibility,
            'gamma_territorial': self.args.gamma_territorial,
            'delta_social': self.args.delta_social,
            'eta_party': self.args.eta_party,
            'mu_incumbency': self.args.mu_incumbency,
            'kappa_viability': self.args.kappa_viability
        }
    
    # Add more properties as needed...
    
    def to_dict(self) -> Dict[str, object]:
        """Export all parameters as a dictionary."""
        if self.args is None:
            return {}
        return vars(self.args).copy()
    
    def save_to_file(self, filepath: Path) -> None:
        """Save parameters to JSON file."""
        if self.args is None:
            raise ValueError("No arguments parsed yet")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vars(self.args), f, indent=2)
