#!/usr/bin/env python3
"""
Main entry point for the Pelotas Election Agent-Based Model.
"""

from models.pelotas_election_abm import PelotasElectionABM
from features.parametersystem import ParameterSystem
from pathlib import Path
import json


def main():
    """Run the election simulation."""
    param_system = ParameterSystem()
    args = param_system.parse_args()
    
    # Save parameters for reproducibility
    output_dir = Path(args.output_dir)
    param_system.save_to_file(output_dir / "params.json")
    
    # Run simulation
    model = PelotasElectionABM(args)
    summary = model.run()
    
    # Print summary
    print("\n" + "="*50)
    print("SIMULATION SUMMARY")
    print("="*50)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
