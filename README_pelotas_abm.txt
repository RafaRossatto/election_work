Files included:
- pelotas_abm.py: complete agent-based model for municipal council elections.
- run_pelotas_abm_grid.sh: bash script for running a parameter grid.

Quick start:
1) Single run
   python3 pelotas_abm.py --output-dir run_test
   
   -- to compare with real data:
   python pelotas_abm.py --candidates candidates.csv --parties parties.csv

2) Grid of runs
   bash run_pelotas_abm_grid.sh

Using empirical tables:
- party_csv must contain columns:
  party_id,name,ideology,organization,central_budget,strategic_concentration
  
- candidate_csv must contain columns:
  candidate_id,name,party_id,ideology,incumbency,priority,quality,initial_capital,
  cpf_donations,cnpj_donations,non_original_donations,total_donations,base_bairro,
  gender,party_base,initial_visibility

Main outputs per run:
- candidate_results.csv
- party_results.csv
- summary.json
- params.json
- timeseries_visibility.csv
- plots/*.png
- voter_results_sample.csv or voter_results.csv
