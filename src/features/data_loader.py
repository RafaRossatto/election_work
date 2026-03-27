# data_loader.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

class DataLoader:
    """Responsável por carregar ou gerar dados da simulação."""
    
    def __init__(self, args, rng: np.random.Generator):
        self.args = args
        self.rng = rng
    
    def load_parties(self) -> pd.DataFrame:
        """Carrega ou gera partidos."""
        if self.args.party_csv and Path(self.args.party_csv).exists():
            return pd.read_csv(self.args.party_csv)
        else:
            return self._generate_parties()
    
    def load_candidates(self, parties: pd.DataFrame) -> pd.DataFrame:
        """Carrega ou gera candidatos."""
        if self.args.candidate_csv and Path(self.args.candidate_csv).exists():
            df = pd.read_csv(self.args.candidate_csv)
        else:
            df = self._generate_candidates(parties)
        
        # Adiciona recursos efetivos (cálculo comum)
        df = self._add_effective_resources(df, parties)
        return df
    
    def load_voters(self, parties: pd.DataFrame) -> pd.DataFrame:
        """Gera eleitores (sempre sintéticos por enquanto)."""
        return self._generate_voters(parties)
    
    def _generate_parties(self) -> pd.DataFrame:
        """Gera partidos sintéticos."""
        n = self.args.n_parties
        names = [f"P{idx+1:02d}" for idx in range(n)]
        ideology = np.linspace(-0.9, 0.9, n)
        ideology += self.rng.normal(0, 0.08, n)
        org = np.clip(self.rng.normal(0.6, 0.15, n), 0.2, 1.0)
        budget = self.rng.lognormal(
            mean=np.log(self.args.party_budget_mean),
            sigma=self.args.party_budget_sigma,
            size=n
        )
        strategic_concentration = np.clip(
            self.rng.normal(self.args.party_concentration_mean, 0.25, n),
            0.1, 3.0
        )
        
        return pd.DataFrame({
            "party_id": np.arange(n, dtype=int),
            "name": names,
            "ideology": ideology,
            "organization": org,
            "central_budget": budget,
            "strategic_concentration": strategic_concentration,
        })
    
    def _generate_candidates(self, parties: pd.DataFrame) -> pd.DataFrame:
        """Gera candidatos sintéticos."""
        n = self.args.n_candidates
        party_ids = self.rng.choice(parties["party_id"].to_numpy(), size=n, replace=True)
        party_lookup = parties.set_index("party_id")
        party_ideology = party_lookup.loc[party_ids, "ideology"].to_numpy()
        
        cand_ideology = np.clip(
            party_ideology + self.rng.normal(0, self.args.candidate_ideology_sigma, n),
            -1.0, 1.0
        )
        
        cpf = self.rng.lognormal(mean=np.log(self.args.cpf_mean), sigma=self.args.cpf_sigma, size=n)
        cnpj = self.rng.lognormal(mean=np.log(self.args.cnpj_mean), sigma=self.args.cnpj_sigma, size=n)
        non_original = self.rng.lognormal(mean=np.log(self.args.non_original_mean), sigma=self.args.non_original_sigma, size=n)
        
        if self.args.cycle in {2020, 2024}:
            cnpj *= 0.0
        
        total = cpf + cnpj + non_original
        
        df = pd.DataFrame({
            "candidate_id": np.arange(n, dtype=int),
            "name": [f"cand_{i:03d}" for i in range(n)],
            "party_id": party_ids,
            "ideology": cand_ideology,
            "incumbency": self.rng.binomial(1, self.args.incumbency_prob, size=n),
            "priority": np.clip(self.rng.lognormal(mean=0.0, sigma=0.5, size=n), 0.2, 5.0),
            "quality": np.clip(self.rng.normal(0.6, 0.18, size=n), 0.05, 1.2),
            "initial_capital": self.rng.lognormal(mean=np.log(self.args.initial_capital_mean), sigma=0.6, size=n),
            "cpf_donations": cpf,
            "cnpj_donations": cnpj,
            "non_original_donations": non_original,
            "total_donations": total,
            "base_bairro": self.rng.integers(0, self.args.n_bairros, size=n),
            "gender": self.rng.binomial(1, 0.4, size=n),
            "party_base": self.rng.integers(1, 4, size=n),
            "initial_visibility": np.clip(self.rng.normal(0.15, 0.08, size=n), 0.01, 0.5),
        })
        
        return df
    
    def _add_effective_resources(self, candidates_df: pd.DataFrame, parties: pd.DataFrame) -> pd.DataFrame:
        """Adiciona coluna de recursos efetivos."""
        party_info = parties.set_index("party_id")
        df = candidates_df.copy()
        df["party_transfer"] = 0.0
        
        for pid, sub in df.groupby("party_id"):
            phi = float(party_info.loc[pid, "strategic_concentration"])
            budget = float(party_info.loc[pid, "central_budget"])
            w = np.power(np.clip(sub["priority"].to_numpy(), 1e-6, None), phi)
            alloc = budget * w / w.sum() if w.sum() > 0 else 0
            df.loc[sub.index, "party_transfer"] = alloc
        
        df["effective_resources"] = (
            self.args.weight_cpf * df["cpf_donations"]
            + self.args.weight_cnpj * df["cnpj_donations"]
            + self.args.weight_non_original * df["non_original_donations"]
            + self.args.weight_party_transfer * df["party_transfer"]
            + self.args.weight_initial_capital * df["initial_capital"]
        )
        
        return df
    
    def _generate_voters(self, parties: pd.DataFrame) -> pd.DataFrame:
        """Gera eleitores sintéticos."""
        n = self.args.n_voters
        
        party_probs = parties["organization"].to_numpy()
        party_probs = party_probs / party_probs.sum()
        preferred_party = self.rng.choice(parties["party_id"].to_numpy(), size=n, p=party_probs)
        
        party_lookup = parties.set_index("party_id")
        base_ideology = party_lookup.loc[preferred_party, "ideology"].to_numpy()
        voter_ideology = np.clip(
            base_ideology + self.rng.normal(0, self.args.voter_ideology_sigma, n),
            -1.0, 1.0
        )
        
        bairros = self.rng.integers(0, self.args.n_bairros, size=n)
        interest = np.clip(self.rng.beta(2.0, 2.0, size=n), 0.02, 0.98)
        party_id_strength = np.clip(self.rng.beta(2.0, 2.5, size=n), 0.0, 1.0)
        campaign_susc = np.clip(self.rng.beta(2.5, 2.0, size=n), 0.0, 1.0)
        social_susc = np.clip(self.rng.beta(2.0, 2.0, size=n), 0.0, 1.0)
        
        return pd.DataFrame({
            "voter_id": np.arange(n, dtype=int),
            "ideology": voter_ideology,
            "bairro": bairros,
            "interest": interest,
            "party_strength": party_id_strength,
            "campaign_susc": campaign_susc,
            "social_susc": social_susc,
            "preferred_party": preferred_party,
        })