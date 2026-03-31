#!/usr/bin/env python3
"""
Agent-based model for municipal council elections in Pelotas.

Implements a three-level model with voters, candidates, and parties.
The dynamics follows the discussion in the accompanying article:
- campaign finance affects visibility, not votes directly;
- voters choose candidates from ideology, territorial affinity, social influence,
  party affinity, incumbency, and campaign visibility;
- final outcomes are determined by proportional aggregation rules, including the
  electoral quotient (EQ) and party quotient (QP).

Main outputs:
- candidate_results.csv
- party_results.csv
- voter_results.csv (optional, sampled if electorate is large)
- summary.json
- timeseries_visibility.csv
- plots/*.png

The code is self-contained and can run without empirical data. If empirical candidate
or party tables are available, they can be supplied via CSV files.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from features.data_loader import DataLoader as DL
from features.plot_generator import PlotGenerator as PG

# -----------------------------
# Helpers
# -----------------------------

def ensure_dir(path: Path) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Directory path to ensure existence
    """
    path.mkdir(parents=True, exist_ok=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function for normalization.
    
    Args:
        x: Input array
        
    Returns:
        Sigmoid-transformed values between 0 and 1
    """
    return 1.0 / (1.0 + np.exp(-x))


def softmax_sample(logits: np.ndarray, rng: np.random.Generator, tau: float) -> int:
    """
    Não está sendo usado
    Sample from a softmax distribution with temperature.
    
    Args:
        logits: Utility values for each option
        rng: Random number generator
        tau: Temperature parameter (lower = more deterministic)
        
    Returns:
        Index of chosen option
    """
    z = logits / max(tau, 1e-9)
    z = z - np.max(z)
    probs = np.exp(z)
    probs /= probs.sum()
    return int(rng.choice(len(logits), p=probs))


def ring_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Parece que não está usando.
    Calculate circular distance on a ring (for ideology space).
    
    Args:
        a: First array of positions
        b: Second array of positions
        
    Returns:
        Normalized distances between 0 and 1
    """
    d = np.abs(a - b)
    return np.minimum(d, 2.0 - d) / 2.0


def make_small_world_neighbors(n: int, k: int, p_rewire: float, rng: np.random.Generator) -> List[np.ndarray]:
    """
    Create a Watts-Strogatz small-world network for social influence.
    Aqui podemos criar a classe que cria o grafo.
    Args:
        n: Number of nodes (voters)
        k: Number of neighbors per node (must be even)
        p_rewire: Probability of rewiring each edge
        rng: Random number generator
        
    Returns:
        List of arrays containing neighbor indices for each node
        
    Raises:
        ValueError: If k is not even
    """
    if k % 2 != 0:
        raise ValueError("k_neighbors must be even for the small-world construction.")
    
    neighbors = [set() for _ in range(n)]
    half = k // 2
    
    # Create regular ring lattice
    for i in range(n):
        for d in range(1, half + 1):
            j = (i + d) % n
            neighbors[i].add(j)
            neighbors[j].add(i)
    
    # Rewire edges with probability p_rewire
    for i in range(n):
        for d in range(1, half + 1):
            j = (i + d) % n
            if rng.random() < p_rewire:
                possible = list(set(range(n)) - {i} - neighbors[i])
                if possible:
                    new_j = int(rng.choice(possible))
                    neighbors[i].discard(j)
                    neighbors[j].discard(i)
                    neighbors[i].add(new_j)
                    neighbors[new_j].add(i)
    
    return [np.array(sorted(list(s)), dtype=np.int32) for s in neighbors]


def allocate_seats_largest_remainder(
    party_votes: pd.Series,
    n_seats: int,
) -> Tuple[pd.Series, float, pd.Series]:
    """
    Allocate seats using the largest remainder method (Hare quota).
    
    This is the standard method for proportional representation in Brazilian
    municipal elections.
    
    Args:
        party_votes: Series of votes per party
        n_seats: Total number of seats to allocate
        
    Returns:
        Tuple containing:
        - seats: Number of seats allocated to each party
        - eq: Electoral quotient (total votes / total seats)
        - remainders: Remainder values for each party
    """
    total_valid = float(party_votes.sum())
    eq = total_valid / n_seats if n_seats > 0 else 0.0
    
    if eq <= 0:
        seats = pd.Series(0, index=party_votes.index, dtype=int)
        remainders = pd.Series(0.0, index=party_votes.index)
        return seats, eq, remainders

    # First allocation: integer division by electoral quotient
    qp = np.floor(party_votes / eq).astype(int)
    seats_assigned = int(qp.sum())
    seats = qp.copy()
    remainders = party_votes / eq - qp

    # Second allocation: distribute remaining seats by largest remainders
    remaining = n_seats - seats_assigned
    if remaining > 0:
        order = remainders.sort_values(ascending=False).index.tolist()
        for idx in order[:remaining]:
            seats.loc[idx] += 1
    elif remaining < 0:
        # Safety guard: remove seats if over-allocated (rare)
        order = remainders.sort_values(ascending=True).index.tolist()
        for idx in order[: abs(remaining)]:
            seats.loc[idx] = max(0, seats.loc[idx] - 1)

    return seats.astype(int), eq, remainders


def save_plot(x, y, xlabel, ylabel, title, path: Path) -> None:
    """
    Save a simple line plot to file.
    
    Args:
        x: X-axis data
        y: Y-axis data
        xlabel: Label for X-axis
        ylabel: Label for Y-axis
        title: Plot title
        path: Output file path
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, y, marker="o", linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


# -----------------------------
# Core model
# -----------------------------

class PelotasElectionABM:
    """
    Main agent-based model class for Pelotas municipal elections.
    
    This class orchestrates the entire simulation including:
    - Generating or loading parties, candidates, and voters
    - Creating social influence network
    - Running the campaign dynamics
    - Computing election results
    - Saving outputs and generating plots
    
    The model follows a sequential update process where:
    1. Campaign visibility is updated based on resources, quality, etc.
    2. Social support is aggregated from previous voting choices
    3. Voters update their choices based on multiple factors
    4. Process repeats for n_steps iterations
    """
    
    def __init__(self, args: argparse.Namespace):
        """
        Initialize the model with configuration parameters.
        
        Args:
            args: Command-line arguments with all model parameters
        """
        self.args = args
        self.rng = np.random.default_rng(args.seed)
        self.output_dir = Path(args.output_dir)
        ensure_dir(self.output_dir)
        ensure_dir(self.output_dir / "plots")

                # Usa o DataLoader
        self.data_loader = DL(args, self.rng)

        self.plot_generator = PG(self.output_dir)
        
        # Carrega os dados
        self.parties = self.data_loader.load_parties()
        self.candidates = self.data_loader.load_candidates(self.parties)
        self.voters = self.data_loader.load_voters(self.parties)
        
        # Create social influence network
        self.neighbors = make_small_world_neighbors(
            args.n_voters,
            args.k_neighbors,
            args.rewire_prob,
            self.rng,
        )

        # Precompute arrays for performance
        self._precompute_candidate_arrays()
        self._precompute_voter_arrays()

    def _precompute_candidate_arrays(self) -> None:
        """
        Precompute candidate arrays for faster access during simulation.
        
        This method extracts candidate attributes into numpy arrays for
        vectorized operations and computes the territorial strength matrix.
        """
        c = self.candidates
        p = self.parties.set_index("party_id")
        
        # Extract basic candidate attributes
        self.candidate_ids = c["candidate_id"].to_numpy()
        self.candidate_names = c["name"].to_numpy()
        self.candidate_party = c["party_id"].to_numpy(dtype=int)
        self.candidate_ideology = c["ideology"].to_numpy(dtype=float)
        self.candidate_incumbency = c["incumbency"].to_numpy(dtype=float)
        self.candidate_quality = c["quality"].to_numpy(dtype=float)
        self.candidate_base_bairro = c["base_bairro"].to_numpy(dtype=int)
        self.candidate_effective_resources = c["effective_resources"].to_numpy(dtype=float)
        self.candidate_visibility = c["initial_visibility"].to_numpy(dtype=float).copy()
        
        # Party-level attributes for each candidate
        self.candidate_party_org = p.loc[self.candidate_party, "organization"].to_numpy(dtype=float)
        self.candidate_party_ideology = p.loc[self.candidate_party, "ideology"].to_numpy(dtype=float)
        self.n_candidates = len(c)

        # Build territorial strength matrix: each candidate has a home neighborhood
        # with decreasing influence in nearby neighborhoods
        self.territorial_strength = np.full(
            (self.n_candidates, self.args.n_bairros), 
            self.args.territorial_background, 
            dtype=float
        )
        
        for j, b0 in enumerate(self.candidate_base_bairro):
            for b in range(self.args.n_bairros):
                # Circular distance on neighborhood ring
                d = min(abs(b - b0), self.args.n_bairros - abs(b - b0))
                # Exponential decay from peak at home neighborhood
                self.territorial_strength[j, b] = (
                    self.args.territorial_background + 
                    self.args.territorial_peak * math.exp(-d / max(self.args.territorial_decay, 1e-9))
                )

    def _precompute_voter_arrays(self) -> None:
        """
        Precompute voter arrays for faster access during simulation.
        
        Extracts voter attributes into numpy arrays for vectorized operations.
        """
        v = self.voters
        self.voter_ideology = v["ideology"].to_numpy(dtype=float)
        self.voter_bairro = v["bairro"].to_numpy(dtype=int)
        self.voter_interest = v["interest"].to_numpy(dtype=float)
        self.voter_party_strength = v["party_strength"].to_numpy(dtype=float)
        self.voter_campaign_susc = v["campaign_susc"].to_numpy(dtype=float)
        self.voter_social_susc = v["social_susc"].to_numpy(dtype=float)
        self.voter_preferred_party = v["preferred_party"].to_numpy(dtype=int)
        self.n_voters = len(v)

    # -----------------------------
    # Simulation dynamics
    # -----------------------------

    def run(self) -> Dict[str, object]:
        """
        Execute the main simulation loop.
        
        The simulation proceeds through n_steps iterations where:
        1. Campaign visibility is updated for all candidates
        2. Social support is aggregated from previous choices
        3. Each voter updates their choice based on multiple factors
        4. Process repeats
        
        Returns:
            Dictionary with summary statistics of the simulation results
        """
        args = self.args
        rng = self.rng
        visibility_history = []
        
        # Initialize with random candidate choices
        current_choices = rng.integers(0, self.n_candidates, size=self.n_voters)

        for t in range(args.n_steps):
            # ---------- Step 1: Update campaign visibility ----------
            # Local term: strength in a random neighborhood
            local_term = self.territorial_strength[:, rng.integers(0, args.n_bairros)]
            res_term = np.log1p(self.candidate_effective_resources)  # Log transformation for diminishing returns
            
            self.candidate_visibility = (
                (1.0 - args.visibility_decay) * self.candidate_visibility  # Decay
                + args.a_resource * res_term  # Resources contribution
                + args.b_quality * self.candidate_quality  # Quality contribution
                + args.c_party_org * self.candidate_party_org  # Party organization
                + args.d_local * local_term  # Local visibility
            )
            self.candidate_visibility = np.clip(self.candidate_visibility, 0.0, None)
            visibility_history.append(self.candidate_visibility.copy())

            # ---------- Step 2: Aggregate social support ----------
            # Count votes from previous round
            candidate_counts_prev = np.bincount(current_choices, minlength=self.n_candidates).astype(float)
            candidate_share_prev = candidate_counts_prev / max(candidate_counts_prev.sum(), 1.0)
            
            # Calculate party strength from previous votes
            party_votes_prev = pd.Series(candidate_counts_prev).groupby(self.candidate_party).sum()
            party_strength_prev = {int(pid): float(v) / self.n_voters for pid, v in party_votes_prev.items()}

            # ---------- Step 3: Update voters sequentially ----------
            new_choices = np.empty_like(current_choices)
            
            for i in range(self.n_voters):
                # Social influence from neighbors
                neigh = self.neighbors[i]
                neigh_choices = current_choices[neigh]
                social_by_candidate = np.bincount(neigh_choices, minlength=self.n_candidates).astype(float)
                if social_by_candidate.sum() > 0:
                    social_by_candidate /= social_by_candidate.sum()

                # Calculate utility components for each candidate
                
                # 1. Ideological distance (negative utility for distance)
                ideology_term = -args.alpha_ideology * np.abs(self.voter_ideology[i] - self.candidate_ideology)
                
                # 2. Campaign visibility (scaled by voter's susceptibility)
                visibility_term = args.beta_visibility * self.voter_campaign_susc[i] * self.candidate_visibility
                
                # 3. Territorial affinity (candidate's strength in voter's neighborhood)
                territorial_term = args.gamma_territorial * self.territorial_strength[:, self.voter_bairro[i]]
                
                # 4. Social influence (what neighbors are doing)
                social_term = args.delta_social * self.voter_social_susc[i] * social_by_candidate
                
                # 5. Party affinity (preference for own party)
                party_term = args.eta_party * self.voter_party_strength[i] * (self.candidate_party == self.voter_preferred_party[i])
                
                # 6. Incumbency advantage
                incumbency_term = args.mu_incumbency * self.candidate_incumbency

                # 7. Strategic voting (viability consideration)
                viability_party = np.array([party_strength_prev.get(int(pid), 0.0) for pid in self.candidate_party])
                viability_term = args.kappa_viability * args.f_strategic * (
                    args.rho1 * candidate_share_prev +      # Current vote share
                    args.rho2 * social_by_candidate +       # Social support
                    args.rho3 * viability_party              # Party strength
                )

                # Add random noise
                noise = rng.normal(0.0, args.noise_sigma, self.n_candidates)
                
                # Combine all components
                utility = (
                    ideology_term
                    + visibility_term
                    + territorial_term
                    + social_term
                    + party_term
                    + incumbency_term
                    + viability_term
                    + noise
                )

                # Optional: limit attention to most visible candidates
                if args.shortlist_size < self.n_candidates:
                    shortlist_score = self.candidate_visibility + 0.5 * self.territorial_strength[:, self.voter_bairro[i]]
                    shortlist = np.argpartition(shortlist_score, -args.shortlist_size)[-args.shortlist_size:]
                    masked = np.full(self.n_candidates, -1e12)
                    masked[shortlist] = utility[shortlist]
                    utility = masked

                # Choose candidate using softmax
                new_choices[i] = softmax_sample(utility, rng, args.tau)

            current_choices = new_choices

        # ---------- Step 4: Compute final results ----------
        # Count final votes
        candidate_votes = np.bincount(current_choices, minlength=self.n_candidates).astype(int)
        
        # Build candidate results dataframe
        candidate_df = self.candidates.copy()
        candidate_df["votes"] = candidate_votes
        candidate_df["final_visibility"] = self.candidate_visibility
        candidate_df["vote_share"] = candidate_df["votes"] / self.n_voters

        # Aggregate party results
        party_df = self._aggregate_parties(candidate_df)
        
        # Allocate seats using proportional representation
        seats, eq, remainders = allocate_seats_largest_remainder(
            party_df.set_index("party_id")["votes"], 
            args.n_seats
        )
        
        # Add electoral calculations to party dataframe
        party_df = party_df.set_index("party_id")
        party_df["electoral_quotient"] = eq
        party_df["party_quotient_floor"] = np.floor(party_df["votes"] / eq).astype(int) if eq > 0 else 0
        party_df["remainder"] = remainders
        party_df["seats"] = seats
        party_df = party_df.reset_index()

        # Assign elected status to candidates
        candidate_df = self._assign_elected(candidate_df, party_df)
        candidate_df["rank_within_party"] = candidate_df.groupby("party_id")["votes"].rank(
            method="first", ascending=False
        ).astype(int)
        
        # Sort results
        candidate_df = candidate_df.sort_values(
            ["elected", "votes", "final_visibility"], 
            ascending=[False, False, False]
        ).reset_index(drop=True)

        # Voter results with their choices
        voter_df = self.voters.copy()
        voter_df["choice_candidate_id"] = current_choices
        voter_df["choice_party_id"] = self.candidate_party[current_choices]

        # Save time series of visibility
        visibility_hist = np.asarray(visibility_history)
        vis_df = pd.DataFrame(visibility_hist, columns=[f"cand_{cid}" for cid in self.candidate_ids])
        vis_df.insert(0, "step", np.arange(len(vis_df), dtype=int))

        # ---------- Step 5: Save outputs ----------
        candidate_df.to_csv(self.output_dir / "candidate_results.csv", index=False)
        party_df.to_csv(self.output_dir / "party_results.csv", index=False)
        vis_df.to_csv(self.output_dir / "timeseries_visibility.csv", index=False)
        
        if args.save_voters:
            voter_df.to_csv(self.output_dir / "voter_results.csv", index=False)
        else:
            sample_n = min(args.voter_sample_size, len(voter_df))
            voter_df.sample(sample_n, random_state=args.seed).to_csv(
                self.output_dir / "voter_results_sample.csv", index=False
            )

        # Generate plots
        #self._make_plots(candidate_df, party_df, visibility_hist)
        # No final, em vez de self._make_plots(...)
        self.plot_generator.generate_all(candidate_df, party_df, visibility_hist)

        # Build and save summary
        summary = self._build_summary(candidate_df, party_df, eq)
        with open(self.output_dir / "summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        with open(self.output_dir / "params.json", "w", encoding="utf-8") as fh:
            json.dump(vars(args), fh, indent=2)

        return summary

    def _aggregate_parties(self, candidate_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate candidate results to party level.
        
        Args:
            candidate_df: DataFrame with candidate results
            
        Returns:
            DataFrame with party-level aggregates
        """
        party_df = candidate_df.groupby("party_id", as_index=False).agg(
            votes=("votes", "sum"),
            n_candidates=("candidate_id", "count"),
            mean_visibility=("final_visibility", "mean"),
            mean_resources=("effective_resources", "mean"),
        )
        party_df = party_df.merge(self.parties, on="party_id", how="left")
        party_df["vote_share"] = party_df["votes"] / party_df["votes"].sum()
        return party_df

    def _assign_elected(self, candidate_df: pd.DataFrame, party_df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign elected status to candidates based on party seat allocation.
        
        Args:
            candidate_df: DataFrame with candidate results
            party_df: DataFrame with party seat allocation
            
        Returns:
            Updated candidate DataFrame with 'elected' column
        """
        party_seats = party_df.set_index("party_id")["seats"].to_dict()
        candidate_df = candidate_df.copy()
        candidate_df["elected"] = 0
        
        for pid, sub in candidate_df.groupby("party_id"):
            seats = int(party_seats.get(pid, 0))
            if seats <= 0:
                continue
            # Elect top candidates by votes (and visibility as tiebreaker)
            winners = sub.sort_values(["votes", "final_visibility"], ascending=[False, False]).head(seats).index
            candidate_df.loc[winners, "elected"] = 1
            
        return candidate_df

    def _build_summary(self, candidate_df: pd.DataFrame, party_df: pd.DataFrame, eq: float) -> Dict[str, object]:
        """
        Build summary statistics of the election results.
        
        Args:
            candidate_df: DataFrame with candidate results
            party_df: DataFrame with party results
            eq: Electoral quotient
            
        Returns:
            Dictionary with summary statistics
        """
        votes = candidate_df["votes"].to_numpy(dtype=float)
        n = len(votes)
        
        # Gini coefficient for vote concentration
        mean_votes = votes.mean() if n > 0 else 0.0
        gini = float(np.abs(votes[:, None] - votes[None, :]).sum() / (2 * n * max(votes.sum(), 1.0))) if n > 0 else 0.0
        
        # Top 10 vote share
        top10_share = float(np.sort(votes)[-min(10, n):].sum() / max(votes.sum(), 1.0)) if n > 0 else 0.0
        
        # Correlations
        corr_resources_votes = float(np.corrcoef(candidate_df["effective_resources"], candidate_df["votes"])[0, 1]) if n > 1 else 0.0
        corr_visibility_votes = float(np.corrcoef(candidate_df["final_visibility"], candidate_df["votes"])[0, 1]) if n > 1 else 0.0
        
        return {
            "cycle": self.args.cycle,
            "seed": self.args.seed,
            "n_voters": int(self.n_voters),
            "n_candidates": int(self.n_candidates),
            "n_parties": int(len(self.parties)),
            "n_seats": int(self.args.n_seats),
            "electoral_quotient": float(eq),
            "total_valid_votes": int(candidate_df["votes"].sum()),
            "elected_total": int(candidate_df["elected"].sum()),
            "gini_votes": gini,
            "top10_vote_share": top10_share,
            "corr_resources_votes": corr_resources_votes,
            "corr_visibility_votes": corr_visibility_votes,
            "largest_party_vote_share": float(party_df["vote_share"].max()),
            "largest_party_seats": int(party_df["seats"].max()),
            "n_parties_with_seats": int((party_df["seats"] > 0).sum()),
        }

    # def _make_plots(self, candidate_df: pd.DataFrame, party_df: pd.DataFrame, visibility_hist: np.ndarray) -> None:
    #     """
    #     Generate and save visualization plots.
        
    #     Args:
    #         candidate_df: DataFrame with candidate results
    #         party_df: DataFrame with party results
    #         visibility_hist: History of visibility scores over time
    #     """
    #     plots = self.output_dir / "plots"
        
    #     # Plot 1: Candidate vote distribution (Pareto-like)
    #     fig, ax = plt.subplots(figsize=(7, 4.5))
    #     sorted_votes = np.sort(candidate_df["votes"].to_numpy())[::-1]
    #     ax.plot(np.arange(1, len(sorted_votes) + 1), sorted_votes, marker="o", linewidth=1.2)
    #     ax.set_xlabel("candidate rank")
    #     ax.set_ylabel("votes")
    #     ax.set_title("Candidate vote distribution")
    #     ax.grid(alpha=0.3)
    #     fig.tight_layout()
    #     fig.savefig(plots / "candidate_vote_distribution.png", dpi=180)
    #     plt.close(fig)

    #     # Plot 2: Seats by party
    #     fig, ax = plt.subplots(figsize=(7.5, 4.5))
    #     party_sorted = party_df.sort_values("seats", ascending=False)
    #     ax.bar(party_sorted["name"], party_sorted["seats"])
    #     ax.set_xlabel("party")
    #     ax.set_ylabel("seats")
    #     ax.set_title("Seats by party")
    #     ax.tick_params(axis="x", rotation=45)
    #     fig.tight_layout()
    #     fig.savefig(plots / "party_seats.png", dpi=180)
    #     plt.close(fig)

    #     # Plot 3: Resources vs votes correlation
    #     fig, ax = plt.subplots(figsize=(6, 4.5))
    #     ax.scatter(candidate_df["effective_resources"], candidate_df["votes"], alpha=0.7)
    #     ax.set_xlabel("effective resources")
    #     ax.set_ylabel("votes")
    #     ax.set_title("Resources vs votes")
    #     ax.grid(alpha=0.3)
    #     fig.tight_layout()
    #     fig.savefig(plots / "resources_vs_votes.png", dpi=180)
    #     plt.close(fig)

    #     # Plot 4: Mean visibility over time
    #     save_plot(
    #         np.arange(visibility_hist.shape[0]),
    #         visibility_hist.mean(axis=1),
    #         "step",
    #         "mean visibility",
    #         "Average candidate visibility over time",
    #         plots / "mean_visibility_over_time.png",
    #     )


# -----------------------------
# CLI
# -----------------------------

def build_parser() -> argparse.ArgumentParser:
    """
    Build command-line argument parser with all model parameters.
    
    Returns:
        Configured ArgumentParser
    """
    p = argparse.ArgumentParser(description="Agent-based model for Pelotas council elections")

    # Basic run control
    p.add_argument("--output-dir", type=str, required=True, help="directory for outputs")
    p.add_argument("--seed", type=int, default=12345, help="random seed for reproducibility")
    p.add_argument("--cycle", type=int, default=2024, choices=[2016, 2020, 2024], 
                   help="election cycle (affects corporate donation rules)")

    # Sizes and scales
    p.add_argument("--n-voters", type=int, default=20000, help="number of voters in simulation")
    p.add_argument("--n-candidates", type=int, default=300, help="number of candidates")
    p.add_argument("--n-parties", type=int, default=12, help="number of political parties")
    p.add_argument("--n-bairros", type=int, default=12, help="number of neighborhoods")
    p.add_argument("--n-seats", type=int, default=21, help="number of council seats")
    p.add_argument("--n-steps", type=int, default=12, help="number of campaign steps to simulate")

    # Social network parameters
    p.add_argument("--k-neighbors", type=int, default=10, 
                   help="number of neighbors per voter in social network (must be even)")
    p.add_argument("--rewire-prob", type=float, default=0.08, 
                   help="probability of rewiring in small-world network")

    # Optional empirical inputs
    p.add_argument("--party-csv", type=str, default="", 
                   help="path to CSV with party data (generates synthetic if not provided)")
    p.add_argument("--candidate-csv", type=str, default="", 
                   help="path to CSV with candidate data (generates synthetic if not provided)")

    # Party generation parameters
    p.add_argument("--party-budget-mean", type=float, default=200000.0, 
                   help="mean party budget (lognormal distribution)")
    p.add_argument("--party-budget-sigma", type=float, default=0.7, 
                   help="sigma for party budget distribution")
    p.add_argument("--party-concentration-mean", type=float, default=1.2, 
                   help="mean strategic concentration (higher = more focus on top candidates)")

    # Candidate generation parameters
    p.add_argument("--candidate-ideology-sigma", type=float, default=0.18, 
                   help="standard deviation of candidate ideology around party ideology")
    p.add_argument("--incumbency-prob", type=float, default=0.08, 
                   help="probability a candidate is incumbent")
    p.add_argument("--initial-capital-mean", type=float, default=12000.0, 
                   help="mean initial capital")
    p.add_argument("--cpf-mean", type=float, default=4000.0, 
                   help="mean individual donations")
    p.add_argument("--cpf-sigma", type=float, default=1.0, 
                   help="sigma for individual donations")
    p.add_argument("--cnpj-mean", type=float, default=2500.0, 
                   help="mean corporate donations")
    p.add_argument("--cnpj-sigma", type=float, default=1.0, 
                   help="sigma for corporate donations")
    p.add_argument("--non-original-mean", type=float, default=18000.0, 
                   help="mean other donations")
    p.add_argument("--non-original-sigma", type=float, default=1.0, 
                   help="sigma for other donations")

    # Voter generation parameters
    p.add_argument("--voter-ideology-sigma", type=float, default=0.25, 
                   help="standard deviation of voter ideology around preferred party")

    # Resource weights for effective resources calculation
    p.add_argument("--weight-cpf", type=float, default=1.0, 
                   help="weight for individual donations")
    p.add_argument("--weight-cnpj", type=float, default=1.0, 
                   help="weight for corporate donations")
    p.add_argument("--weight-non-original", type=float, default=1.1, 
                   help="weight for other donations")
    p.add_argument("--weight-party-transfer", type=float, default=1.2, 
                   help="weight for party transfers")
    p.add_argument("--weight-initial-capital", type=float, default=0.5, 
                   help="weight for initial capital")

    # Visibility dynamics parameters
    p.add_argument("--visibility-decay", type=float, default=0.25, 
                   help="decay rate of visibility per step")
    p.add_argument("--a-resource", type=float, default=0.09, 
                   help="resource contribution to visibility")
    p.add_argument("--b-quality", type=float, default=0.20, 
                   help="quality contribution to visibility")
    p.add_argument("--c-party-org", type=float, default=0.16, 
                   help="party organization contribution to visibility")
    p.add_argument("--d-local", type=float, default=0.12, 
                   help="local presence contribution to visibility")

    # Territorial field parameters
    p.add_argument("--territorial-background", type=float, default=0.05, 
                   help="baseline territorial strength")
    p.add_argument("--territorial-peak", type=float, default=0.90, 
                   help="peak territorial strength at home base")
    p.add_argument("--territorial-decay", type=float, default=1.5, 
                   help="decay rate of territorial strength with distance")

    # Utility parameters (weights for different voter decision factors)
    p.add_argument("--alpha-ideology", type=float, default=1.2, 
                   help="weight for ideological alignment")
    p.add_argument("--beta-visibility", type=float, default=1.0, 
                   help="weight for campaign visibility")
    p.add_argument("--gamma-territorial", type=float, default=1.1, 
                   help="weight for territorial affinity")
    p.add_argument("--delta-social", type=float, default=0.85, 
                   help="weight for social influence")
    p.add_argument("--eta-party", type=float, default=0.95, 
                   help="weight for party identification")
    p.add_argument("--mu-incumbency", type=float, default=0.5, 
                   help="weight for incumbency advantage")
    p.add_argument("--kappa-viability", type=float, default=0.7, 
                   help="weight for strategic voting (viability consideration)")

    # Strategic voting parameters
    p.add_argument("--f-strategic", type=float, default=0.35, 
                   help="fraction of voters engaging in strategic voting")
    p.add_argument("--rho1", type=float, default=0.45, 
                   help="weight for current vote share in viability calculation")
    p.add_argument("--rho2", type=float, default=0.35, 
                   help="weight for social support in viability calculation")
    p.add_argument("--rho3", type=float, default=0.20, 
                   help="weight for party strength in viability calculation")

    # Choice noise and attention parameters
    p.add_argument("--tau", type=float, default=0.7, 
                   help="temperature for softmax choice (lower = more deterministic)")
    p.add_argument("--noise-sigma", type=float, default=0.03, 
                   help="standard deviation of random noise in utility")
    p.add_argument("--shortlist-size", type=int, default=35, 
                   help="number of candidates voters consider (0 = all)")

    # Output options
    p.add_argument("--save-voters", action="store_true", 
                   help="save full voter results (may be large)")
    p.add_argument("--voter-sample-size", type=int, default=2000, 
                   help="sample size for voter results when not saving all")

    return p


def main() -> None:
    """
    Main entry point for the simulation.
    
    Parses command-line arguments, initializes the model, runs the simulation,
    and prints summary statistics.
    """
    parser = build_parser()
    args = parser.parse_args()
    model = PelotasElectionABM(args)
    summary = model.run()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
