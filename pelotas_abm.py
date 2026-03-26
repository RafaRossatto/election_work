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
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Dataclasses
# -----------------------------

@dataclass
class Party:
    party_id: int
    name: str
    ideology: float
    organization: float
    central_budget: float
    strategic_concentration: float


@dataclass
class Candidate:
    candidate_id: int
    name: str
    party_id: int
    ideology: float
    incumbency: int
    priority: float
    quality: float
    initial_capital: float
    cpf_donations: float
    cnpj_donations: float
    non_original_donations: float
    total_donations: float
    base_bairro: int
    gender: int
    party_base: int
    initial_visibility: float


# -----------------------------
# Helpers
# -----------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax_sample(logits: np.ndarray, rng: np.random.Generator, tau: float) -> int:
    z = logits / max(tau, 1e-9)
    z = z - np.max(z)
    probs = np.exp(z)
    probs /= probs.sum()
    return int(rng.choice(len(logits), p=probs))


def ring_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.abs(a - b)
    return np.minimum(d, 2.0 - d) / 2.0


def make_small_world_neighbors(n: int, k: int, p_rewire: float, rng: np.random.Generator) -> List[np.ndarray]:
    if k % 2 != 0:
        raise ValueError("k_neighbors must be even for the small-world construction.")
    neighbors = [set() for _ in range(n)]
    half = k // 2
    for i in range(n):
        for d in range(1, half + 1):
            j = (i + d) % n
            neighbors[i].add(j)
            neighbors[j].add(i)
    # Rewire edges from i to j for positive direction only
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
    total_valid = float(party_votes.sum())
    eq = total_valid / n_seats if n_seats > 0 else 0.0
    if eq <= 0:
        seats = pd.Series(0, index=party_votes.index, dtype=int)
        remainders = pd.Series(0.0, index=party_votes.index)
        return seats, eq, remainders

    qp = np.floor(party_votes / eq).astype(int)
    seats_assigned = int(qp.sum())
    seats = qp.copy()
    remainders = party_votes / eq - qp

    remaining = n_seats - seats_assigned
    if remaining > 0:
        order = remainders.sort_values(ascending=False).index.tolist()
        for idx in order[:remaining]:
            seats.loc[idx] += 1
    elif remaining < 0:
        # Very rare due to floor, but keep a safe guard.
        order = remainders.sort_values(ascending=True).index.tolist()
        for idx in order[: abs(remaining)]:
            seats.loc[idx] = max(0, seats.loc[idx] - 1)

    return seats.astype(int), eq, remainders


def save_plot(x, y, xlabel, ylabel, title, path: Path) -> None:
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
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.rng = np.random.default_rng(args.seed)
        self.output_dir = Path(args.output_dir)
        ensure_dir(self.output_dir)
        ensure_dir(self.output_dir / "plots")

        self.parties = self._load_or_generate_parties()
        self.candidates = self._load_or_generate_candidates(self.parties)
        self.voters = self._generate_voters(self.parties)
        self.neighbors = make_small_world_neighbors(
            args.n_voters,
            args.k_neighbors,
            args.rewire_prob,
            self.rng,
        )

        self._precompute_candidate_arrays()
        self._precompute_voter_arrays()

    # -----------------------------
    # Data initialization
    # -----------------------------

    def _load_or_generate_parties(self) -> pd.DataFrame:
        if self.args.party_csv and Path(self.args.party_csv).exists():
            df = pd.read_csv(self.args.party_csv)
        else:
            names = [f"P{idx+1:02d}" for idx in range(self.args.n_parties)]
            ideology = np.linspace(-0.9, 0.9, self.args.n_parties)
            ideology += self.rng.normal(0, 0.08, self.args.n_parties)
            org = np.clip(self.rng.normal(0.6, 0.15, self.args.n_parties), 0.2, 1.0)
            budget = self.rng.lognormal(mean=np.log(self.args.party_budget_mean), sigma=self.args.party_budget_sigma, size=self.args.n_parties)
            strategic_concentration = np.clip(self.rng.normal(self.args.party_concentration_mean, 0.25, self.args.n_parties), 0.1, 3.0)
            df = pd.DataFrame({
                "party_id": np.arange(self.args.n_parties, dtype=int),
                "name": names,
                "ideology": ideology,
                "organization": org,
                "central_budget": budget,
                "strategic_concentration": strategic_concentration,
            })
        required = {"party_id", "name", "ideology", "organization", "central_budget", "strategic_concentration"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"party_csv missing columns: {sorted(missing)}")
        return df.copy()

    def _load_or_generate_candidates(self, parties: pd.DataFrame) -> pd.DataFrame:
        if self.args.candidate_csv and Path(self.args.candidate_csv).exists():
            df = pd.read_csv(self.args.candidate_csv)
        else:
            n = self.args.n_candidates
            party_ids = self.rng.choice(parties["party_id"].to_numpy(), size=n, replace=True)
            party_lookup = parties.set_index("party_id")
            party_ideology = party_lookup.loc[party_ids, "ideology"].to_numpy()
            cand_ideology = np.clip(party_ideology + self.rng.normal(0, self.args.candidate_ideology_sigma, n), -1.0, 1.0)
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
        required = {
            "candidate_id", "name", "party_id", "ideology", "incumbency", "priority", "quality",
            "initial_capital", "cpf_donations", "cnpj_donations", "non_original_donations", "total_donations",
            "base_bairro", "gender", "party_base", "initial_visibility"
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"candidate_csv missing columns: {sorted(missing)}")

        # Allocate party central budgets to candidates.
        party_info = parties.set_index("party_id")
        df = df.copy()
        df["party_transfer"] = 0.0
        for pid, sub in df.groupby("party_id"):
            phi = float(party_info.loc[pid, "strategic_concentration"])
            budget = float(party_info.loc[pid, "central_budget"])
            w = np.power(np.clip(sub["priority"].to_numpy(), 1e-6, None), phi)
            alloc = budget * w / w.sum()
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
        n = self.args.n_voters
        party_probs = parties["organization"].to_numpy()
        party_probs = party_probs / party_probs.sum()
        preferred_party = self.rng.choice(parties["party_id"].to_numpy(), size=n, p=party_probs)
        party_lookup = parties.set_index("party_id")
        base_ideology = party_lookup.loc[preferred_party, "ideology"].to_numpy()
        voter_ideology = np.clip(base_ideology + self.rng.normal(0, self.args.voter_ideology_sigma, n), -1.0, 1.0)
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

    def _precompute_candidate_arrays(self) -> None:
        c = self.candidates
        p = self.parties.set_index("party_id")
        self.candidate_ids = c["candidate_id"].to_numpy()
        self.candidate_names = c["name"].to_numpy()
        self.candidate_party = c["party_id"].to_numpy(dtype=int)
        self.candidate_ideology = c["ideology"].to_numpy(dtype=float)
        self.candidate_incumbency = c["incumbency"].to_numpy(dtype=float)
        self.candidate_quality = c["quality"].to_numpy(dtype=float)
        self.candidate_base_bairro = c["base_bairro"].to_numpy(dtype=int)
        self.candidate_effective_resources = c["effective_resources"].to_numpy(dtype=float)
        self.candidate_visibility = c["initial_visibility"].to_numpy(dtype=float).copy()
        self.candidate_party_org = p.loc[self.candidate_party, "organization"].to_numpy(dtype=float)
        self.candidate_party_ideology = p.loc[self.candidate_party, "ideology"].to_numpy(dtype=float)
        self.n_candidates = len(c)

        # Territorial matrix: candidate has one strong bairro and weaker nearby presence.
        self.territorial_strength = np.full((self.n_candidates, self.args.n_bairros), self.args.territorial_background, dtype=float)
        for j, b0 in enumerate(self.candidate_base_bairro):
            for b in range(self.args.n_bairros):
                d = min(abs(b - b0), self.args.n_bairros - abs(b - b0))
                self.territorial_strength[j, b] = self.args.territorial_background + self.args.territorial_peak * math.exp(-d / max(self.args.territorial_decay, 1e-9))

    def _precompute_voter_arrays(self) -> None:
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
        args = self.args
        rng = self.rng
        visibility_history = []
        current_choices = rng.integers(0, self.n_candidates, size=self.n_voters)

        for t in range(args.n_steps):
            # Campaign visibility update
            local_term = self.territorial_strength[:, rng.integers(0, args.n_bairros)]
            res_term = np.log1p(self.candidate_effective_resources)
            self.candidate_visibility = (
                (1.0 - args.visibility_decay) * self.candidate_visibility
                + args.a_resource * res_term
                + args.b_quality * self.candidate_quality
                + args.c_party_org * self.candidate_party_org
                + args.d_local * local_term
            )
            self.candidate_visibility = np.clip(self.candidate_visibility, 0.0, None)
            visibility_history.append(self.candidate_visibility.copy())

            # Aggregate social support from previous round
            candidate_counts_prev = np.bincount(current_choices, minlength=self.n_candidates).astype(float)
            candidate_share_prev = candidate_counts_prev / max(candidate_counts_prev.sum(), 1.0)
            party_votes_prev = pd.Series(candidate_counts_prev).groupby(self.candidate_party).sum()
            party_strength_prev = {int(pid): float(v) / self.n_voters for pid, v in party_votes_prev.items()}

            # Update voters sequentially
            new_choices = np.empty_like(current_choices)
            for i in range(self.n_voters):
                neigh = self.neighbors[i]
                neigh_choices = current_choices[neigh]
                social_by_candidate = np.bincount(neigh_choices, minlength=self.n_candidates).astype(float)
                if social_by_candidate.sum() > 0:
                    social_by_candidate /= social_by_candidate.sum()

                ideology_term = -args.alpha_ideology * np.abs(self.voter_ideology[i] - self.candidate_ideology)
                visibility_term = args.beta_visibility * self.voter_campaign_susc[i] * self.candidate_visibility
                territorial_term = args.gamma_territorial * self.territorial_strength[:, self.voter_bairro[i]]
                social_term = args.delta_social * self.voter_social_susc[i] * social_by_candidate
                party_term = args.eta_party * self.voter_party_strength[i] * (self.candidate_party == self.voter_preferred_party[i])
                incumbency_term = args.mu_incumbency * self.candidate_incumbency

                viability_party = np.array([party_strength_prev.get(int(pid), 0.0) for pid in self.candidate_party])
                viability_term = args.kappa_viability * args.f_strategic * (
                    args.rho1 * candidate_share_prev + args.rho2 * social_by_candidate + args.rho3 * viability_party
                )

                noise = rng.normal(0.0, args.noise_sigma, self.n_candidates)
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

                # Optionally limit attention to a shortlist of visible candidates.
                if args.shortlist_size < self.n_candidates:
                    shortlist_score = self.candidate_visibility + 0.5 * self.territorial_strength[:, self.voter_bairro[i]]
                    shortlist = np.argpartition(shortlist_score, -args.shortlist_size)[-args.shortlist_size:]
                    masked = np.full(self.n_candidates, -1e12)
                    masked[shortlist] = utility[shortlist]
                    utility = masked

                new_choices[i] = softmax_sample(utility, rng, args.tau)

            current_choices = new_choices

        # Final count
        candidate_votes = np.bincount(current_choices, minlength=self.n_candidates).astype(int)
        candidate_df = self.candidates.copy()
        candidate_df["votes"] = candidate_votes
        candidate_df["final_visibility"] = self.candidate_visibility
        candidate_df["vote_share"] = candidate_df["votes"] / self.n_voters

        party_df = self._aggregate_parties(candidate_df)
        seats, eq, remainders = allocate_seats_largest_remainder(party_df.set_index("party_id")["votes"], args.n_seats)
        party_df = party_df.set_index("party_id")
        party_df["electoral_quotient"] = eq
        party_df["party_quotient_floor"] = np.floor(party_df["votes"] / eq).astype(int) if eq > 0 else 0
        party_df["remainder"] = remainders
        party_df["seats"] = seats
        party_df = party_df.reset_index()

        candidate_df = self._assign_elected(candidate_df, party_df)
        candidate_df["rank_within_party"] = candidate_df.groupby("party_id")["votes"].rank(method="first", ascending=False).astype(int)
        candidate_df = candidate_df.sort_values(["elected", "votes", "final_visibility"], ascending=[False, False, False]).reset_index(drop=True)

        voter_df = self.voters.copy()
        voter_df["choice_candidate_id"] = current_choices
        voter_df["choice_party_id"] = self.candidate_party[current_choices]

        # Timeseries output
        visibility_hist = np.asarray(visibility_history)
        vis_df = pd.DataFrame(visibility_hist, columns=[f"cand_{cid}" for cid in self.candidate_ids])
        vis_df.insert(0, "step", np.arange(len(vis_df), dtype=int))

        # Save outputs
        candidate_df.to_csv(self.output_dir / "candidate_results.csv", index=False)
        party_df.to_csv(self.output_dir / "party_results.csv", index=False)
        vis_df.to_csv(self.output_dir / "timeseries_visibility.csv", index=False)
        if args.save_voters:
            voter_df.to_csv(self.output_dir / "voter_results.csv", index=False)
        else:
            sample_n = min(args.voter_sample_size, len(voter_df))
            voter_df.sample(sample_n, random_state=args.seed).to_csv(self.output_dir / "voter_results_sample.csv", index=False)

        self._make_plots(candidate_df, party_df, visibility_hist)

        summary = self._build_summary(candidate_df, party_df, eq)
        with open(self.output_dir / "summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        with open(self.output_dir / "params.json", "w", encoding="utf-8") as fh:
            json.dump(vars(args), fh, indent=2)

        return summary

    def _aggregate_parties(self, candidate_df: pd.DataFrame) -> pd.DataFrame:
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
        party_seats = party_df.set_index("party_id")["seats"].to_dict()
        candidate_df = candidate_df.copy()
        candidate_df["elected"] = 0
        for pid, sub in candidate_df.groupby("party_id"):
            seats = int(party_seats.get(pid, 0))
            if seats <= 0:
                continue
            winners = sub.sort_values(["votes", "final_visibility"], ascending=[False, False]).head(seats).index
            candidate_df.loc[winners, "elected"] = 1
        return candidate_df

    def _build_summary(self, candidate_df: pd.DataFrame, party_df: pd.DataFrame, eq: float) -> Dict[str, object]:
        votes = candidate_df["votes"].to_numpy(dtype=float)
        n = len(votes)
        mean_votes = votes.mean() if n > 0 else 0.0
        gini = float(np.abs(votes[:, None] - votes[None, :]).sum() / (2 * n * max(votes.sum(), 1.0))) if n > 0 else 0.0
        top10_share = float(np.sort(votes)[-min(10, n):].sum() / max(votes.sum(), 1.0)) if n > 0 else 0.0
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

    def _make_plots(self, candidate_df: pd.DataFrame, party_df: pd.DataFrame, visibility_hist: np.ndarray) -> None:
        plots = self.output_dir / "plots"
        # Candidate votes distribution
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sorted_votes = np.sort(candidate_df["votes"].to_numpy())[::-1]
        ax.plot(np.arange(1, len(sorted_votes) + 1), sorted_votes, marker="o", linewidth=1.2)
        ax.set_xlabel("candidate rank")
        ax.set_ylabel("votes")
        ax.set_title("Candidate vote distribution")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots / "candidate_vote_distribution.png", dpi=180)
        plt.close(fig)

        # Party seats
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        party_sorted = party_df.sort_values("seats", ascending=False)
        ax.bar(party_sorted["name"], party_sorted["seats"])
        ax.set_xlabel("party")
        ax.set_ylabel("seats")
        ax.set_title("Seats by party")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(plots / "party_seats.png", dpi=180)
        plt.close(fig)

        # Resources vs votes
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.scatter(candidate_df["effective_resources"], candidate_df["votes"], alpha=0.7)
        ax.set_xlabel("effective resources")
        ax.set_ylabel("votes")
        ax.set_title("Resources vs votes")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots / "resources_vs_votes.png", dpi=180)
        plt.close(fig)

        # Mean visibility over time
        save_plot(
            np.arange(visibility_hist.shape[0]),
            visibility_hist.mean(axis=1),
            "step",
            "mean visibility",
            "Average candidate visibility over time",
            plots / "mean_visibility_over_time.png",
        )


# -----------------------------
# CLI
# -----------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Agent-based model for Pelotas council elections")

    # Basic run control
    p.add_argument("--output-dir", type=str, required=True, help="directory for outputs")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--cycle", type=int, default=2024, choices=[2016, 2020, 2024])

    # Sizes
    p.add_argument("--n-voters", type=int, default=20000)
    p.add_argument("--n-candidates", type=int, default=300)
    p.add_argument("--n-parties", type=int, default=12)
    p.add_argument("--n-bairros", type=int, default=12)
    p.add_argument("--n-seats", type=int, default=21)
    p.add_argument("--n-steps", type=int, default=12)

    # Network
    p.add_argument("--k-neighbors", type=int, default=10)
    p.add_argument("--rewire-prob", type=float, default=0.08)

    # Optional empirical inputs
    p.add_argument("--party-csv", type=str, default="")
    p.add_argument("--candidate-csv", type=str, default="")

    # Party generation
    p.add_argument("--party-budget-mean", type=float, default=200000.0)
    p.add_argument("--party-budget-sigma", type=float, default=0.7)
    p.add_argument("--party-concentration-mean", type=float, default=1.2)

    # Candidate generation
    p.add_argument("--candidate-ideology-sigma", type=float, default=0.18)
    p.add_argument("--incumbency-prob", type=float, default=0.08)
    p.add_argument("--initial-capital-mean", type=float, default=12000.0)
    p.add_argument("--cpf-mean", type=float, default=4000.0)
    p.add_argument("--cpf-sigma", type=float, default=1.0)
    p.add_argument("--cnpj-mean", type=float, default=2500.0)
    p.add_argument("--cnpj-sigma", type=float, default=1.0)
    p.add_argument("--non-original-mean", type=float, default=18000.0)
    p.add_argument("--non-original-sigma", type=float, default=1.0)

    # Voter generation
    p.add_argument("--voter-ideology-sigma", type=float, default=0.25)

    # Resource weights
    p.add_argument("--weight-cpf", type=float, default=1.0)
    p.add_argument("--weight-cnpj", type=float, default=1.0)
    p.add_argument("--weight-non-original", type=float, default=1.1)
    p.add_argument("--weight-party-transfer", type=float, default=1.2)
    p.add_argument("--weight-initial-capital", type=float, default=0.5)

    # Visibility dynamics
    p.add_argument("--visibility-decay", type=float, default=0.25)
    p.add_argument("--a-resource", type=float, default=0.09)
    p.add_argument("--b-quality", type=float, default=0.20)
    p.add_argument("--c-party-org", type=float, default=0.16)
    p.add_argument("--d-local", type=float, default=0.12)

    # Territorial field
    p.add_argument("--territorial-background", type=float, default=0.05)
    p.add_argument("--territorial-peak", type=float, default=0.90)
    p.add_argument("--territorial-decay", type=float, default=1.5)

    # Utility parameters
    p.add_argument("--alpha-ideology", type=float, default=1.2)
    p.add_argument("--beta-visibility", type=float, default=1.0)
    p.add_argument("--gamma-territorial", type=float, default=1.1)
    p.add_argument("--delta-social", type=float, default=0.85)
    p.add_argument("--eta-party", type=float, default=0.95)
    p.add_argument("--mu-incumbency", type=float, default=0.5)
    p.add_argument("--kappa-viability", type=float, default=0.7)

    # Strategic vote and viability
    p.add_argument("--f-strategic", type=float, default=0.35)
    p.add_argument("--rho1", type=float, default=0.45)
    p.add_argument("--rho2", type=float, default=0.35)
    p.add_argument("--rho3", type=float, default=0.20)

    # Choice noise / shortlist
    p.add_argument("--tau", type=float, default=0.7)
    p.add_argument("--noise-sigma", type=float, default=0.03)
    p.add_argument("--shortlist-size", type=int, default=35)

    # Outputs
    p.add_argument("--save-voters", action="store_true")
    p.add_argument("--voter-sample-size", type=int, default=2000)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    model = PelotasElectionABM(args)
    summary = model.run()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
