# outputs/plot_generator.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

class PlotGenerator:
    """
    Responsável por gerar todos os gráficos da simulação.
    """
    
    def __init__(self, output_dir: Path, dpi: int = 180, figsize: tuple = (7, 4.5)):
        """
        Args:
            output_dir: Diretório onde os plots serão salvos
            dpi: Resolução dos gráficos
            figsize: Tamanho padrão das figuras
        """
        self.output_dir = output_dir
        self.plots_dir = output_dir / "plots"
        self.dpi = dpi
        self.figsize = figsize
        
        # Garante que o diretório existe
        self.plots_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_all(self, candidate_df: pd.DataFrame, party_df: pd.DataFrame, visibility_hist: np.ndarray):
        """Gera todos os gráficos de uma vez."""
        self.plot_candidate_vote_distribution(candidate_df)
        self.plot_party_seats(party_df)
        self.plot_resources_vs_votes(candidate_df)
        self.plot_mean_visibility(visibility_hist)
    
    def plot_candidate_vote_distribution(self, candidate_df: pd.DataFrame):
        """Gráfico de distribuição de votos por candidato."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        sorted_votes = np.sort(candidate_df["votes"].to_numpy())[::-1]
        ax.plot(np.arange(1, len(sorted_votes) + 1), sorted_votes, marker="o", linewidth=1.2)
        
        ax.set_xlabel("candidate rank")
        ax.set_ylabel("votes")
        ax.set_title("Candidate vote distribution")
        ax.grid(alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(self.plots_dir / "candidate_vote_distribution.png", dpi=self.dpi)
        plt.close(fig)
    
    def plot_party_seats(self, party_df: pd.DataFrame):
        """Gráfico de barras com cadeiras por partido."""
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        
        party_sorted = party_df.sort_values("seats", ascending=False)
        ax.bar(party_sorted["name"], party_sorted["seats"])
        
        ax.set_xlabel("party")
        ax.set_ylabel("seats")
        ax.set_title("Seats by party")
        ax.tick_params(axis="x", rotation=45)
        
        fig.tight_layout()
        fig.savefig(self.plots_dir / "party_seats.png", dpi=self.dpi)
        plt.close(fig)
    
    def plot_resources_vs_votes(self, candidate_df: pd.DataFrame):
        """Gráfico de dispersão: recursos vs votos."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.scatter(candidate_df["effective_resources"], candidate_df["votes"], alpha=0.7)
        
        ax.set_xlabel("effective resources")
        ax.set_ylabel("votes")
        ax.set_title("Resources vs votes")
        ax.grid(alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(self.plots_dir / "resources_vs_votes.png", dpi=self.dpi)
        plt.close(fig)
    
    def plot_mean_visibility(self, visibility_hist: np.ndarray):
        """Gráfico da visibilidade média ao longo do tempo."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        steps = np.arange(visibility_hist.shape[0])
        mean_visibility = visibility_hist.mean(axis=1)
        
        ax.plot(steps, mean_visibility, marker="o", linewidth=1.5)
        
        ax.set_xlabel("step")
        ax.set_ylabel("mean visibility")
        ax.set_title("Average candidate visibility over time")
        ax.grid(alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(self.plots_dir / "mean_visibility_over_time.png", dpi=self.dpi)
        plt.close(fig)
    
    @staticmethod
    def save_simple_plot(x, y, xlabel, ylabel, title, path: Path, dpi: int = 180):
        """Método estático para plots simples (se precisar)."""
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(x, y, marker="o", linewidth=1.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=dpi)
        plt.close(fig)