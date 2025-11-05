# /workspace/CheXpert/R2GenGPT/models/game_theory_modules.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class MultiLevelGameTheoryFramework(nn.Module):
    """
    Complete Multi-Level Game Theory Framework
    Integrates: Token-level, Section-level, Disease-level games
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # Dimensions
        self.visual_dim = args.visual_hidden_size  # From Swin: 1024
        self.text_dim = args.llm_hidden_size  # From LLaMA: 4096
        self.hidden_dim = args.game_hidden_dim  # 512
        
        # Game modules
        self.token_game = TokenLevelTernaryGame(
            visual_dim=self.visual_dim,
            text_dim=self.text_dim,
            hidden_dim=self.hidden_dim
        )
        
        self.section_game = SectionLevelStackelbergGame(
            hidden_dim=self.text_dim,
            num_sections=5
        )
        
        self.disease_game = DiseaseLevelShapleyGame(
            hidden_dim=self.text_dim,
            num_diseases=14
        )
        
        # Loss weights
        self.lambda_token = args.lambda_token
        self.lambda_section = args.lambda_section
        self.lambda_disease = args.lambda_disease
        
    def forward(self, visual_features, text_features, disease_labels=None, 
                section_features=None):
        """
        Complete forward pass through all game levels
        
        Args:
            visual_features: [B, N_patches, D_v] from Swin Transformer
            text_features: [B, N_tokens, D_t] from LLaMA embeddings
            disease_labels: [B, 14] disease labels (optional)
            section_features: Dict of section features (optional)
        
        Returns:
            losses and alignment results
        """
        results = {}
        total_loss = 0.0
        
        # Level 1: Token-level game
        token_results = self.token_game(visual_features, text_features)
        results['token_alignment'] = token_results['alignment']
        results['token_equilibrium'] = token_results['equilibrium_value']
        token_loss = self.compute_token_loss(token_results)
        total_loss += self.lambda_token * token_loss
        results['token_loss'] = token_loss
        
        # Level 2: Section-level game (if section features provided)
        if section_features is not None:
            section_results = self.section_game(visual_features, section_features)
            results['section_coherence_loss'] = section_results['coherence_loss']
            total_loss += self.lambda_section * section_results['coherence_loss']
        
        # Level 3: Disease-level game (if labels provided)
        if disease_labels is not None:
            # Use global features for disease game
            visual_global = visual_features.mean(dim=1)  # [B, D_v]
            text_global = text_features.mean(dim=1)  # [B, D_t]
            
            disease_results = self.disease_game(visual_global, text_global, disease_labels)
            results['shapley_values'] = disease_results['shapley_values']
            results['disease_loss'] = disease_results['cooccurrence_loss']
            total_loss += self.lambda_disease * disease_results['cooccurrence_loss']
        
        results['total_game_loss'] = total_loss
        return results
    
    def compute_token_loss(self, token_results):
        """Compute loss for token-level alignment"""
        alignment = token_results['alignment']
        # Encourage sparse, confident alignments
        entropy = -(alignment * torch.log(alignment + 1e-8)).sum(dim=[1, 2]).mean()
        # Maximize equilibrium value
        eq_loss = -token_results['equilibrium_value']
        return entropy + eq_loss


class TokenLevelTernaryGame(nn.Module):
    """
    Level 1: Token-level ternary cooperative game
    Players: Vision patches × Report tokens × Knowledge entities
    """
    def __init__(self, visual_dim: int, text_dim: int, hidden_dim: int = 512):
        super().__init__()
        
        # Project to common dimension
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Payoff computation (using dot product + learned scaling)
        self.payoff_scale = nn.Parameter(torch.ones(1) * 0.1)
        
        # Nash equilibrium temperature
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)
        
    def compute_payoff_matrix(self, visual_feat, text_feat):
        """
        Compute payoff matrix between vision and text
        
        Args:
            visual_feat: [B, N_v, D_v]
            text_feat: [B, N_t, D_t]
        Returns:
            payoff: [B, N_v, N_t]
        """
        # Project to common space
        v_proj = self.visual_proj(visual_feat)  # [B, N_v, H]
        t_proj = self.text_proj(text_feat)      # [B, N_t, H]
        
        # Normalize
        v_norm = F.normalize(v_proj, dim=-1)
        t_norm = F.normalize(t_proj, dim=-1)
        
        # Compute similarity (payoff)
        payoff = torch.bmm(v_norm, t_norm.transpose(1, 2))  # [B, N_v, N_t]
        payoff = payoff * self.payoff_scale
        
        return payoff
    
    def nash_equilibrium_solver(self, payoff, iterations=5):
        """
        Solve for Nash equilibrium via iterative best response
        
        Args:
            payoff: [B, N_v, N_t]
            iterations: int
        Returns:
            alignment: [B, N_v, N_t]
            eq_value: scalar
        """
        B, N_v, N_t = payoff.shape
        
        # Initialize uniform strategies
        strategy_v = torch.ones_like(payoff) / N_t
        strategy_t = torch.ones_like(payoff) / N_v
        
        # Iterative best response
        for _ in range(iterations):
            # Visual player's best response given text strategy
            expected_payoff_v = payoff * strategy_t
            strategy_v = F.softmax(expected_payoff_v / self.temperature, dim=2)
            
            # Text player's best response given visual strategy
            expected_payoff_t = payoff.transpose(1, 2) * strategy_v.transpose(1, 2)
            strategy_t = F.softmax(expected_payoff_t / self.temperature, dim=2).transpose(1, 2)
        
        # Final alignment
        alignment = strategy_v * strategy_t
        alignment = F.normalize(alignment, p=1, dim=2)  # Normalize to sum to 1
        
        # Equilibrium value
        eq_value = (payoff * alignment).sum() / B
        
        return alignment, eq_value
    
    def forward(self, visual_feat, text_feat):
        """Forward pass"""
        payoff = self.compute_payoff_matrix(visual_feat, text_feat)
        alignment, eq_value = self.nash_equilibrium_solver(payoff)
        
        return {
            'alignment': alignment,
            'equilibrium_value': eq_value,
            'payoff': payoff
        }


class SectionLevelStackelbergGame(nn.Module):
    """
    Level 2: Section-level Stackelberg game
    Leader: Findings section (most important)
    Followers: Other sections
    """
    def __init__(self, hidden_dim: int = 4096, num_sections: int = 5):
        super().__init__()
        
        self.num_sections = num_sections
        self.hidden_dim = hidden_dim
        
        # Section embeddings
        self.section_embeddings = nn.Embedding(num_sections, hidden_dim)
        
        # Leader strategy network (for Findings)
        self.leader_strategy = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )
        
        # Follower strategy networks
        self.follower_strategy = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 2,
                dropout=0.1,
                batch_first=True,
                norm_first=True
            )
            for _ in range(num_sections - 1)
        ])
        
        # Dependency matrix (learned)
        self.register_buffer('section_order', torch.tensor([
            [0, 1, 1, 0, 0],  # Clinical History depends on Findings
            [1, 0, 1, 1, 0],  # Findings (leader)
            [1, 1, 0, 0, 0],  # Impression depends on both
            [0, 1, 0, 0, 0],  # Comparison depends on Findings
            [0, 0, 0, 0, 0],  # Technique (independent)
        ], dtype=torch.float32))
        
    def forward(self, image_feat, section_features):
        """
        Args:
            image_feat: [B, N_patches, D]
            section_features: Dict with keys like 'findings', 'impression', etc.
        """
        if 'findings' not in section_features:
            # If no section info, return zero loss
            return {'coherence_loss': torch.tensor(0.0, device=image_feat.device)}
        
        findings_feat = section_features['findings']  # [B, L_findings, D]
        
        # Leader plays: Findings section attends to image
        combined = torch.cat([findings_feat, image_feat], dim=1)
        leader_output = self.leader_strategy(combined)
        leader_repr = leader_output[:, :findings_feat.shape[1], :].mean(dim=1)  # [B, D]
        
        # Collect all section representations
        section_reprs = [leader_repr]
        
        # Followers play: Other sections respond to leader
        for idx, (name, feat) in enumerate(section_features.items()):
            if name == 'findings':
                continue
            # Attend to leader strategy
            combined = torch.cat([feat, leader_output], dim=1)
            follower_output = self.follower_strategy[idx](combined)
            follower_repr = follower_output[:, :feat.shape[1], :].mean(dim=1)
            section_reprs.append(follower_repr)
        
        # Compute coherence loss (encourage proper dependencies)
        if len(section_reprs) > 1:
            section_tensor = torch.stack(section_reprs, dim=1)  # [B, num_sections, D]
            
            # Compute pairwise similarities
            similarities = torch.bmm(
                F.normalize(section_tensor, dim=-1),
                F.normalize(section_tensor, dim=-1).transpose(1, 2)
            )  # [B, S, S]
            
            # Expected structure
            expected = self.section_order[:len(section_reprs), :len(section_reprs)]
            expected = expected.unsqueeze(0).expand(similarities.shape[0], -1, -1)
            
            # MSE loss
            coherence_loss = F.mse_loss(similarities, expected)
        else:
            coherence_loss = torch.tensor(0.0, device=image_feat.device)
        
        return {
            'coherence_loss': coherence_loss,
            'leader_output': leader_repr
        }


class DiseaseLevelShapleyGame(nn.Module):
    """
    Level 3: Disease-level coalition game with Shapley values
    Players: Images × Reports × Diseases
    """
    def __init__(self, hidden_dim: int = 4096, num_diseases: int = 14):
        super().__init__()
        
        self.num_diseases = num_diseases
        self.hidden_dim = hidden_dim
        
        # Disease prototypes (learnable)
        self.disease_prototypes = nn.Parameter(
            torch.randn(num_diseases, hidden_dim) * 0.02
        )
        
        # Projection layers
        self.visual_proj = nn.Linear(hidden_dim, hidden_dim)
        self.text_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Disease co-occurrence (learned, initialized with medical knowledge)
        # Common co-occurrences: Edema + Cardiomegaly, Atelectasis + Effusion, etc.
        cooccur_init = torch.eye(num_diseases)
        # Add some known co-occurrences
        known_pairs = [(1, 3), (0, 9), (7, 9)]  # Cardiomegaly-Edema, etc.
        for i, j in known_pairs:
            cooccur_init[i, j] = 0.7
            cooccur_init[j, i] = 0.7
        
        self.cooccurrence = nn.Parameter(cooccur_init)
        
    def compute_coalition_value(self, image_feat, text_feat, disease_mask):
        """
        Compute value of a disease coalition
        
        Args:
            image_feat: [B, D]
            text_feat: [B, D]
            disease_mask: [B, K] binary mask
        Returns:
            value: [B]
        """
        # Project features
        v_proj = F.normalize(self.visual_proj(image_feat), dim=-1)
        t_proj = F.normalize(self.text_proj(text_feat), dim=-1)
        d_proto = F.normalize(self.disease_prototypes, dim=-1)
        
        # Compute similarities to disease prototypes
        v_sim = torch.mm(v_proj, d_proto.t())  # [B, K]
        t_sim = torch.mm(t_proj, d_proto.t())  # [B, K]
        
        # Weight by disease mask
        v_contrib = (v_sim * disease_mask).sum(dim=1)
        t_contrib = (t_sim * disease_mask).sum(dim=1)
        
        # Coalition value = cross-modal agreement
        value = (v_contrib + t_contrib) / 2.0
        
        return value
    
    def compute_shapley_values(self, image_feat, text_feat, disease_labels):
        """
        Compute Shapley values for each disease
        Simplified version using sampling
        
        Args:
            image_feat: [B, D]
            text_feat: [B, D]
            disease_labels: [B, K]
        Returns:
            shapley: [B, K]
        """
        B, K = disease_labels.shape
        shapley_values = torch.zeros_like(disease_labels, dtype=torch.float32)
        
        # Number of samples per disease (trade-off between accuracy and speed)
        num_samples = 5
        
        for i in range(K):
            marginal_sum = torch.zeros(B, device=disease_labels.device)
            
            for _ in range(num_samples):
                # Sample random coalition (without disease i)
                coalition = (torch.rand(B, K, device=disease_labels.device) > 0.5).float()
                coalition[:, i] = 0
                
                # Value with disease i
                coalition_with_i = coalition.clone()
                coalition_with_i[:, i] = disease_labels[:, i]
                value_with = self.compute_coalition_value(image_feat, text_feat, coalition_with_i)
                
                # Value without disease i
                value_without = self.compute_coalition_value(image_feat, text_feat, coalition)
                
                # Marginal contribution
                marginal = value_with - value_without
                marginal_sum += marginal
            
            shapley_values[:, i] = marginal_sum / num_samples
        
        return shapley_values
    
    def forward(self, image_feat, text_feat, disease_labels):
        """Forward pass"""
        # Compute Shapley values
        shapley_vals = self.compute_shapley_values(image_feat, text_feat, disease_labels)
        
        # Compute co-occurrence loss
        # Encourage learned co-occurrence to match observed patterns
        batch_cooccur = torch.mm(disease_labels.float().t(), disease_labels.float()) / disease_labels.shape[0]
        learned_cooccur = torch.sigmoid(self.cooccurrence)
        cooccur_loss = F.mse_loss(learned_cooccur, batch_cooccur.detach())
        
        return {
            'shapley_values': shapley_vals,
            'disease_prototypes': self.disease_prototypes,
            'cooccurrence_loss': cooccur_loss
        }
