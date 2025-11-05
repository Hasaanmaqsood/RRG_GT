# /workspace/CheXpert/R2GenGPT/models/game_theory_framework.py

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
            visual_dim=self.visual_dim,  # 1024
            text_dim=self.text_dim,      # 4096  
            num_diseases=14
        )
        
        # Loss weights
        self.lambda_token = args.lambda_token
        self.lambda_section = args.lambda_section
        self.lambda_disease = args.lambda_disease
        
        # Temporal consistency module - FIXED: Add projection for text features
        self.temporal_game = TemporalConsistencyGame(
            input_dim=self.text_dim,  # 4096
            hidden_dim=self.hidden_dim  # 512
        )
        
    def forward(self, visual_features, text_features, disease_labels=None, 
                section_features=None):
        """
        Complete forward pass through all game levels
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
        
        # Level 4: Temporal consistency
        temporal_results = self.temporal_game(text_features)
        results['temporal_loss'] = temporal_results['temporal_loss']
        total_loss += 0.1 * temporal_results['temporal_loss']  # Smaller weight for temporal
        
        results['total_game_loss'] = total_loss
        return results
    
    def compute_token_loss(self, token_results):
        """Compute PROPERLY SCALED loss for token-level alignment"""
        alignment = token_results['alignment']
        batch_size, num_patches, num_tokens = alignment.shape
        
        # FIX 1: Use mean instead of sum to avoid huge values
        entropy = -(alignment * torch.log(alignment + 1e-8)).mean()
        
        # FIX 2: Proper equilibrium loss (we want to maximize equilibrium value)
        eq_loss = 1.0 - token_results['equilibrium_value']
        
        # FIX 3: Both terms are now properly scaled (0-1 range)
        combined_loss = (entropy + eq_loss) / 2.0
        
        return combined_loss


class TokenLevelTernaryGame(nn.Module):
    """
    Level 1: Token-level ternary cooperative game
    Players: Vision patches × Report tokens × Knowledge entities
    """
    def __init__(self, visual_dim: int, text_dim: int, hidden_dim: int = 512):
        super().__init__()
        
        # Enhanced projection with residual connections
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Better initialization
        self.payoff_scale = nn.Parameter(torch.ones(1) * 0.05)
        self.temperature = nn.Parameter(torch.ones(1) * 0.2)
        
    def compute_payoff_matrix(self, visual_feat, text_feat):
        """
        Compute payoff matrix between vision and text
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
        """
        B, N_v, N_t = payoff.shape
        
        # Initialize uniform strategies
        strategy_v = torch.ones(B, N_v, N_t, device=payoff.device) / N_t
        strategy_t = torch.ones(B, N_v, N_t, device=payoff.device) / N_v
        
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
        
        # Equilibrium value - normalized to be in reasonable range
        eq_value = (payoff * alignment).sum() / (B * N_v * N_t)
        
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


class TemporalConsistencyGame(nn.Module):
    """Ensure temporal consistency in reports - FIXED VERSION"""
    def __init__(self, input_dim: int = 4096, hidden_dim: int = 512):
        super().__init__()
        
        # Projection layer to handle different input dimensions
        self.text_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.temporal_encoder = nn.LSTM(
            hidden_dim, hidden_dim // 2, 
            batch_first=True, bidirectional=True
        )
        
        # FIXED: Use BCEWithLogitsLoss instead of Sigmoid + BCE
        self.consistency_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
            # Removed Sigmoid - will use BCEWithLogits
        )
        
    def detect_temporal_patterns(self, text_embeddings):
        """Detect temporal patterns in text embeddings"""
        # Project text embeddings to LSTM dimension
        projected_text = self.text_proj(text_embeddings)  # [B, seq_len, 512]
        
        # Encode temporal patterns
        temporal_features, _ = self.temporal_encoder(projected_text)
        
        # Predict temporal consistency (logits, not probabilities)
        consistency_logits = self.consistency_predictor(temporal_features.mean(dim=1))
        
        return consistency_logits
    
    def forward(self, text_embeddings):
        """Forward pass - FIXED: Use BCEWithLogitsLoss"""
        consistency_logits = self.detect_temporal_patterns(text_embeddings)
        
        # FIXED: Use BCEWithLogitsLoss instead of BCE + Sigmoid
        # Target: encourage high consistency (close to 1)
        target = torch.ones_like(consistency_logits) * 0.9
        temporal_loss = F.binary_cross_entropy_with_logits(consistency_logits, target)
        
        return {
            'temporal_consistency': torch.sigmoid(consistency_logits),  # For logging
            'temporal_loss': temporal_loss
        }

class SectionLevelStackelbergGame(nn.Module):
    """Section-level Stackelberg game - FIXED VERSION"""
    def __init__(self, hidden_dim: int = 4096, num_sections: int = 5):
        super().__init__()
        self.num_sections = num_sections
        self.hidden_dim = hidden_dim
        
        # Simplified section coherence model
        self.section_coherence = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 2,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        
    def forward(self, image_feat, section_features):
        """
        Simplified section coherence game
        """
        if not section_features:
            return {'coherence_loss': torch.tensor(0.0, device=image_feat.device)}
        
        # Combine section features
        section_tensors = []
        for name, feat in section_features.items():
            if feat is not None:
                section_tensors.append(feat.mean(dim=1, keepdim=True))
        
        if len(section_tensors) < 2:
            return {'coherence_loss': torch.tensor(0.0, device=image_feat.device)}
        
        sections_combined = torch.cat(section_tensors, dim=1)
        
        # Encode section relationships
        encoded_sections = self.section_coherence(sections_combined)
        
        # Compute coherence loss (sections should be similar but distinct)
        similarity_matrix = F.cosine_similarity(
            encoded_sections.unsqueeze(1), 
            encoded_sections.unsqueeze(2), 
            dim=-1
        )
        
        # Ideal: moderate similarity (not too similar, not too different)
        target_similarity = torch.eye(similarity_matrix.size(1), 
                                    device=similarity_matrix.device) * 0.5
        coherence_loss = F.mse_loss(similarity_matrix, target_similarity)
        
        return {
            'coherence_loss': coherence_loss,
            'section_similarities': similarity_matrix
        }


class DiseaseLevelShapleyGame(nn.Module):
    """Disease-level coalition game with Shapley values - FIXED VERSION"""
    def __init__(self, visual_dim: int = 1024, text_dim: int = 4096, num_diseases: int = 14):
        super().__init__()
        self.num_diseases = num_diseases
        
        # Projection layers to handle different input dimensions
        self.visual_proj = nn.Linear(visual_dim, 512)
        self.text_proj = nn.Linear(text_dim, 512)
        
        # Disease prototypes in common space
        self.disease_prototypes = nn.Parameter(
            torch.randn(num_diseases, 512) * 0.01
        )
        
        # Simplified co-occurrence matrix
        self.cooccurrence = nn.Parameter(torch.eye(num_diseases) * 0.8)
        
    def forward(self, image_feat, text_feat, disease_labels):
        """Fixed disease game forward pass"""
        # Project features to common dimension
        v_proj = F.normalize(self.visual_proj(image_feat), dim=-1)  # [B, 512]
        t_proj = F.normalize(self.text_proj(text_feat), dim=-1)     # [B, 512]
        d_proto = F.normalize(self.disease_prototypes, dim=-1)      # [14, 512]
        
        # Compute disease similarities
        v_sim = torch.mm(v_proj, d_proto.t())  # [B, 14]
        t_sim = torch.mm(t_proj, d_proto.t())  # [B, 14]
        
        # Disease alignment loss
        disease_loss = F.mse_loss(v_sim, t_sim) * 0.5 + \
                      F.mse_loss(t_sim, disease_labels.float()) * 0.5
        
        # Co-occurrence regularization
        cooccur_loss = F.mse_loss(
            torch.sigmoid(self.cooccurrence),
            torch.eye(self.num_diseases, device=disease_labels.device)
        )
        
        total_loss = disease_loss + 0.1 * cooccur_loss
        
        return {
            'shapley_values': (v_sim + t_sim) / 2,  # Simplified Shapley
            'cooccurrence_loss': total_loss
        }