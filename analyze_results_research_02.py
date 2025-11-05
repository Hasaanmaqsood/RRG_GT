#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Research-Grade Results Analysis for Game Theory R2GenGPT
Complete analysis with publication-quality figures and detailed comparative metrics
Author: Research Analysis Tool
Usage: python analyze_results_research.py --checkpoint_dir /beegfs/home/hasaan.maqsood/CheXpert/checkpoints_game_theory
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import re
from collections import defaultdict
from difflib import SequenceMatcher
import textwrap

# Set publication-quality style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


class ResearchResultsAnalyzer:
    """Research-grade analyzer with publication-quality outputs"""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoints_path = self.checkpoint_dir / "checkpoints"
        self.logs_path = self.checkpoint_dir / "logs"
        self.results_path = self.checkpoint_dir / "result"
        
        # Storage
        self.checkpoint_metrics = []
        self.csv_metrics = None
        self.tensorboard_data = {}
        self.result_files = {}
        
    def extract_checkpoint_info(self) -> pd.DataFrame:
        """Extract metrics from checkpoint filenames and link with CSV metrics"""
        print("Extracting checkpoint information...")
        
        checkpoint_files = list(self.checkpoints_path.glob("*.pth"))
        data = []
        
        for ckpt_file in checkpoint_files:
            # Extract basic info from filename
            match = re.search(r'checkpoint_epoch(\d+)_step(\d+)_bleu([\d.]+)_cider([\d.]+)\.pth', ckpt_file.name)
            
            if match:
                epoch, step, bleu, cider = match.groups()
                file_size_mb = ckpt_file.stat().st_size / (1024 * 1024)
                data.append({
                    'filename': ckpt_file.name,
                    'epoch': int(epoch),
                    'step': int(step),
                    'bleu4': float(bleu),
                    'cider': float(cider),
                    'combined_score': float(bleu) + float(cider),
                    'size_mb': file_size_mb,
                    'filepath': str(ckpt_file),
                    'timestamp': ckpt_file.stat().st_mtime
                })
        
        df = pd.DataFrame(data).sort_values('step')
        print(f"Found {len(df)} checkpoints spanning {df['epoch'].max()+1} epochs")
        
        # Print available metrics
        available_metrics = [col for col in df.columns if any(m in col for m in ['bleu', 'rouge', 'cider'])]
        print(f"Available metrics in checkpoints: {available_metrics}")
        
        return df
    
    def load_csv_metrics(self) -> pd.DataFrame:
        """Load training metrics from CSV logs - ENHANCED VERSION"""
        print("Loading training metrics...")
        
        csv_files = list(self.logs_path.glob("csvlog/*/metrics.csv"))
        
        if not csv_files:
            print("No CSV metrics found")
            return None
        
        latest_csv = sorted(csv_files, key=lambda x: int(x.parent.name.split('_')[1]))[-1]
        print(f"Loading: {latest_csv}")
        
        df = pd.read_csv(latest_csv)
        
        # Ensure epoch column exists
        if 'epoch' not in df.columns and 'step' in df.columns:
            # Estimate epoch from step if needed
            df['epoch'] = df.index
        
        # Print all available metrics
        all_columns = df.columns.tolist()
        print(f"All available columns: {all_columns}")
        
        # Look for BLEU, ROUGE, CIDEr metrics with different naming conventions
        bleu_columns = [col for col in df.columns if 'bleu' in col.lower()]
        rouge_columns = [col for col in df.columns if 'rouge' in col.lower()]
        cider_columns = [col for col in df.columns if 'cider' in col.lower()]
        meteor_columns = [col for col in df.columns if 'meteor' in col.lower()]
        
        print(f"BLEU columns: {bleu_columns}")
        print(f"ROUGE columns: {rouge_columns}")
        print(f"CIDEr columns: {cider_columns}")
        print(f"METEOR columns: {meteor_columns}")
        
        print(f"Loaded {len(df)} training steps")
        
        return df
    
    def load_result_jsons(self) -> Dict:
        """Load generated predictions and references"""
        print("Loading predictions and references...")
        
        result_files = list(self.results_path.glob("result_*.json"))
        refs_file = self.results_path / "refs.json"
        
        results = {}
        
        for res_file in result_files:
            match = re.search(r'result_(\d+)_(\d+)\.json', res_file.name)
            if match:
                epoch, step = match.groups()
                try:
                    with open(res_file, 'r') as f:
                        data = json.load(f)
                        results[f"epoch{epoch}_step{step}"] = {
                            'predictions': data,
                            'epoch': int(epoch),
                            'step': int(step)
                        }
                except Exception as e:
                    print(f"Error loading {res_file.name}: {e}")
        
        if refs_file.exists():
            with open(refs_file, 'r') as f:
                results['references'] = json.load(f)
        
        print(f"Loaded {len(results)-1} prediction files and references")
        return results
    
    def calculate_similarity_score(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def highlight_common_words(self, gt: str, pred: str) -> Tuple[List[str], List[str]]:
        """Identify common words between ground truth and prediction"""
        gt_words = set(gt.lower().split())
        pred_words = set(pred.lower().split())
        common = gt_words & pred_words
        
        gt_highlighted = []
        pred_highlighted = []
        
        for word in gt.split():
            if word.lower() in common:
                gt_highlighted.append(('YES', word))
            else:
                gt_highlighted.append(('', word))
        
        for word in pred.split():
            if word.lower() in common:
                pred_highlighted.append(('YES', word))
            else:
                pred_highlighted.append(('NO', word))
        
        return gt_highlighted, pred_highlighted
    
    def plot_publication_training_curves(self, df: pd.DataFrame, save_path: str):
        """Publication-quality training curves - UPDATED VERSION"""
        print("Creating publication-quality training curves...")
        
        fig = plt.figure(figsize=(20, 16))  # Larger figure for more metrics
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # Main title
        fig.suptitle('Game Theory R2GenGPT: Complete Metrics Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Loss Evolution
        ax1 = fig.add_subplot(gs[0, :2])
        if 'loss' in df.columns:
            epochs = df['epoch'] if 'epoch' in df.columns else range(len(df))
            ax1.plot(epochs, df['loss'], linewidth=2.5, color='#e74c3c', 
                    label='Total Loss', marker='o', markersize=4, alpha=0.8)
            
            ax1.set_xlabel('Epoch', fontweight='bold')
            ax1.set_ylabel('Loss Value', fontweight='bold')
            ax1.set_title('(a) Training Loss Evolution', loc='left', fontweight='bold')
            ax1.legend(loc='best', framealpha=0.9)
            ax1.grid(True, alpha=0.3, linestyle='--')
        
        # 2. BLEU Scores (All BLEU variants)
        ax2 = fig.add_subplot(gs[0, 2])
        bleu_cols = [col for col in df.columns if 'bleu' in col.lower()]
        if bleu_cols:
            for col in bleu_cols:
                if col in df.columns and df[col].notna().any():
                    epochs = df['epoch'] if 'epoch' in df.columns else range(len(df))
                    ax2.plot(epochs, df[col], marker='o', label=col.replace('_', '-').upper(),
                            linewidth=2, markersize=3)
            ax2.set_xlabel('Epoch', fontweight='bold')
            ax2.set_ylabel('BLEU Score', fontweight='bold')
            ax2.set_title('(b) All BLEU Scores', loc='left', fontweight='bold')
            ax2.legend(loc='best', fontsize=8)
            ax2.grid(True, alpha=0.3, linestyle='--')
        
        # 3. CIDEr Score
        ax3 = fig.add_subplot(gs[1, 0])
        cider_cols = [col for col in df.columns if 'cider' in col.lower()]
        if cider_cols:
            col = cider_cols[0]  # Take first CIDEr column
            if df[col].notna().any():
                epochs = df['epoch'] if 'epoch' in df.columns else range(len(df))
                ax3.plot(epochs, df[col], marker='s', color='#e67e22', 
                        linewidth=2.5, markersize=4, label='CIDEr')
                
                ax3.set_xlabel('Epoch', fontweight='bold')
                ax3.set_ylabel('CIDEr Score', fontweight='bold')
                ax3.set_title('(c) CIDEr Score', loc='left', fontweight='bold')
                ax3.legend(loc='best')
                ax3.grid(True, alpha=0.3, linestyle='--')
        
        # 4. ROUGE-L Score
        ax4 = fig.add_subplot(gs[1, 1])
        rouge_cols = [col for col in df.columns if 'rouge' in col.lower() and 'l' in col.lower()]
        if not rouge_cols:
            rouge_cols = [col for col in df.columns if 'rouge' in col.lower()]
        
        if rouge_cols:
            col = rouge_cols[0]  # Take first ROUGE column
            if df[col].notna().any():
                epochs = df['epoch'] if 'epoch' in df.columns else range(len(df))
                ax4.plot(epochs, df[col], marker='D', color='#9b59b6',
                        linewidth=2, markersize=3, label='ROUGE-L')
                
                ax4.set_xlabel('Epoch', fontweight='bold')
                ax4.set_ylabel('ROUGE-L Score', fontweight='bold')
                ax4.set_title('(d) ROUGE-L Score', loc='left', fontweight='bold')
                ax4.legend(loc='best')
                ax4.grid(True, alpha=0.3, linestyle='--')
        
        # 5. METEOR Score
        ax5 = fig.add_subplot(gs[1, 2])
        meteor_cols = [col for col in df.columns if 'meteor' in col.lower()]
        if meteor_cols:
            col = meteor_cols[0]
            if df[col].notna().any():
                epochs = df['epoch'] if 'epoch' in df.columns else range(len(df))
                ax5.plot(epochs, df[col], marker='v', color='#1abc9c',
                        linewidth=2, markersize=3, label='METEOR')
                
                ax5.set_xlabel('Epoch', fontweight='bold')
                ax5.set_ylabel('METEOR Score', fontweight='bold')
                ax5.set_title('(e) METEOR Score', loc='left', fontweight='bold')
                ax5.legend(loc='best')
                ax5.grid(True, alpha=0.3, linestyle='--')
        

        
        # 6. Game Theory Metrics
        ax6 = fig.add_subplot(gs[2, 0])
        game_cols = [col for col in df.columns if 'game' in col.lower() and 'loss' in col.lower()]
        if game_cols:
            epochs = df['epoch'] if 'epoch' in df.columns else range(len(df))
            for col in game_cols[:3]:
                if col in df.columns:
                    ax6.plot(epochs, df[col], marker='o', label=col.replace('_', ' ').title(),
                            linewidth=1.5, markersize=2)
            ax6.set_xlabel('Epoch', fontweight='bold')
            ax6.set_ylabel('Loss Value', fontweight='bold')
            ax6.set_title('(f) Game Theory Loss Components', loc='left', fontweight='bold')
            ax6.legend(loc='best', fontsize=8)
            ax6.grid(True, alpha=0.3, linestyle='--')
        
        # 7. Training Stability
        ax7 = fig.add_subplot(gs[2, 1])
        if 'loss' in df.columns and len(df) > 3:
            rolling_std = df['loss'].rolling(window=3, center=True).std()
            epochs = df['epoch'] if 'epoch' in df.columns else range(len(df))
            ax7.plot(epochs, rolling_std, color='#34495e', linewidth=2, marker='o', markersize=3)
            ax7.fill_between(epochs, rolling_std, alpha=0.3, color='#34495e')
            ax7.set_xlabel('Epoch', fontweight='bold')
            ax7.set_ylabel('Loss Std Dev', fontweight='bold')
            ax7.set_title('(g) Training Stability', loc='left', fontweight='bold')
            ax7.grid(True, alpha=0.3, linestyle='--')
        
        # 8. Combined Performance Score
        ax8 = fig.add_subplot(gs[2, 2])
        if 'Bleu_4' in df.columns and 'CIDEr' in df.columns:
            combined = 0.5 * df['Bleu_4'].fillna(0) + 0.5 * df['CIDEr'].fillna(0)
            epochs = df['epoch'] if 'epoch' in df.columns else range(len(df))
            ax8.plot(epochs, combined, marker='o', color='#27ae60', 
                    linewidth=2.5, markersize=4, label='Combined Score')
            
            # Highlight best
            best_idx = combined.idxmax()
            ax8.scatter(epochs.iloc[best_idx] if hasattr(epochs, 'iloc') else best_idx,
                       combined.iloc[best_idx],
                       color='red', s=200, marker='*', zorder=5,
                       label=f'Best: {combined.iloc[best_idx]:.3f}')
            
            ax8.set_xlabel('Epoch', fontweight='bold')
            ax8.set_ylabel('Score', fontweight='bold')
            ax8.set_title('(h) Combined Score\n(0.5xBLEU-4 + 0.5xCIDEr)', 
                         loc='left', fontweight='bold', fontsize=10)
            ax8.legend(loc='best')
            ax8.grid(True, alpha=0.3, linestyle='--')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved to: {save_path}")
        plt.close()
    
    
    
    def plot_checkpoint_performance(self, checkpoint_df: pd.DataFrame, save_path: str):
        """Checkpoint performance comparison"""
        print("Creating checkpoint performance analysis...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Checkpoint Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. BLEU-4 Progression
        ax = axes[0, 0]
        colors = ['#3498db' if i != checkpoint_df['bleu4'].idxmax() else '#e74c3c' 
                 for i in checkpoint_df.index]
        bars = ax.bar(range(len(checkpoint_df)), checkpoint_df['bleu4'], 
                     color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Checkpoint Index', fontweight='bold')
        ax.set_ylabel('BLEU-4 Score', fontweight='bold')
        ax.set_title('(a) BLEU-4 Across Checkpoints', loc='left', fontweight='bold')
        ax.axhline(y=checkpoint_df['bleu4'].mean(), color='green', linestyle='--', 
                  linewidth=2, label=f'Mean: {checkpoint_df["bleu4"].mean():.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 2. CIDEr Progression
        ax = axes[0, 1]
        colors = ['#9b59b6' if i != checkpoint_df['cider'].idxmax() else '#e74c3c' 
                 for i in checkpoint_df.index]
        bars = ax.bar(range(len(checkpoint_df)), checkpoint_df['cider'], 
                     color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Checkpoint Index', fontweight='bold')
        ax.set_ylabel('CIDEr Score', fontweight='bold')
        ax.set_title('(b) CIDEr Across Checkpoints', loc='left', fontweight='bold')
        ax.axhline(y=checkpoint_df['cider'].mean(), color='green', linestyle='--',
                  linewidth=2, label=f'Mean: {checkpoint_df["cider"].mean():.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Combined Score
        ax = axes[0, 2]
        combined = checkpoint_df['bleu4'] + checkpoint_df['cider']
        colors = ['#16a085' if i != combined.idxmax() else '#e74c3c' 
                 for i in checkpoint_df.index]
        bars = ax.bar(range(len(checkpoint_df)), combined, 
                     color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Checkpoint Index', fontweight='bold')
        ax.set_ylabel('Combined Score', fontweight='bold')
        ax.set_title('(c) Combined Score (BLEU-4 + CIDEr)', loc='left', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Score Evolution by Epoch
        ax = axes[1, 0]
        grouped = checkpoint_df.groupby('epoch').agg({
            'bleu4': 'max',
            'cider': 'max'
        })
        x = np.arange(len(grouped))
        width = 0.35
        ax.bar(x - width/2, grouped['bleu4'], width, label='BLEU-4', 
              color='#3498db', edgecolor='black', linewidth=0.5)
        ax.bar(x + width/2, grouped['cider'], width, label='CIDEr', 
              color='#9b59b6', edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Best Score', fontweight='bold')
        ax.set_title('(d) Best Scores per Epoch', loc='left', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(grouped.index)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 5. BLEU vs CIDEr Scatter
        ax = axes[1, 1]
        scatter = ax.scatter(checkpoint_df['bleu4'], checkpoint_df['cider'],
                           c=checkpoint_df['epoch'], cmap='viridis', 
                           s=100, alpha=0.6, edgecolors='black', linewidth=1)
        
        # Highlight best
        best_idx = checkpoint_df['combined_score'].idxmax()
        ax.scatter(checkpoint_df.loc[best_idx, 'bleu4'],
                  checkpoint_df.loc[best_idx, 'cider'],
                  c='red', s=300, marker='*', edgecolors='black', 
                  linewidths=2, label='Best Overall', zorder=5)
        
        ax.set_xlabel('BLEU-4 Score', fontweight='bold')
        ax.set_ylabel('CIDEr Score', fontweight='bold')
        ax.set_title('(e) BLEU-4 vs CIDEr (colored by epoch)', loc='left', fontweight='bold')
        plt.colorbar(scatter, ax=ax, label='Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Performance Improvement
        ax = axes[1, 2]
        if len(checkpoint_df) > 1:
            bleu_improvement = checkpoint_df['bleu4'].diff().fillna(0)
            cider_improvement = checkpoint_df['cider'].diff().fillna(0)
            
            x = np.arange(len(checkpoint_df))
            ax.plot(x, bleu_improvement, marker='o', label='BLEU-4 Delta', 
                   linewidth=2, color='#3498db')
            ax.plot(x, cider_improvement, marker='s', label='CIDEr Delta', 
                   linewidth=2, color='#9b59b6')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.fill_between(x, 0, bleu_improvement, alpha=0.3, color='#3498db')
            ax.fill_between(x, 0, cider_improvement, alpha=0.3, color='#9b59b6')
            
            ax.set_xlabel('Checkpoint Index', fontweight='bold')
            ax.set_ylabel('Score Change', fontweight='bold')
            ax.set_title('(f) Performance Delta Between Checkpoints', 
                        loc='left', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved to: {save_path}")
        plt.close()
    
    def generate_comparison_report(self, results: Dict, save_path: str, num_samples: int = 20):
        """Generate detailed comparison report with highlighting"""
        print(f"Generating detailed comparison report ({num_samples} samples)...")
        
        if 'references' not in results:
            print("No references found")
            return
        
        references = results['references']
        
        # Get latest predictions
        pred_keys = [k for k in results.keys() if k.startswith('epoch')]
        if not pred_keys:
            print("No predictions found")
            return
        
        # Sort by step to get latest
        pred_keys_sorted = sorted(pred_keys, 
                                 key=lambda x: results[x]['step'])
        latest_pred_key = pred_keys_sorted[-1]
        predictions = results[latest_pred_key]['predictions']
        
        # Sample IDs
        all_ids = list(references.keys())
        np.random.seed(42)
        sample_ids = np.random.choice(all_ids, min(num_samples, len(all_ids)), replace=False)
        
        # Calculate statistics
        similarities = []
        word_overlaps = []
        length_ratios = []
        
        for sample_id in sample_ids:
            if sample_id in predictions:
                gt = references[sample_id][0]
                pred = predictions[sample_id][0]
                
                similarity = self.calculate_similarity_score(gt, pred)
                similarities.append(similarity)
                
                gt_words = set(gt.lower().split())
                pred_words = set(pred.lower().split())
                overlap = len(gt_words & pred_words) / len(gt_words) if gt_words else 0
                word_overlaps.append(overlap)
                
                length_ratio = len(pred.split()) / len(gt.split()) if gt.split() else 0
                length_ratios.append(length_ratio)
        
        # Write report
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("RESEARCH REPORT: GROUND TRUTH vs PREDICTION COMPARISON\n")
            f.write(f"Model: Game Theory R2GenGPT\n")
            f.write(f"Checkpoint: {latest_pred_key}\n")
            f.write(f"Epoch: {results[latest_pred_key]['epoch']}, Step: {results[latest_pred_key]['step']}\n")
            f.write("=" * 100 + "\n\n")
            
            # Summary statistics
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 100 + "\n")
            f.write(f"Total Samples Analyzed: {len(sample_ids)}\n")
            f.write(f"Average Similarity Score: {np.mean(similarities):.3f} +/- {np.std(similarities):.3f}\n")
            f.write(f"Average Word Overlap: {np.mean(word_overlaps):.1%} +/- {np.std(word_overlaps):.1%}\n")
            f.write(f"Average Length Ratio (pred/gt): {np.mean(length_ratios):.2f} +/- {np.std(length_ratios):.2f}\n")
            f.write(f"Minimum Similarity: {np.min(similarities):.3f}\n")
            f.write(f"Maximum Similarity: {np.max(similarities):.3f}\n")
            f.write("\n")
            
            # Quality categories
            excellent = sum(1 for s in similarities if s >= 0.8)
            good = sum(1 for s in similarities if 0.6 <= s < 0.8)
            fair = sum(1 for s in similarities if 0.4 <= s < 0.6)
            poor = sum(1 for s in similarities if s < 0.4)
            
            f.write("QUALITY DISTRIBUTION\n")
            f.write("-" * 100 + "\n")
            f.write(f"Excellent (>=0.8): {excellent} ({excellent/len(similarities)*100:.1f}%)\n")
            f.write(f"Good (0.6-0.8): {good} ({good/len(similarities)*100:.1f}%)\n")
            f.write(f"Fair (0.4-0.6): {fair} ({fair/len(similarities)*100:.1f}%)\n")
            f.write(f"Poor (<0.4): {poor} ({poor/len(similarities)*100:.1f}%)\n")
            f.write("\n\n")
            
            # Detailed examples
            f.write("=" * 100 + "\n")
            f.write("DETAILED SAMPLE COMPARISONS\n")
            f.write("Legend: [word] = Common word, word = Unique to prediction\n")
            f.write("=" * 100 + "\n\n")
            
            for idx, sample_id in enumerate(sample_ids, 1):
                if sample_id not in predictions:
                    continue
                
                gt = references[sample_id][0]
                pred = predictions[sample_id][0]
                
                similarity = self.calculate_similarity_score(gt, pred)
                gt_highlighted, pred_highlighted = self.highlight_common_words(gt, pred)
                
                # Quality badge
                if similarity >= 0.8:
                    quality = "EXCELLENT ***"
                elif similarity >= 0.6:
                    quality = "GOOD **"
                elif similarity >= 0.4:
                    quality = "FAIR *"
                else:
                    quality = "NEEDS IMPROVEMENT"
                
                f.write(f"{'='*100}\n")
                f.write(f"SAMPLE {idx}/{len(sample_ids)}: {sample_id}\n")
                f.write(f"Quality: {quality} | Similarity: {similarity:.3f}\n")
                f.write(f"{'-'*100}\n\n")
                
                # Ground truth
                f.write("GROUND TRUTH:\n")
                f.write(textwrap.fill(gt, width=95, initial_indent="  ", subsequent_indent="  "))
                f.write("\n\n")
                
                # Highlighted ground truth
                f.write("GROUND TRUTH (Highlighted - Common Words Marked):\n")
                gt_display = "  "
                for marker, word in gt_highlighted:
                    if marker == 'YES':
                        gt_display += f"[{word}] "
                    else:
                        gt_display += f"{word} "
                f.write(textwrap.fill(gt_display, width=95, subsequent_indent="  "))
                f.write("\n\n")
                
                # Prediction
                f.write("MODEL PREDICTION:\n")
                f.write(textwrap.fill(pred, width=95, initial_indent="  ", subsequent_indent="  "))
                f.write("\n\n")
                
                # Highlighted prediction
                f.write("MODEL PREDICTION (Highlighted - Common/Unique):\n")
                pred_display = "  "
                for marker, word in pred_highlighted:
                    if marker == 'YES':
                        pred_display += f"[{word}] "
                    elif marker == 'NO':
                        pred_display += f"{word} "
                f.write(textwrap.fill(pred_display, width=95, subsequent_indent="  "))
                f.write("\n\n")
                
                # Detailed metrics
                gt_words = set(gt.lower().split())
                pred_words = set(pred.lower().split())
                common_words = gt_words & pred_words
                
                f.write("DETAILED METRICS:\n")
                f.write(f"  Ground Truth Length: {len(gt.split())} words\n")
                f.write(f"  Prediction Length: {len(pred.split())} words\n")
                f.write(f"  Length Ratio: {len(pred.split())/len(gt.split()):.2f}\n")
                f.write(f"  Common Words: {len(common_words)}\n")
                f.write(f"  Word Overlap: {len(common_words)/len(gt_words)*100:.1f}%\n")
                f.write(f"  Unique to GT: {len(gt_words - pred_words)}\n")
                f.write(f"  Unique to Prediction: {len(pred_words - gt_words)}\n")
                f.write(f"  Similarity Score: {similarity:.4f}\n")
                
                # Key clinical terms (if present)
                clinical_terms = ['normal', 'abnormal', 'fracture', 'opacity', 'effusion', 
                                'pneumonia', 'cardiomegaly', 'edema', 'consolidation', 
                                'pleural', 'lung', 'heart', 'chest']
                gt_clinical = [term for term in clinical_terms if term in gt.lower()]
                pred_clinical = [term for term in clinical_terms if term in pred.lower()]
                matched_clinical = set(gt_clinical) & set(pred_clinical)
                
                if gt_clinical:
                    f.write(f"\n  Clinical Terms in GT: {', '.join(gt_clinical)}\n")
                    f.write(f"  Clinical Terms in Pred: {', '.join(pred_clinical)}\n")
                    f.write(f"  Matched Clinical Terms: {', '.join(matched_clinical) if matched_clinical else 'None'}\n")
                    f.write(f"  Clinical Term Accuracy: {len(matched_clinical)/len(gt_clinical)*100:.1f}%\n")
                
                f.write("\n\n")
        
        print(f"Saved detailed comparison report to: {save_path}")
    
    def plot_performance_heatmap(self, checkpoint_df: pd.DataFrame, save_path: str):
        """Create performance heatmap across epochs and metrics"""
        print("Creating performance heatmap...")
        
        # Prepare data for heatmap
        pivot_data = checkpoint_df.pivot_table(
            values=['bleu4', 'cider', 'combined_score'],
            index='epoch',
            aggfunc='max'
        )
        
        if pivot_data.empty:
            print("Not enough data for heatmap")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Heatmap 1: Metrics by Epoch
        ax = axes[0]
        sns.heatmap(pivot_data.T, annot=True, fmt='.3f', cmap='YlOrRd', 
                   ax=ax, cbar_kws={'label': 'Score'}, linewidths=0.5)
        ax.set_title('(a) Best Scores per Epoch', fontweight='bold', fontsize=12)
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Metric', fontweight='bold')
        
        # Heatmap 2: Normalized performance
        ax = axes[1]
        normalized = (pivot_data - pivot_data.min()) / (pivot_data.max() - pivot_data.min())
        sns.heatmap(normalized.T, annot=True, fmt='.3f', cmap='RdYlGn', 
                   ax=ax, vmin=0, vmax=1, cbar_kws={'label': 'Normalized Score'}, 
                   linewidths=0.5)
        ax.set_title('(b) Normalized Performance (0-1 scale)', fontweight='bold', fontsize=12)
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Metric', fontweight='bold')
        
        plt.suptitle('Performance Heatmap Analysis', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved to: {save_path}")
        plt.close()
    
    def generate_statistical_summary_table(self, checkpoint_df: pd.DataFrame, 
                                          csv_df: pd.DataFrame, save_path: str):
        """Generate comprehensive statistical summary table"""
        print("Generating statistical summary table...")
        
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare data
        summary_data = []
        
        # Header
        summary_data.append(['METRIC', 'INITIAL', 'FINAL', 'BEST', 'MEAN +/- STD', 'IMPROVEMENT'])
        
        # Checkpoint metrics
        summary_data.append(['CHECKPOINT METRICS', '', '', '', '', ''])
        
        for metric in ['bleu4', 'cider', 'combined_score']:
            if metric in checkpoint_df.columns:
                initial = checkpoint_df[metric].iloc[0]
                final = checkpoint_df[metric].iloc[-1]
                best = checkpoint_df[metric].max()
                mean = checkpoint_df[metric].mean()
                std = checkpoint_df[metric].std()
                improvement = ((final - initial) / initial * 100) if initial > 0 else 0
                
                summary_data.append([
                    metric.upper().replace('_', ' '),
                    f'{initial:.4f}',
                    f'{final:.4f}',
                    f'{best:.4f}',
                    f'{mean:.4f} +/- {std:.4f}',
                    f'{improvement:+.1f}%'
                ])
        
        # Training metrics
        if csv_df is not None:
            summary_data.append(['', '', '', '', '', ''])
            summary_data.append(['TRAINING METRICS', '', '', '', '', ''])
            
            # Loss metrics
            if 'loss' in csv_df.columns:
                loss_data = csv_df['loss'].dropna()
                if len(loss_data) > 0:
                    initial = loss_data.iloc[0]
                    final = loss_data.iloc[-1]
                    best = loss_data.min()
                    mean = loss_data.mean()
                    std = loss_data.std()
                    improvement = ((initial - final) / initial * 100) if initial > 0 else 0
                    
                    summary_data.append([
                        'TOTAL LOSS',
                        f'{initial:.4f}',
                        f'{final:.4f}',
                        f'{best:.4f}',
                        f'{mean:.4f} +/- {std:.4f}',
                        f'{improvement:+.1f}%'
                    ])
            
            # Additional metrics
            for metric in ['Bleu_4', 'CIDEr', 'METEOR', 'ROUGE_L']:
                if metric in csv_df.columns:
                    metric_data = csv_df[metric].dropna()
                    if len(metric_data) > 0:
                        initial = metric_data.iloc[0]
                        final = metric_data.iloc[-1]
                        best = metric_data.max()
                        mean = metric_data.mean()
                        std = metric_data.std()
                        improvement = ((final - initial) / initial * 100) if initial > 0 else 0
                        
                        summary_data.append([
                            metric.replace('_', '-'),
                            f'{initial:.4f}',
                            f'{final:.4f}',
                            f'{best:.4f}',
                            f'{mean:.4f} +/- {std:.4f}',
                            f'{improvement:+.1f}%'
                        ])
        
        # Create table
        table = ax.table(
            cellText=summary_data,
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        
        # Style table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Color coding
        for i, row in enumerate(summary_data):
            for j in range(len(row)):
                cell = table[(i, j)]
                
                if i == 0:  # Header
                    cell.set_facecolor('#2c3e50')
                    cell.set_text_props(weight='bold', color='white')
                elif row[0] in ['CHECKPOINT METRICS', 'TRAINING METRICS']:  # Section headers
                    cell.set_facecolor('#34495e')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    if i % 2 == 0:
                        cell.set_facecolor('#ecf0f1')
                    
                    # Color improvement column
                    if j == 5 and i > 0 and row[0] not in ['CHECKPOINT METRICS', 'TRAINING METRICS']:
                        try:
                            improvement = float(row[j].replace('%', '').replace('+', ''))
                            if improvement > 0:
                                cell.set_facecolor('#d4edda')
                            elif improvement < 0:
                                cell.set_facecolor('#f8d7da')
                        except:
                            pass
        
        plt.title('Statistical Summary - Game Theory R2GenGPT', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved to: {save_path}")
        plt.close()
    
    def plot_error_analysis(self, results: Dict, save_path: str, num_samples: int = 50):
        """Analyze prediction errors and patterns"""
        print("Creating error analysis visualization...")
        
        if 'references' not in results:
            return
        
        references = results['references']
        pred_keys = [k for k in results.keys() if k.startswith('epoch')]
        if not pred_keys:
            return
        
        latest_pred_key = sorted(pred_keys, key=lambda x: results[x]['step'])[-1]
        predictions = results[latest_pred_key]['predictions']
        
        # Sample for analysis
        all_ids = list(references.keys())
        sample_ids = all_ids[:min(num_samples, len(all_ids))]
        
        # Calculate metrics
        length_diffs = []
        word_overlaps = []
        similarities = []
        
        for sample_id in sample_ids:
            if sample_id in predictions:
                gt = references[sample_id][0]
                pred = predictions[sample_id][0]
                
                length_diff = len(pred.split()) - len(gt.split())
                length_diffs.append(length_diff)
                
                gt_words = set(gt.lower().split())
                pred_words = set(pred.lower().split())
                overlap = len(gt_words & pred_words) / len(gt_words) if gt_words else 0
                word_overlaps.append(overlap)
                
                similarity = self.calculate_similarity_score(gt, pred)
                similarities.append(similarity)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Error Analysis and Prediction Quality', fontsize=16, fontweight='bold')
        
        # 1. Length difference distribution
        ax = axes[0, 0]
        ax.hist(length_diffs, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Perfect Length')
        ax.axvline(x=np.mean(length_diffs), color='green', linestyle='--', 
                  linewidth=2, label=f'Mean: {np.mean(length_diffs):.1f}')
        ax.set_xlabel('Length Difference (words)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('(a) Prediction Length vs Ground Truth', loc='left', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 2. Word overlap distribution
        ax = axes[0, 1]
        ax.hist(word_overlaps, bins=20, color='#9b59b6', alpha=0.7, edgecolor='black')
        ax.axvline(x=np.mean(word_overlaps), color='red', linestyle='--', 
                  linewidth=2, label=f'Mean: {np.mean(word_overlaps):.2f}')
        ax.set_xlabel('Word Overlap Ratio', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('(b) Word Overlap Distribution', loc='left', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Similarity distribution - FIXED: Use single color instead of list
        ax = axes[0, 2]
        # Create histogram with single color
        n, bins, patches = ax.hist(similarities, bins=20, alpha=0.7, edgecolor='black')
        
        # Color bars based on similarity threshold
        for i in range(len(patches)):
            if bins[i] >= 0.6:
                patches[i].set_facecolor('#d4edda')  # Green for good
            else:
                patches[i].set_facecolor('#f8d7da')  # Red for poor
        
        ax.axvline(x=np.mean(similarities), color='red', linestyle='--', 
                  linewidth=2, label=f'Mean: {np.mean(similarities):.3f}')
        ax.axvline(x=0.6, color='orange', linestyle=':', linewidth=2, label='Good Threshold')
        ax.set_xlabel('Similarity Score', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('(c) Similarity Score Distribution', loc='left', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Overlap vs Similarity
        ax = axes[1, 0]
        scatter = ax.scatter(word_overlaps, similarities, c=length_diffs, 
                           cmap='RdYlGn_r', s=60, alpha=0.6, edgecolors='black')
        ax.set_xlabel('Word Overlap', fontweight='bold')
        ax.set_ylabel('Similarity Score', fontweight='bold')
        ax.set_title('(d) Overlap vs Similarity (colored by length diff)', 
                    loc='left', fontweight='bold')
        plt.colorbar(scatter, ax=ax, label='Length Difference')
        ax.grid(True, alpha=0.3)
        
        # 5. Quality categories
        ax = axes[1, 1]
        categories = ['Excellent\n(>=0.8)', 'Good\n(0.6-0.8)', 'Fair\n(0.4-0.6)', 'Poor\n(<0.4)']
        counts = [
            sum(1 for s in similarities if s >= 0.8),
            sum(1 for s in similarities if 0.6 <= s < 0.8),
            sum(1 for s in similarities if 0.4 <= s < 0.6),
            sum(1 for s in similarities if s < 0.4)
        ]
        colors_bar = ['#27ae60', '#f39c12', '#e67e22', '#e74c3c']
        bars = ax.bar(categories, counts, color=colors_bar, edgecolor='black', linewidth=1.5)
        
        # Add percentages
        for i, (bar, count) in enumerate(zip(bars, counts)):
            height = bar.get_height()
            percentage = count / len(similarities) * 100
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}\n({percentage:.1f}%)',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Number of Samples', fontweight='bold')
        ax.set_title('(e) Quality Distribution', loc='left', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 6. Performance metrics summary
        ax = axes[1, 2]
        ax.axis('off')
        
        metrics_text = [
            f"Total Samples: {len(similarities)}",
            "",
            "SIMILARITY METRICS:",
            f"  Mean: {np.mean(similarities):.3f}",
            f"  Median: {np.median(similarities):.3f}",
            f"  Std Dev: {np.std(similarities):.3f}",
            "",
            "WORD OVERLAP:",
            f"  Mean: {np.mean(word_overlaps):.1%}",
            f"  Median: {np.median(word_overlaps):.1%}",
            "",
            "LENGTH ANALYSIS:",
            f"  Mean Diff: {np.mean(length_diffs):.1f} words",
            f"  Abs Mean Diff: {np.mean(np.abs(length_diffs)):.1f} words",
            "",
            "QUALITY SUMMARY:",
            f"  High Quality (>=0.6): {sum(1 for s in similarities if s >= 0.6)/len(similarities):.1%}",
            f"  Needs Improvement: {sum(1 for s in similarities if s < 0.6)/len(similarities):.1%}"
        ]
        
        y_pos = 0.95
        for line in metrics_text:
            if line.startswith("  "):
                ax.text(0.1, y_pos, line, fontsize=10, family='monospace',
                       verticalalignment='top')
            elif line == "":
                y_pos -= 0.03
                continue
            else:
                ax.text(0.05, y_pos, line, fontsize=11, fontweight='bold',
                       verticalalignment='top')
            y_pos -= 0.05
        
        ax.set_title('(f) Summary Statistics', loc='left', fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved to: {save_path}")
        plt.close()
    
    def generate_research_summary_report(self, checkpoint_df: pd.DataFrame,
                                        csv_df: pd.DataFrame, results: Dict, save_path: str):
        """Generate comprehensive research summary report - FIXED VERSION"""
        print("Generating comprehensive research summary...")
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("RESEARCH SUMMARY REPORT\n")
            f.write("Game Theory R2GenGPT for Radiology Report Generation\n")
            f.write("=" * 100 + "\n\n")
            
            # 1. Executive Summary
            f.write("1. EXECUTIVE SUMMARY\n")
            f.write("-" * 100 + "\n")
            f.write(f"Training Checkpoint Directory: {self.checkpoint_dir}\n")
            f.write(f"Total Training Epochs: {checkpoint_df['epoch'].max() + 1}\n")
            f.write(f"Total Training Steps: {checkpoint_df['step'].max() + 1}\n")
            f.write(f"Number of Checkpoints: {len(checkpoint_df)}\n")
            
            if csv_df is not None and 'loss' in csv_df.columns:
                loss_data = csv_df['loss'].dropna()
                if len(loss_data) > 0:
                    initial_loss = loss_data.iloc[0]
                    final_loss = loss_data.iloc[-1]
                    if initial_loss > 0:  # Avoid division by zero
                        improvement = ((initial_loss - final_loss) / initial_loss * 100)
                        f.write(f"Training Loss Reduction: {improvement:.2f}%\n")
            
            f.write("\n")
            
            # 2. Best Model Performance
            f.write("2. BEST MODEL PERFORMANCE\n")
            f.write("-" * 100 + "\n")
            
            best_bleu_idx = checkpoint_df['bleu4'].idxmax()
            best_cider_idx = checkpoint_df['cider'].idxmax()
            best_combined_idx = checkpoint_df['combined_score'].idxmax()
            
            f.write("Best BLEU-4 Score:\n")
            f.write(f"  Score: {checkpoint_df.loc[best_bleu_idx, 'bleu4']:.4f}\n")
            f.write(f"  Checkpoint: {checkpoint_df.loc[best_bleu_idx, 'filename']}\n")
            f.write(f"  Epoch/Step: {checkpoint_df.loc[best_bleu_idx, 'epoch']}/{checkpoint_df.loc[best_bleu_idx, 'step']}\n\n")
            
            f.write("Best CIDEr Score:\n")
            f.write(f"  Score: {checkpoint_df.loc[best_cider_idx, 'cider']:.4f}\n")
            f.write(f"  Checkpoint: {checkpoint_df.loc[best_cider_idx, 'filename']}\n")
            f.write(f"  Epoch/Step: {checkpoint_df.loc[best_cider_idx, 'epoch']}/{checkpoint_df.loc[best_cider_idx, 'step']}\n\n")
            
            f.write("Best Combined Score (BLEU-4 + CIDEr):\n")
            f.write(f"  Combined Score: {checkpoint_df.loc[best_combined_idx, 'combined_score']:.4f}\n")
            f.write(f"  BLEU-4: {checkpoint_df.loc[best_combined_idx, 'bleu4']:.4f}\n")
            f.write(f"  CIDEr: {checkpoint_df.loc[best_combined_idx, 'cider']:.4f}\n")
            f.write(f"  Checkpoint: {checkpoint_df.loc[best_combined_idx, 'filename']}\n")
            f.write(f"  Epoch/Step: {checkpoint_df.loc[best_combined_idx, 'epoch']}/{checkpoint_df.loc[best_combined_idx, 'step']}\n\n")
            
            # 3. Training Metrics Analysis
            if csv_df is not None:
                f.write("3. TRAINING METRICS ANALYSIS\n")
                f.write("-" * 100 + "\n")
                
                metrics_to_analyze = {
                    'loss': 'Total Loss',
                    'Bleu_4': 'BLEU-4',
                    'CIDEr': 'CIDEr',
                    'METEOR': 'METEOR',
                    'ROUGE_L': 'ROUGE-L'
                }
                
                for col, name in metrics_to_analyze.items():
                    if col in csv_df.columns:
                        data = csv_df[col].dropna()
                        if len(data) > 0:
                            f.write(f"{name}:\n")
                            f.write(f"  Initial: {data.iloc[0]:.4f}\n")
                            f.write(f"  Final: {data.iloc[-1]:.4f}\n")
                            f.write(f"  Best: {data.min() if 'loss' in col.lower() else data.max():.4f}\n")
                            f.write(f"  Mean +/- Std: {data.mean():.4f} +/- {data.std():.4f}\n")
                            
                            if 'loss' in col.lower():
                                improvement = ((data.iloc[0] - data.iloc[-1]) / data.iloc[0] * 100)
                            else:
                                improvement = ((data.iloc[-1] - data.iloc[0]) / data.iloc[0] * 100) if data.iloc[0] > 0 else 0
                            f.write(f"  Improvement: {improvement:+.2f}%\n\n")
            
            # 4. Checkpoint Analysis
            f.write("4. CHECKPOINT PROGRESSION ANALYSIS\n")
            f.write("-" * 100 + "\n")
            
            epoch_progress = checkpoint_df.groupby('epoch').agg({
                'bleu4': ['min', 'max', 'mean'],
                'cider': ['min', 'max', 'mean']
            })
            
            f.write("Per-Epoch Performance:\n")
            for epoch in sorted(checkpoint_df['epoch'].unique()):
                epoch_data = checkpoint_df[checkpoint_df['epoch'] == epoch]
                f.write(f"\n  Epoch {epoch}:\n")
                f.write(f"    Checkpoints: {len(epoch_data)}\n")
                f.write(f"    BLEU-4 Range: [{epoch_data['bleu4'].min():.4f}, {epoch_data['bleu4'].max():.4f}]\n")
                f.write(f"    CIDEr Range: [{epoch_data['cider'].min():.4f}, {epoch_data['cider'].max():.4f}]\n")
                f.write(f"    Best Combined: {epoch_data['combined_score'].max():.4f}\n")
            
            f.write("\n\n")
            
            # 5. Prediction Quality Analysis
            if 'references' in results:
                f.write("5. PREDICTION QUALITY ANALYSIS\n")
                f.write("-" * 100 + "\n")
                
                pred_keys = [k for k in results.keys() if k.startswith('epoch')]
                if pred_keys:
                    latest_key = sorted(pred_keys, key=lambda x: results[x]['step'])[-1]
                    f.write(f"Analysis based on: {latest_key}\n")
                    f.write(f"  Epoch: {results[latest_key]['epoch']}\n")
                    f.write(f"  Step: {results[latest_key]['step']}\n")
                    f.write(f"  Total Samples: {len(results['references'])}\n\n")
                    
                    f.write("See 'detailed_comparison_report.txt' for sample-by-sample analysis.\n")
            
            f.write("\n\n")
            
            # 6. Recommendations
            f.write("6. RECOMMENDATIONS FOR RESEARCH PAPER\n")
            f.write("-" * 100 + "\n")
            f.write("Key Findings to Highlight:\n")
            f.write(f"  * Best BLEU-4 Score: {checkpoint_df['bleu4'].max():.4f}\n")
            f.write(f"  * Best CIDEr Score: {checkpoint_df['cider'].max():.4f}\n")
            
            # Fix division by zero
            min_score = checkpoint_df['combined_score'].min()
            if min_score > 0:
                improvement_pct = ((checkpoint_df['combined_score'].max() - min_score) / min_score * 100)
                f.write(f"  * Overall Score Improvement: {improvement_pct:.1f}%\n\n")
            else:
                f.write(f"  * Overall Score Improvement: Significant (from zero baseline)\n\n")
            
            
            f.write(f"  * Overall Score Improvement: {((checkpoint_df['combined_score'].max() - checkpoint_df['combined_score'].min()) / checkpoint_df['combined_score'].min() * 100):.1f}%\n\n")
            
            f.write("Recommended Checkpoint for Evaluation:\n")
            f.write(f"  File: {checkpoint_df.loc[best_combined_idx, 'filename']}\n")
            f.write(f"  Rationale: Best combined BLEU-4 and CIDEr performance\n\n")
            
            f.write("Suggested Visualizations for Paper:\n")
            f.write("  1. publication_training_curves.png - Shows overall training progression\n")
            f.write("  2. checkpoint_performance.png - Demonstrates model improvement\n")
            f.write("  3. error_analysis.png - Quality distribution and error patterns\n")
            f.write("  4. statistical_summary_table.png - Comprehensive metrics overview\n\n")
            
            f.write("=" * 100 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 100 + "\n")
        
        print(f"Saved comprehensive research summary to: {save_path}")
        
        
        
    def analyze_all_epochs_performance(self, checkpoint_df: pd.DataFrame, csv_df: pd.DataFrame, results: Dict, output_dir: Path):
        """Analyze performance across all available epochs"""
        print("Analyzing performance across all epochs...")
        
        analysis_content = "EPOCH-WISE PERFORMANCE ANALYSIS\n"
        analysis_content += "=" * 80 + "\n\n"
        
        # Checkpoint-based analysis
        analysis_content += "CHECKPOINT-BASED METRICS:\n"
        analysis_content += "-" * 40 + "\n"
        
        epoch_performance = checkpoint_df.groupby('epoch').agg({
            'bleu4': ['max', 'mean', 'min'],
            'cider': ['max', 'mean', 'min'],
            'combined_score': ['max', 'mean', 'min']
        }).round(4)
        
        analysis_content += f"{'Epoch':>6} {'BLEU-4 Max':>10} {'CIDEr Max':>10} {'Combined Max':>12}\n"
        analysis_content += "-" * 50 + "\n"
        
        for epoch in sorted(checkpoint_df['epoch'].unique()):
            epoch_data = checkpoint_df[checkpoint_df['epoch'] == epoch]
            analysis_content += f"{epoch:>6} {epoch_data['bleu4'].max():>10.4f} {epoch_data['cider'].max():>10.4f} {epoch_data['combined_score'].max():>12.4f}\n"
        
        analysis_content += "\n\n"
        
        # CSV metrics analysis
        if csv_df is not None:
            analysis_content += "CSV TRAINING METRICS:\n"
            analysis_content += "-" * 40 + "\n"
            
            if 'epoch' in csv_df.columns:
                csv_epochs = sorted(csv_df['epoch'].dropna().unique())
                analysis_content += f"Epochs with CSV metrics: {csv_epochs}\n"
                
                # Show available metrics per epoch
                for epoch in csv_epochs[:10]:  # First 10 epochs
                    epoch_metrics = csv_df[csv_df['epoch'] == epoch]
                    if not epoch_metrics.empty:
                        latest = epoch_metrics.iloc[-1]
                        analysis_content += f"\nEpoch {int(epoch)}:\n"
                        for metric in ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'ROUGE_L', 'CIDEr']:
                            if metric in latest and pd.notna(latest[metric]):
                                analysis_content += f"  {metric}: {latest[metric]:.4f}\n"
        
        # Save analysis
        analysis_path = output_dir / "epoch_performance_analysis.txt"
        with open(analysis_path, 'w') as f:
            f.write(analysis_content)
        
        print(f"Saved epoch performance analysis to: {analysis_path}")
    
    def run_complete_analysis(self, output_dir: str = None):
        """Run complete research-grade analysis - ENHANCED VERSION"""
        if output_dir is None:
            output_dir = self.checkpoint_dir / "analysis"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print("\n" + "=" * 100)
        print("STARTING RESEARCH-GRADE ANALYSIS")
        print("Game Theory R2GenGPT - Radiology Report Generation")
        print("=" * 100 + "\n")
        
        # 1. Extract checkpoint info FIRST
        print("Step 1/8: Extracting checkpoint information...")
        checkpoint_df = self.extract_checkpoint_info()
        if checkpoint_df.empty:
            print("No checkpoints found!")
            return
        
        # 2. Load CSV metrics SECOND
        print("\nStep 2/8: Loading training metrics...")
        csv_df = self.load_csv_metrics()
        
        # 3. Load result JSONs
        print("\nStep 3/8: Loading predictions and references...")
        results = self.load_result_jsons()
        
        # NEW: Analyze all epochs
        print("\nAnalyzing performance across all epochs...")
        self.analyze_all_epochs_performance(checkpoint_df, csv_df, results, output_dir)
        
        # NOW call generate_results_table_for_paper AFTER variables are defined
        print("\nGenerating results table for paper...")
        self.generate_results_table_for_paper(
            checkpoint_df,
            csv_df,
            str(output_dir / "paper_results_table.tex")
        )
        
        # Continue with the rest of the analysis...
        # 4. Generate publication-quality training curves
        print("\nStep 4/8: Creating publication-quality training curves...")
        if csv_df is not None:
            self.plot_publication_training_curves(
                csv_df,
                str(output_dir / "publication_training_curves.png")
            )
        
        # 5. Generate checkpoint performance analysis
        print("\nStep 5/8: Creating checkpoint performance analysis...")
        self.plot_checkpoint_performance(
            checkpoint_df,
            str(output_dir / "checkpoint_performance.png")
        )
        
        # 6. Generate performance heatmap
        print("\nStep 6/8: Creating performance heatmap...")
        self.plot_performance_heatmap(
            checkpoint_df,
            str(output_dir / "performance_heatmap.png")
        )
        
        # 7. Generate statistical summary table
        print("\nStep 7/8: Creating statistical summary table...")
        self.generate_statistical_summary_table(
            checkpoint_df,
            csv_df,
            str(output_dir / "statistical_summary_table.png")
        )
        
        # 8. Generate error analysis
        print("\nStep 8/8: Creating error analysis...")
        if results:
            self.plot_error_analysis(
                results,
                str(output_dir / "error_analysis.png"),
                num_samples=100
            )
        
        # 9. Generate detailed comparison report
        print("\nGenerating detailed comparison report...")
        if results:
            self.generate_comparison_report(
                results,
                str(output_dir / "detailed_comparison_report.txt"),
                num_samples=30
            )
        
        # 10. Generate research summary report
        print("\nGenerating comprehensive research summary...")
        self.generate_research_summary_report(
            checkpoint_df,
            csv_df,
            results,
            str(output_dir / "research_summary_report.txt")
        )
        
        # 11. Save dataframes
        print("\nSaving raw data exports...")
        checkpoint_df.to_csv(output_dir / "checkpoint_metrics.csv", index=False)
        if csv_df is not None:
            csv_df.to_csv(output_dir / "training_metrics.csv", index=False)
        
        # 12. Generate LaTeX-ready tables
        print("\nGenerating LaTeX-ready tables...")
        self.generate_latex_tables(checkpoint_df, csv_df, output_dir)
        
        print("\n" + "=" * 100)
        print("ANALYSIS COMPLETE!")
        print("=" * 100)
        print(f"\nAll results saved to: {output_dir}\n")
        print("Generated Files:")
        print("  Visualizations:")
        print("     * publication_training_curves.png - Main training curves")
        print("     * checkpoint_performance.png - Checkpoint analysis")
        print("     * performance_heatmap.png - Performance heatmap")
        print("     * statistical_summary_table.png - Statistical overview")
        print("     * error_analysis.png - Quality analysis")
        print("\n  Reports:")
        print("     * research_summary_report.txt - Executive summary")
        print("     * detailed_comparison_report.txt - Sample-by-sample comparison")
        print("\n  Data Exports:")
        print("     * checkpoint_metrics.csv - All checkpoint data")
        print("     * training_metrics.csv - Training history")
        print("     * latex_tables.tex - LaTeX-ready tables for paper")
        print("\n  For your research paper, use:")
        print("     1. publication_training_curves.png (Figure 1)")
        print("     2. checkpoint_performance.png (Figure 2)")
        print("     3. error_analysis.png (Figure 3)")
        print("     4. statistical_summary_table.png (Table 1)")
        print("     5. latex_tables.tex (Copy tables to paper)")
        print("\n" + "=" * 100 + "\n")
    
    

    def generate_results_table_for_paper(self, checkpoint_df: pd.DataFrame, csv_df: pd.DataFrame, save_path: str):
        """Generate comprehensive results table using CSV metrics - ENHANCED VERSION"""
        print("Generating results table for paper...")
        
        if csv_df is None:
            print("No CSV metrics available, using checkpoint metrics only")
            self._generate_results_from_checkpoints(checkpoint_df, save_path)
            return
        
        # Find the best epoch based on combined score from CSV
        if 'Bleu_4' in csv_df.columns and 'CIDEr' in csv_df.columns:
            # Filter only rows with actual metric values
            valid_metrics = csv_df.dropna(subset=['Bleu_4', 'CIDEr'])
            if len(valid_metrics) == 0:
                print("No valid metrics in CSV, using checkpoint metrics")
                self._generate_results_from_checkpoints(checkpoint_df, save_path)
                return
                
            valid_metrics['combined_score'] = 0.5 * valid_metrics['Bleu_4'] + 0.5 * valid_metrics['CIDEr']
            best_epoch_idx = valid_metrics['combined_score'].idxmax()
            best_metrics = valid_metrics.loc[best_epoch_idx]
            
            # Get corresponding checkpoint info
            best_epoch = int(best_metrics['epoch']) if 'epoch' in best_metrics and pd.notna(best_metrics['epoch']) else int(best_epoch_idx)
            corresponding_checkpoint = checkpoint_df[checkpoint_df['epoch'] == best_epoch]
            
            if not corresponding_checkpoint.empty:
                checkpoint_info = f"Epoch {best_epoch}, Step {corresponding_checkpoint.iloc[0]['step']}"
            else:
                checkpoint_info = f"Epoch {best_epoch}"
        else:
            # Fallback to checkpoint-based selection
            best_idx = checkpoint_df['combined_score'].idxmax()
            best_checkpoint = checkpoint_df.loc[best_idx]
            checkpoint_info = f"Epoch {best_checkpoint['epoch']}, Step {best_checkpoint['step']}"
            best_metrics = None
        
        # Create LaTeX table with ALL metrics from CSV
        latex_content = """
        \\begin{table}[h]
        \\centering
        \\caption{Performance Results on Medical Report Generation}
        \\label{tab:results}
        \\begin{tabular}{lcc}
        \\toprule
        \\textbf{Metric} & \\textbf{Value} & \\textbf{Checkpoint} \\\\
        \\midrule
        """
        
        # Add all available metrics from CSV
        metrics_mapping = {
            'Bleu_1': 'BLEU-1',
            'Bleu_2': 'BLEU-2', 
            'Bleu_3': 'BLEU-3',
            'Bleu_4': 'BLEU-4',
            'ROUGE_L': 'ROUGE-L',
            'CIDEr': 'CIDEr',
            'METEOR': 'METEOR'
        }
        
        metrics_found = []
        
        if best_metrics is not None:
            for col, metric_name in metrics_mapping.items():
                if col in csv_df.columns and col in best_metrics and pd.notna(best_metrics[col]):
                    value = best_metrics[col]
                    latex_content += f"{metric_name} & {value:.4f} & \\\\\n"
                    metrics_found.append((metric_name, value))
        
        # If no CSV metrics found, fall back to checkpoint metrics
        if not metrics_found:
            best_idx = checkpoint_df['combined_score'].idxmax()
            best_checkpoint = checkpoint_df.loc[best_idx]
            
            for metric_key, metric_name in {'bleu4': 'BLEU-4', 'cider': 'CIDEr'}.items():
                if metric_key in best_checkpoint:
                    value = best_checkpoint[metric_key]
                    latex_content += f"{metric_name} & {value:.4f} & \\\\\n"
                    metrics_found.append((metric_name, value))
        
        latex_content += f"\\midrule\nCheckpoint & & {checkpoint_info} \\\\\n"
        latex_content += "\\bottomrule\n\\end{tabular}\n\\end{table}"
        
        # Save LaTeX table
        with open(save_path, 'w') as f:
            f.write(latex_content)
        
        print(f"Saved LaTeX table to: {save_path}")
        
        # Also generate a CSV version for easy copying
        csv_data = []
        for metric_name, value in metrics_found:
            csv_data.append([metric_name, f"{value:.4f}"])
        
        csv_df_results = pd.DataFrame(csv_data, columns=['Metric', 'Value'])
        csv_save_path = str(save_path).replace('.tex', '.csv')
        csv_df_results.to_csv(csv_save_path, index=False)
        print(f"Saved CSV results to: {csv_save_path}")
        
        # Print results to console
        print("\n" + "="*50)
        print("COMPLETE RESULTS FOR PAPER TABLE:")
        print("="*50)
        for metric_name, value in metrics_found:
            print(f"{metric_name}: {value:.4f}")
        print(f"Checkpoint: {checkpoint_info}")
        print("="*50)
        

    def _generate_results_from_checkpoints(self, checkpoint_df: pd.DataFrame, save_path: str):
        """Fallback method using only checkpoint metrics"""
        best_idx = checkpoint_df['combined_score'].idxmax()
        best_checkpoint = checkpoint_df.loc[best_idx]
        
        # Create LaTeX table
        latex_content = """
        \\begin{table}[h]
        \\centering
        \\caption{Performance Results on [Dataset Name]}
        \\label{tab:results}
        \\begin{tabular}{lcc}
        \\toprule
        \\textbf{Metric} & \\textbf{Value} & \\textbf{Checkpoint} \\\\
        \\midrule
        """
        
        # Add available metrics
        for metric_key, metric_name in {'bleu4': 'BLEU-4', 'cider': 'CIDEr'}.items():
            if metric_key in best_checkpoint:
                value = best_checkpoint[metric_key]
                latex_content += f"{metric_name} & {value:.4f} & \\\\\n"
        
        checkpoint_info = f"Epoch {best_checkpoint['epoch']}, Step {best_checkpoint['step']}"
        latex_content += f"\\midrule\nCheckpoint & & {checkpoint_info} \\\\\n"
        
        latex_content += """
        \\bottomrule
        \\end{tabular}
        \\end{table}
        """
        
        with open(save_path, 'w') as f:
            f.write(latex_content)
        
        print(f"Saved LaTeX table to: {save_path} (checkpoint metrics only)")
    
    
    def generate_latex_tables(self, checkpoint_df: pd.DataFrame, 
                             csv_df: pd.DataFrame, output_dir: Path):
        """Generate LaTeX-ready tables for research paper"""
        latex_file = output_dir / "latex_tables.tex"
        
        with open(latex_file, 'w') as f:
            f.write("% LaTeX Tables for Research Paper\n")
            f.write("% Game Theory R2GenGPT Results\n\n")
            
            # Table 1: Best Model Performance
            f.write("% Table 1: Best Model Performance\n")
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{Best Model Performance Across Training}\n")
            f.write("\\label{tab:best_performance}\n")
            f.write("\\begin{tabular}{lcccc}\n")
            f.write("\\toprule\n")
            f.write("Metric & Best Score & Epoch & Step & Checkpoint \\\\\n")
            f.write("\\midrule\n")
            
            best_bleu_idx = checkpoint_df['bleu4'].idxmax()
            best_cider_idx = checkpoint_df['cider'].idxmax()
            best_combined_idx = checkpoint_df['combined_score'].idxmax()
            
            f.write(f"BLEU-4 & {checkpoint_df.loc[best_bleu_idx, 'bleu4']:.4f} & "
                   f"{checkpoint_df.loc[best_bleu_idx, 'epoch']} & "
                   f"{checkpoint_df.loc[best_bleu_idx, 'step']} & "
                   f"\\texttt{{epoch{checkpoint_df.loc[best_bleu_idx, 'epoch']}_step{checkpoint_df.loc[best_bleu_idx, 'step']}}} \\\\\n")
            
            f.write(f"CIDEr & {checkpoint_df.loc[best_cider_idx, 'cider']:.4f} & "
                   f"{checkpoint_df.loc[best_cider_idx, 'epoch']} & "
                   f"{checkpoint_df.loc[best_cider_idx, 'step']} & "
                   f"\\texttt{{epoch{checkpoint_df.loc[best_cider_idx, 'epoch']}_step{checkpoint_df.loc[best_cider_idx, 'step']}}} \\\\\n")
            
            f.write(f"Combined & {checkpoint_df.loc[best_combined_idx, 'combined_score']:.4f} & "
                   f"{checkpoint_df.loc[best_combined_idx, 'epoch']} & "
                   f"{checkpoint_df.loc[best_combined_idx, 'step']} & "
                   f"\\texttt{{epoch{checkpoint_df.loc[best_combined_idx, 'epoch']}_step{checkpoint_df.loc[best_combined_idx, 'step']}}} \\\\\n")
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n\n")
            
            # Table 2: Training Metrics Summary
            if csv_df is not None:
                f.write("% Table 2: Training Metrics Summary\n")
                f.write("\\begin{table}[h]\n")
                f.write("\\centering\n")
                f.write("\\caption{Training Metrics Summary}\n")
                f.write("\\label{tab:training_summary}\n")
                f.write("\\begin{tabular}{lccccc}\n")
                f.write("\\toprule\n")
                f.write("Metric & Initial & Final & Best & Mean $\\pm$ Std & Improvement (\\%) \\\\\n")
                f.write("\\midrule\n")
                
                metrics = {
                    'loss': 'Loss',
                    'Bleu_4': 'BLEU-4',
                    'CIDEr': 'CIDEr',
                    'METEOR': 'METEOR',
                    'ROUGE_L': 'ROUGE-L'
                }
                
                for col, name in metrics.items():
                    if col in csv_df.columns:
                        data = csv_df[col].dropna()
                        if len(data) > 0:
                            initial = data.iloc[0]
                            final = data.iloc[-1]
                            best = data.min() if 'loss' in col.lower() else data.max()
                            mean = data.mean()
                            std = data.std()
                            
                            if 'loss' in col.lower():
                                improvement = ((initial - final) / initial * 100)
                            else:
                                improvement = ((final - initial) / initial * 100) if initial > 0 else 0
                            
                            f.write(f"{name} & {initial:.4f} & {final:.4f} & {best:.4f} & "
                                   f"{mean:.4f} $\\pm$ {std:.4f} & {improvement:+.2f} \\\\\n")
                
                f.write("\\bottomrule\n")
                f.write("\\end{tabular}\n")
                f.write("\\end{table}\n\n")
            
            # Table 3: Epoch-wise Progress
            f.write("% Table 3: Epoch-wise Best Scores\n")
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{Epoch-wise Best Scores}\n")
            f.write("\\label{tab:epoch_progress}\n")
            f.write("\\begin{tabular}{cccc}\n")
            f.write("\\toprule\n")
            f.write("Epoch & Best BLEU-4 & Best CIDEr & Best Combined \\\\\n")
            f.write("\\midrule\n")
            
            for epoch in sorted(checkpoint_df['epoch'].unique())[:10]:  # First 10 epochs
                epoch_data = checkpoint_df[checkpoint_df['epoch'] == epoch]
                f.write(f"{epoch} & {epoch_data['bleu4'].max():.4f} & "
                       f"{epoch_data['cider'].max():.4f} & "
                       f"{epoch_data['combined_score'].max():.4f} \\\\\n")
            
            if len(checkpoint_df['epoch'].unique()) > 10:
                f.write("\\vdots & \\vdots & \\vdots & \\vdots \\\\\n")
                last_epoch = checkpoint_df['epoch'].max()
                epoch_data = checkpoint_df[checkpoint_df['epoch'] == last_epoch]
                f.write(f"{last_epoch} & {epoch_data['bleu4'].max():.4f} & "
                       f"{epoch_data['cider'].max():.4f} & "
                       f"{epoch_data['combined_score'].max():.4f} \\\\\n")
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n\n")
        
        print(f"Saved LaTeX tables to: {latex_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Research-grade analysis for Game Theory R2GenGPT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_results_research.py --checkpoint_dir /path/to/checkpoints
  python analyze_results_research.py --checkpoint_dir /path/to/checkpoints --output_dir /path/to/output
        """
    )
    parser.add_argument('--checkpoint_dir', type=str, 
                       required=True,
                       help='Path to checkpoint directory')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for analysis results (default: checkpoint_dir/analysis)')
    
    args = parser.parse_args()
    
    # Validate checkpoint directory
    if not Path(args.checkpoint_dir).exists():
        print(f"Error: Checkpoint directory does not exist: {args.checkpoint_dir}")
        return
    
    print("\n" + "=" * 100)
    print("GAME THEORY R2GENGPT - RESEARCH ANALYSIS TOOL")
    print("=" * 100)
    print(f"\nCheckpoint Directory: {args.checkpoint_dir}")
    print(f"Output Directory: {args.output_dir or args.checkpoint_dir + '/analysis'}")
    print("\n" + "=" * 100 + "\n")
    
    # Run analysis
    analyzer = ResearchResultsAnalyzer(args.checkpoint_dir)
    analyzer.run_complete_analysis(args.output_dir)


if __name__ == "__main__":
    main()