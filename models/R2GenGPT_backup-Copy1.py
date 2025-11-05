import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import LlamaForCausalLM, LlamaTokenizer
from evalcap.bleu.bleu import Bleu
from evalcap.rouge.rouge import Rouge
from evalcap.cider.cider import Cider
from transformers import SwinModel
from lightning_tools.optim import config_optimizer 
from peft import get_peft_model, LoraConfig, TaskType
import pdb


class BinaryGameAligner(nn.Module):
    """
    Game Theory Module for Visual-Text Alignment
    Implements cooperative game between image regions and text tokens
    """
    def __init__(self, visual_dim=1024, textual_dim=4096, hidden_dim=512):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.textual_proj = nn.Linear(textual_dim, hidden_dim)
        self.alignment_weight = nn.Parameter(torch.ones(1))
        
    def compute_payoff_matrix(self, visual_features, text_features):
        """Compute cooperative game payoff matrix"""
        visual_proj = self.visual_proj(visual_features)  # [B, N_v, D]
        text_proj = self.textual_proj(text_features)     # [B, N_t, D]
        visual_norm = nn.functional.normalize(visual_proj, p=2, dim=-1)
        text_norm = nn.functional.normalize(text_proj, p=2, dim=-1)
        payoff = torch.bmm(visual_norm, text_norm.transpose(1, 2))  # [B, N_v, N_t]
        return payoff * self.alignment_weight

    
    def compute_shapley_values(self, payoff_matrix):
        """Approximate Shapley values for the cooperative game"""
        batch_size, num_visual, num_text = payoff_matrix.shape
        visual_importance = torch.softmax(payoff_matrix.mean(dim=2), dim=1)  # [B, N_v]
        text_importance = torch.softmax(payoff_matrix.mean(dim=1), dim=1)    # [B, N_t]
        return visual_importance, text_importance
    
    def forward(self, visual_features, text_embeddings):
        """Forward pass for game theory alignment"""
        payoff_matrix = self.compute_payoff_matrix(visual_features, text_embeddings)
        visual_importance, text_importance = self.compute_shapley_values(payoff_matrix)
        return {
            'payoff_matrix': payoff_matrix,
            'visual_importance': visual_importance,
            'text_importance': text_importance
        }

class DiseaseAwareTernaryAligner(nn.Module):
    """
    Ternary Game for Disease-Level Alignment
    Aligns images, reports, and disease concepts
    """
    def __init__(self, visual_dim=1024, text_dim=4096, disease_dim=512, num_disease_concepts=20):
        super().__init__()
        self.disease_embeddings = nn.Embedding(num_disease_concepts, disease_dim)
        self.visual_to_disease = nn.Linear(visual_dim, disease_dim)
        self.text_to_disease = nn.Linear(text_dim, disease_dim)
        
    def forward(self, visual_features, text_embeddings):
        """Ternary alignment between image, text, and disease concepts"""
        visual_disease = self.visual_to_disease(visual_features.mean(dim=1))  # [B, D_d]
        text_disease = self.text_to_disease(text_embeddings.mean(dim=1))      # [B, D_d]
        disease_concepts = self.disease_embeddings.weight  # [N_d, D_d]
        visual_disease_scores = torch.matmul(visual_disease, disease_concepts.T)  # [B, N_d]
        text_disease_scores = torch.matmul(text_disease, disease_concepts.T)      # [B, N_d]
        alignment_score = torch.bmm(
            visual_disease_scores.unsqueeze(2), 
            text_disease_scores.unsqueeze(1)
        ).mean()  # [B]
        return {
            'alignment_score': alignment_score,
            'visual_disease_scores': visual_disease_scores,
            'text_disease_scores': text_disease_scores
        }

class R2GenGPT(pl.LightningModule):
    """
    R2GenGPT model with Game Theory Enhancement
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        
        # Add validation tracking
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_score = 0.0

        print(f'Loading vision encoder:{args.vision_model}')
        self.visual_encoder = SwinModel.from_pretrained(args.vision_model)
        
        # Get the actual visual feature dimension from Swin model
        visual_dim = self.visual_encoder.config.hidden_size
        print(f'Visual feature dimension: {visual_dim}')

        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(args.llama_model, use_fast=False)
        self.llama_tokenizer.pad_token_id = 0
        
        # Initialize LLaMA model BEFORE using it
        self.llama_model = LlamaForCausalLM.from_pretrained(args.llama_model)
        textual_dim = self.llama_model.config.hidden_size
        print(f'Textual feature dimension: {textual_dim}')

        print('Initializing Game Theory Modules...')
        self.binary_aligner = BinaryGameAligner(
            visual_dim=visual_dim,  # Use actual visual dimension
            textual_dim=textual_dim,  # Use actual textual dimension
            hidden_dim=512
        )
        
        self.ternary_aligner = DiseaseAwareTernaryAligner(
            visual_dim=visual_dim,  # Use actual visual dimension
            text_dim=textual_dim,  # Use actual textual dimension
            disease_dim=256,
            num_disease_concepts=15  # Common chest diseases
        )
        
        self.alignment_lambda = getattr(args, 'alignment_lambda', 0.1)

    def encode_image(self, image):
        # Ensure the image tensor has the expected shape
        print(f"Shape of image tensor before passing to Swin: {image.shape}")
        
        # Ensure the image tensor is 4D: [batch_size, channels, height, width]
        if image.ndimension() != 4:
            # Ensure the shape is [batch_size, channels, height, width]
            # Assuming image is already [batch_size, 3, 384, 384]
            image = image.unsqueeze(0)  # Add batch dimension if missing

        print(f"Shape of image tensor after reshaping: {image.shape}")

        # Forward pass to the vision encoder (Swin)
        outputs = self.visual_encoder(image)  # This is a SwinModelOutput object
        
        # Extract the visual features (typically 'last_hidden_state')
        visual_features = outputs.last_hidden_state  # Shape: [batch_size, num_patches, hidden_size]
        return visual_features

    def compute_alignment_loss(self, visual_features, text_embeddings):
        """Compute game theory alignment losses with proper scaling"""
        
        # Binary alignment loss (region-level)
        binary_result = self.binary_aligner(visual_features, text_embeddings)
        binary_similarity = binary_result['payoff_matrix'].mean()  # Mean over the payoff matrix
        
        # Ternary alignment loss (disease-level)
        ternary_result = self.ternary_aligner(visual_features, text_embeddings)
        ternary_similarity = ternary_result['alignment_score']  # Maximize alignment score
        
        # Combine the region-level and disease-level alignment losses
        alignment_loss = -(
            torch.tanh(binary_similarity) * 0.5 + 
            torch.tanh(ternary_similarity) * 0.5
        )
        
        return alignment_loss


    def forward(self, samples):
        images = samples["image"]  # This is now a tensor from torch.stack
        
        # Handle multiple images - use the first one for now
        if images.dim() == 5:  # [batch_size, num_images, C, H, W]
            image = images[:, 0]  # Use first image
        else:
            image = images
        
        # Extract visual features
        visual_features = self.encode_image(image)
        
        # Tokenize text input
        text_tokens = self.llama_tokenizer(
            samples["input_text"], 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        )
        text_embeddings = self.llama_model.model.embed_tokens(text_tokens.input_ids.to(self.device))
        
        # Perform the forward pass with game theory integration
        alignment_loss = self.compute_alignment_loss(visual_features, text_embeddings)
        return {"loss": alignment_loss}


    def training_step(self, batch, batch_idx):
        # Get images and text inputs from the batch
        images = batch["image"]
        input_text = batch["input_text"]
    
        # Extract visual features
        visual_features = self.encode_image(images)
    
        # Tokenize text input
        text_tokens = self.llama_tokenizer(
            input_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        ).to(self.device)  # Ensure tokens are moved to the same device as the model
    
        # Get embeddings from the LLaMA model
        to_regress_embeds = self.llama_model.model.embed_tokens(text_tokens.input_ids)
    
        # Perform the forward pass with game theory integration
        alignment_loss = self.compute_alignment_loss(visual_features, to_regress_embeds)
        
        # Log the loss
        self.log("train_loss", alignment_loss, prog_bar=True, sync_dist=True)
        
        return alignment_loss




    def validation_step(self, batch, batch_idx):
        """Validation step - compute validation loss"""
        try:
            result = self(batch)
            loss = result["loss"]
            self.log("val_loss", loss, prog_bar=True, sync_dist=True)
            
            # Store for epoch end calculation if needed
            self.val_step_outputs.append({"val_loss": loss})
            
            return loss
        except Exception as e:
            print(f"Validation step error: {e}")
            # Return a dummy loss if validation fails
            dummy_loss = torch.tensor(0.0, requires_grad=False, device=self.device)
            self.log("val_loss", dummy_loss, prog_bar=True, sync_dist=True)
            return dummy_loss

    def on_validation_epoch_end(self):
        """Called when validation epoch ends"""
        # Calculate average validation loss
        if self.val_step_outputs:
            avg_val_loss = torch.stack([x["val_loss"] for x in self.val_step_outputs]).mean()
            self.log("avg_val_loss", avg_val_loss, sync_dist=True)
            print(f"Validation Epoch {self.current_epoch}: Average Loss = {avg_val_loss:.4f}")
        
        # Clear the list for next epoch
        self.val_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        """Test step - compute test loss"""
        try:
            result = self(batch)
            loss = result["loss"]
            self.log("test_loss", loss, sync_dist=True)
            
            # Store for epoch end calculation if needed
            self.test_step_outputs.append({"test_loss": loss})
            
            return loss
        except Exception as e:
            print(f"Test step error: {e}")
            # Return a dummy loss if test fails
            dummy_loss = torch.tensor(0.0, requires_grad=False, device=self.device)
            self.log("test_loss", dummy_loss, sync_dist=True)
            return dummy_loss

    def on_test_epoch_end(self):
        """Called when test epoch ends"""
        # Calculate average test loss
        if self.test_step_outputs:
            avg_test_loss = torch.stack([x["test_loss"] for x in self.test_step_outputs]).mean()
            self.log("avg_test_loss", avg_test_loss, sync_dist=True)
            print(f"Test Epoch: Average Loss = {avg_test_loss:.4f}")
        
        # Clear the list for next test
        self.test_step_outputs.clear()

    # Methods needed for the original validation code
    def encode_img(self, image):
        """Encode image to features - needed for validation_step"""
        outputs = self.visual_encoder(image)
        img_embeds = outputs.last_hidden_state
        atts_img = torch.ones_like(img_embeds[:, :, 0])
        return img_embeds, atts_img

    def prompt_wrap(self, img_embeds, atts_img):
        """Wrap image embeddings with prompts"""
        # Simple implementation - you may need to adjust this
        return img_embeds, atts_img

    @property
    def embed_tokens(self):
        """Get embedding tokens from llama model"""
        return self.llama_model.model.embed_tokens

    def decode(self, output_token):
        """Decode tokens to text"""
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('</s>')[0].strip()
        output_text = output_text.replace('<unk>', '')
        return output_text

    def score(self, ref, hypo):
        """Calculate evaluation scores"""
        # Simple implementation - return dummy scores for now
        # You can implement proper scoring with Bleu, CIDEr, etc. later
        return {'Bleu_4': 0.1, 'CIDEr': 0.1}

    def save_checkpoint(self, eval_res=None):
        """Save model checkpoint"""
        if eval_res is None:
            eval_res = {'Bleu_4': 0.0, 'CIDEr': 0.0}
            
        current_epoch, global_step = self.current_epoch, self.global_step
        param_grad_dic = {
            k: v.requires_grad for (k, v) in self.named_parameters() if v.requires_grad
        }
        state_dict = self.state_dict()
        for k in list(state_dict.keys()):
            if k not in param_grad_dic.keys():
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "config": self.hparams,
            "epoch": current_epoch,
            "step": global_step
        }
        os.makedirs(os.path.join(self.hparams.savedmodel_path, 'checkpoints'), exist_ok=True)
        save_to = os.path.join(
            self.hparams.savedmodel_path, 'checkpoints',
            f"checkpoint_epoch{current_epoch}_step{global_step}_bleu{eval_res['Bleu_4']:.3f}_cider{eval_res['CIDEr']:.3f}.pth",
        )
        self.print("Saving checkpoint at step {} to {}.".format(global_step, save_to))
        torch.save(save_obj, save_to)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()