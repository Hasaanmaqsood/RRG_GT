# /workspace/CheXpert/R2GenGPT/train.py

import os
from pprint import pprint
from configs.config import create_parser
from dataset.data_module import DataModule
from lightning_tools.callbacks import add_callbacks

# FIXED: Import the game theory enhanced model
from models.R2GenGPT_GameTheory import R2GenGPT_GameTheory

from pytorch_lightning import LightningDataModule
from pytorch_lightning import seed_everything
import pytorch_lightning as pl

def train(args):
    print(args)
    
    # Create data module with explicit setup
    dm = DataModule(args)
    dm.setup('fit')  # Explicitly call setup for training
    
    # Setup callbacks
    callbacks = add_callbacks(args)
    
    # Create trainer with explicit settings
    trainer = pl.Trainer(
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=args.strategy,
        accelerator=args.accelerator,
        precision=args.precision,
        val_check_interval=args.val_check_interval,
        limit_val_batches=args.limit_val_batches,
        limit_train_batches=args.limit_train_batches,
        max_epochs=args.max_epochs,
        num_sanity_val_steps=args.num_sanity_val_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=callbacks["callbacks"],
        logger=callbacks["loggers"],
        enable_progress_bar=True,
        enable_model_summary=True,
        enable_checkpointing=True
    )
    
    # Load or create model
    if args.ckpt_file is not None:
        print(f"Loading model from checkpoint: {args.ckpt_file}")
        model = R2GenGPT_GameTheory.load_from_checkpoint(args.ckpt_file, strict=False)
        print(f"Model loaded from checkpoint: {args.ckpt_file}")
    else:
        print("Initializing new R2GenGPT_GameTheory model...")
        model = R2GenGPT_GameTheory(args)
        print("Model initialized from scratch")
    
    # DEBUG: Force check of required methods
    required_methods = ['configure_optimizers', 'training_step', 'forward']
    for method in required_methods:
        if hasattr(model, method) and callable(getattr(model, method)):
            print(f"✓ {method} method found and callable")
        else:
            print(f"✗ {method} method missing or not callable")
    
    # Test configure_optimizers explicitly
    try:
        optimizer_config = model.configure_optimizers()
        print(f"✓ configure_optimizers returns: {type(optimizer_config)}")
        if isinstance(optimizer_config, dict) and 'optimizer' in optimizer_config:
            print("✓ Optimizer configuration is valid")
        else:
            print("⚠️  Optimizer configuration might be invalid")
    except Exception as e:
        print(f"✗ configure_optimizers failed: {e}")
    
    # Run training
    print("Starting training...")
    trainer.fit(model, datamodule=dm)


def main():
    # FIXED: Create parser here
    parser = create_parser()
    args = parser.parse_args()
    
    # Process list arguments
    args.weights = [float(x) for x in args.weights.split(',')]
    args.scorer_types = [x.strip() for x in args.scorer_types.split(',')]
    
    os.makedirs(args.savedmodel_path, exist_ok=True)
    
    print("\n" + "="*70)
    print("GAME THEORY ENHANCED R2GENGPT")
    print("="*70)
    pprint(vars(args))
    print("="*70 + "\n")
    
    seed_everything(42, workers=True)
    train(args)


if __name__ == '__main__':
    main()