# /workspace/CheXpert/R2GenGPT/configs/config.py

import argparse

def create_parser():
    parser = argparse.ArgumentParser(description="Game Theory Enhanced R2GenGPT")

    # ========================= Dataset Configs ==========================
    parser.add_argument('--test', action='store_true', help="only run test set")
    parser.add_argument('--validate', action='store_true', help="only run validation set")
    parser.add_argument('--dataset', type=str, default='chexpert', help="dataset name")
    parser.add_argument('--annotation', type=str, required=True, help="annotation file")
    parser.add_argument('--base_dir', type=str, required=True, help="base directory for images")
    parser.add_argument('--batch_size', default=4, type=int, help="training batch size")
    parser.add_argument('--val_batch_size', default=8, type=int, help="validation batch size")
    parser.add_argument('--test_batch_size', default=8, type=int, help="test batch size")
    parser.add_argument('--prefetch_factor', default=4, type=int, help="prefetch factor")
    parser.add_argument('--num_workers', default=4, type=int, help="number of workers")

    # ========================= Model Settings ============================
    parser.add_argument('--vision_model', type=str, required=True, help="Path to vision model")
    parser.add_argument('--llama_model', type=str, required=True, help="Path to llama model")
    parser.add_argument('--freeze_vm', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--llm_use_lora', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--llm_r', default=16, type=int)
    parser.add_argument('--llm_alpha', default=16, type=int)
    parser.add_argument('--vis_use_lora', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--vis_r', default=16, type=int)
    parser.add_argument('--vis_alpha', default=16, type=int)
    parser.add_argument('--lora_dropout', default=0.1, type=float)
    parser.add_argument('--global_only', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--low_resource', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--end_sym', default='</s>', type=str)

    # ==================== NOVEL: Game Theory Settings ====================
    parser.add_argument('--use_game_theory', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Enable multi-level game theory framework')
    parser.add_argument('--game_hidden_dim', default=512, type=int,
                        help='Hidden dimension for game theory modules')
    parser.add_argument('--lambda_token', default=0.4, type=float,
                        help='Weight for token-level game loss')
    parser.add_argument('--lambda_section', default=0.2, type=float,
                        help='Weight for section-level game loss')
    parser.add_argument('--lambda_disease', default=0.4, type=float,
                        help='Weight for disease-level game loss')
    parser.add_argument('--use_disease_labels', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Use disease labels for disease-level game')
    parser.add_argument('--nash_iterations', default=5, type=int,
                        help='Iterations for Nash equilibrium solver')
    parser.add_argument('--shapley_samples', default=5, type=int,
                        help='Samples for Shapley value computation')
    parser.add_argument('--num_samples', default=5, type=int, help="Number of samples for custom module")
    parser.add_argument('--image_attention', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help="Enable image attention visualization")

    # ======================== SavedModel Configs ===========================
    parser.add_argument('--savedmodel_path', type=str, required=True, help="Path to save models")
    parser.add_argument('--ckpt_file', type=str, default=None)
    parser.add_argument('--delta_file', type=str, default=None)
    
    # Fix these list arguments - they can't be passed via command line easily
    parser.add_argument('--weights', type=str, default="0.5,0.5", help="Comma-separated weights")
    parser.add_argument('--scorer_types', type=str, default="Bleu_4,CIDEr", help="Comma-separated scorer types")

    # ========================= Learning Configs ==========================
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--gradient_clip_val', default=1.0, type=float)

    # ========================= Decoding Settings ==========================
    parser.add_argument('--beam_size', type=int, default=3)
    parser.add_argument('--do_sample', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--no_repeat_ngram_size', type=int, default=2)
    parser.add_argument('--num_beam_groups', type=int, default=1)
    parser.add_argument('--min_new_tokens', type=int, default=80)
    parser.add_argument('--max_new_tokens', type=int, default=150)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--repetition_penalty', type=float, default=2.0)
    parser.add_argument('--length_penalty', type=float, default=1.0)
    parser.add_argument('--diversity_penalty', type=float, default=0)
    parser.add_argument('--temperature', type=float, default=0.7)

    # ====================== Pytorch Lightning ===========================
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--accelerator', type=str, default="gpu")
    parser.add_argument('--strategy', type=str, default="auto")
    parser.add_argument('--precision', type=str, default='bf16-mixed')
    parser.add_argument('--limit_val_batches', type=float, default=1.0)
    parser.add_argument('--limit_test_batches', type=float, default=1.0)
    parser.add_argument('--limit_train_batches', type=float, default=1.0)
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--every_n_train_steps', type=int, default=0)
    parser.add_argument('--val_check_interval', type=float, default=1.0)
    parser.add_argument('--accumulate_grad_batches', type=int, default=4)
    parser.add_argument('--num_sanity_val_steps', type=int, default=2)

    return parser

# Create the parser instance
parser = create_parser()