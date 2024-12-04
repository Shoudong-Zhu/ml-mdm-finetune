# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.
import logging
import os

import torch

from ml_mdm import helpers, reader
from ml_mdm.config import get_arguments, get_model, get_pipeline
from ml_mdm.language_models import factory
from ml_mdm.trainer import Trainer


def main(args):
    # Setup distributed training (single node, multi-GPU)
    local_rank, global_rank, world_size = helpers.init_distributed_singlenode()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and language model
    tokenizer, language_model = factory.create_lm(args, device=device)
    language_model_dim = language_model.embed_dim
    args.unet_config.conditioning_feature_dim = language_model_dim

    # Load diffusion model
    denoising_model = get_model(args.model)(
        input_channels=3, output_channels=3, unet_config=args.unet_config
    ).to(device)

    diffusion_model = get_pipeline(args.model)(
        denoising_model, diffusion_config=args.diffusion_config
    ).to(device)

    # Load training dataset
    train_loader = reader.get_dataset(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        file_list=args.file_list,
        config=args.reader_config,
        num_epochs=args.num_epochs,
        load_numpy=args.use_precomputed_text_embeddings,
    )

    # Setup optimizer, scheduler, and trainer
    optimizer = torch.optim.AdamW(
        params=diffusion_model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.num_training_steps
    )
    trainer = Trainer(
        model=diffusion_model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        args=args,
    )

    # Training loop
    logging.info(f"Starting training for {args.num_training_steps} steps")
    for step, batch in enumerate(train_loader, start=1):
        loss = trainer.train_step(batch)

        # Log progress
        if step % args.log_freq == 0:
            logging.info(f"Step {step}: Loss = {loss:.4f}")

        # Save checkpoint
        if step % args.save_freq == 0 or step == args.num_training_steps:
            checkpoint_path = os.path.join(args.output_dir, f"model_step_{step}.pth")
            torch.save(diffusion_model.state_dict(), checkpoint_path)
            logging.info(f"Checkpoint saved: {checkpoint_path}")

        # Stop training after reaching the target number of steps
        if step >= args.num_training_steps:
            break

    logging.info("Training completed!")


if __name__ == "__main__":
    args = get_arguments(mode="trainer")
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    helpers.print_args(args)
    main(args)
