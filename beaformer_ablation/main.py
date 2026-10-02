import os
import yaml
import argparse
import logging
import sys
import random
import numpy as np
import re


if __name__ == "__main__":
    # setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_fpath", default="config.yaml", type=str)
    args = parser.parse_args()

    with open(args.config_fpath, "r") as f:
        config = yaml.safe_load(f)
        # This setting must be before 'import torch'
        os.environ["CUDA_VISIBLE_DEVICES"] = config["CUDA_VISIBLE_DEVICES"]

    for key, value in config.items():
        setattr(args, key, value)

    for key, value in config.get(args.dataset).items():
        setattr(args, key, value)

    if args.deterministic:
        import torch
        import torch.backends.cudnn as cudnn

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    outputs_dir = args.outputs_dir
    if outputs_dir is None:
        outputs_dir = f"./outputs/{args.dataset}"
        if os.path.exists(outputs_dir):
            pattern = re.compile(r"^run(\d+)$")
            existing = [
                int(m.group(1))
                for name in os.listdir(outputs_dir)
                if (m := pattern.match(name))
            ]
            next_idx = max(existing, default=0) + 1
        else:
            next_idx = 1
        
        outputs_dir = os.path.join(outputs_dir, f"run{next_idx}")
        args.outputs_dir = outputs_dir
    
    os.makedirs(args.outputs_dir, exist_ok=True)

    log_fpath = os.path.join(args.outputs_dir, f"{args.mode}_log.txt")
    logging.basicConfig(
        filename=log_fpath,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    assert args.mode in ["train", "test"], f"invalid mode: {args.mode}"
    import trainer
    import inference

    if args.mode == "train":
        args.checkpoint_fpath = os.path.join(args.outputs_dir, "checkpoint.pth")
        args.best_model_fpath = os.path.join(args.outputs_dir, "best_model.pth")

        train = getattr(trainer, f"train_{args.dataset}")
        train(args)

    elif args.mode == "test":
        if args.pretrained_fpath is None:
            model_state_dict_fpath = os.path.join(args.outputs_dir, "best_model.pth")
            if os.path.exists(model_state_dict_fpath):
                args.pretrained_fpath = model_state_dict_fpath

        infer = getattr(inference, f"infer_{args.dataset}")
        infer(args)
