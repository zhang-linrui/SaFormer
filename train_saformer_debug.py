import os
os.environ['WANDB_MODE'] = 'offline'
import uuid
import types
from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import bullet_safety_gym  # noqa
import dsrl
#import gymnasium as gym  # noqa
import numpy as np
import pyrallis
import torch
from dsrl.infos import DENSITY_CFG
from dsrl.offline_env import OfflineEnvWrapper, wrap_env  # noqa
from fsrl.utils import WandbLogger
from torch.utils.data import DataLoader
from tqdm.auto import trange  # noqa

from examples.configs.saformer_configs import SaFormer_DEFAULT_CONFIG, SaFormerTrainConfig
from osrl.algorithms import SaFormer, SaFormerTrainer
from osrl.common.saformer_dataset import SequenceDataset
from osrl.common.exp_util import auto_name, seed_all


@pyrallis.wrap()
def train(args: SaFormerTrainConfig):
    # update config
    cfg, old_cfg = asdict(args), asdict(SaFormerTrainConfig())
    differing_values = {key: cfg[key] for key in cfg.keys() if cfg[key] != old_cfg[key]}
    cfg = asdict(SaFormer_DEFAULT_CONFIG[args.task]())
    cfg.update(differing_values)
    args = types.SimpleNamespace(**cfg)

    # setup logger
    default_cfg = asdict(SaFormer_DEFAULT_CONFIG[args.task]())
    if args.name is None:
        args.name = auto_name(default_cfg, cfg, args.prefix, args.suffix)
    if args.group is None:
        args.group = args.task + "-cost-" + str(int(args.cost_limit))
    if args.logdir is not None:
        args.logdir = os.path.join(args.logdir, args.group, args.name)
    
    logger = None
    # logger = WandbLogger(cfg, args.project, args.group, args.name, args.logdir)
    # logger.save_config(cfg, verbose=args.verbose)

    # set seed
    seed_all(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(args.threads)

    # initialize environment
    if "Metadrive" in args.task:
        import gym
    env = gym.make(args.task)

    # print(env.action_space)
    # raise

    # pre-process offline dataset
    data = env.get_dataset()
    env.set_target_cost(args.cost_limit)

    cbins, rbins, max_npb, min_npb = None, None, None, None
    if args.density != 1.0:
        density_cfg = DENSITY_CFG[args.task + "_density" + str(args.density)]
        cbins = density_cfg["cbins"]
        rbins = density_cfg["rbins"]
        max_npb = density_cfg["max_npb"]
        min_npb = density_cfg["min_npb"]
    data = env.pre_process_data(data,
                                args.outliers_percent,
                                args.noise_scale,
                                args.inpaint_ranges,
                                args.epsilon,
                                args.density,
                                cbins=cbins,
                                rbins=rbins,
                                max_npb=max_npb,
                                min_npb=min_npb)

    # wrapper
    env = wrap_env(
        env=env,
        reward_scale=args.reward_scale,
    )
    env = OfflineEnvWrapper(env)

    ct = lambda x: 70 - x if args.linear else 1 / (x + 10)

    dataset = SequenceDataset(
        data,
        seq_len=args.seq_len,
        reward_scale=args.reward_scale,
        cost_scale=args.cost_scale,
        deg=args.deg,
        pf_sample=args.pf_sample, # False
        max_rew_decrease=args.max_rew_decrease,
        beta=args.beta, # 1.0
        augment_percent=args.augment_percent, # 0.2
        cost_reverse=args.cost_reverse,
        max_reward=args.max_reward,
        min_reward=args.min_reward,
        pf_only=args.pf_only, # False
        rmin=args.rmin,
        cost_bins=args.cost_bins,
        npb=args.npb,
        cost_sample=args.cost_sample,
        cost_transform=ct,
        start_sampling=args.start_sampling,
        prob=args.prob,
        random_aug=args.random_aug,
        aug_rmin=args.aug_rmin,
        aug_rmax=args.aug_rmax,
        aug_cmin=args.aug_cmin,
        aug_cmax=args.aug_cmax,
        cgap=args.cgap,
        rstd=args.rstd,
        cstd=args.cstd,
    )

    import matplotlib.pyplot as plt
    ctg_list=np.array([i['returns'][0] for i in dataset.dataset])
    rtg_list=np.array([i['rewards_to_go'][0] for i in dataset.dataset])
    traj_idx = np.arange(len(ctg_list))
    print(len(traj_idx))
    plt.figure()
    plt.scatter(
        ctg_list[traj_idx],
        rtg_list[traj_idx],
        c='dodgerblue',
        label='remained data'
        )
    fontsize=18
    plt.legend(fontsize=fontsize)
    plt.xlabel("cost return", fontsize=fontsize)
    plt.ylabel("reward return", fontsize=fontsize)
    plt.tight_layout()
    # plt.show()
    # raise

    rt = lambda x: (x - dataset.info['rtg_mean'])/dataset.info['rtg_std']
    ct = lambda x: (x - dataset.info['ctg_mean'])/dataset.info['ctg_std']

    # model & optimizer & scheduler setup
    model = SaFormer(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=env.action_space.high[0],
        embedding_dim=args.embedding_dim,
        seq_len=args.seq_len,
        episode_len=args.episode_len,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        attention_dropout=args.attention_dropout,
        residual_dropout=args.residual_dropout,
        embedding_dropout=args.embedding_dropout,
        time_emb=args.time_emb,
        use_rew=args.use_rew,
        use_cost=args.use_cost,
        reward_transform = rt,
        cost_transform = ct,
        add_cost_feat=args.add_cost_feat,
        mul_cost_feat=args.mul_cost_feat,
        cat_cost_feat=args.cat_cost_feat,
        action_head_layers=args.action_head_layers,
        cost_prefix=args.cost_prefix,
        stochastic=args.stochastic,
        init_temperature=args.init_temperature,
        target_entropy=-env.action_space.shape[0],
    ).to(args.device)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # def checkpoint_fn():
    #     return {"model_state": model.state_dict()}

    # logger.setup_checkpoint_fn(checkpoint_fn)

    rt = lambda x: x*dataset.info['rtg_std'] + dataset.info['rtg_mean']
    # trainer
    trainer = SaFormerTrainer(model,
                         env,
                         logger=logger,
                         learning_rate=args.learning_rate,
                         weight_decay=args.weight_decay,
                         betas=args.betas,
                         clip_grad=args.clip_grad,
                         lr_warmup_steps=args.lr_warmup_steps,
                         reward_scale=args.reward_scale,
                         cost_scale=args.cost_scale,
                         loss_cost_weight=args.loss_cost_weight,
                         loss_state_weight=args.loss_state_weight,
                         cost_reverse=args.cost_reverse,
                         no_entropy=args.no_entropy,
                         reward_transform=rt,
                         device=args.device)

    

    trainloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    trainloader_iter = iter(trainloader)

    # for saving the best
    best_reward = -np.inf
    best_cost = np.inf
    best_idx = 0

    for step in trange(args.update_steps, desc="Training"):
        batch = next(trainloader_iter)
        states, actions, returns, costs_return, time_steps, mask, episode_cost, costs, cost_to_go, reward_to_go = [
            b.to(args.device) for b in batch
        ]
        trainer.train_one_step(states, actions, returns, costs_return, time_steps, mask,
                               episode_cost, costs, cost_to_go, reward_to_go)

        # evaluation
        if (step + 1) % args.eval_every == 0 or step == args.update_steps - 1:
            average_reward, average_cost = [], []
            log_cost, log_reward, log_len = {}, {}, {}
            for target_return in args.target_returns:
                reward_return, cost_return = target_return
                if args.cost_reverse:
                    # critical step, rescale the return!
                    ret, cost, length = trainer.evaluate(
                        args.eval_episodes, reward_return * args.reward_scale,
                        (args.episode_len - cost_return) * args.cost_scale)
                else:
                    ret, cost, length = trainer.evaluate(
                        args.eval_episodes, reward_return * args.reward_scale,
                        cost_return * args.cost_scale)
                average_cost.append(cost)
                average_reward.append(ret)

                name = "c_" + str(int(cost_return)) + "_r_" + str(int(reward_return))
                log_cost.update({name: cost})
                log_reward.update({name: ret})
                log_len.update({name: length})

            # logger.store(tab="cost", **log_cost)
            # logger.store(tab="ret", **log_reward)
            # logger.store(tab="length", **log_len)

            # # save the current weight
            # logger.save_checkpoint()
            # # save the best weight
            mean_ret = np.mean(average_reward)
            mean_cost = np.mean(average_cost)
            if mean_cost < best_cost or (mean_cost == best_cost
                                         and mean_ret > best_reward):
                best_cost = mean_cost
                best_reward = mean_ret
                best_idx = step
            #     logger.save_checkpoint(suffix="best")

            # logger.store(tab="train", best_idx=best_idx)
            # logger.write(step, display=False)

        else:
            pass
            # logger.write_without_reset(step)


if __name__ == "__main__":
    train()
