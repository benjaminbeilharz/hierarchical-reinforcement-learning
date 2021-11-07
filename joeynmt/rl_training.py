# coding: utf-8
"""
Training module
"""

import argparse
import time
import shutil
from typing import List
import os
import queue

import numpy as np

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from joeynmt.helpers import load_config, log_cfg, \
    load_checkpoint, make_model_dir, \
    make_logger, set_seed, symlink_update, ConfigurationError
from joeynmt.prediction import rl_validate
from joeynmt.builders import build_optimizer, build_scheduler, \
    build_gradient_clipper
from joeynmt.rl_builders import build_loss, build_environment, build_sampler, build_agent
from joeynmt.rl_helpers import EpisodeInfo, evaluate_step


# pylint: disable=too-many-instance-attributes
class TrainManager:
    """ Manages training loop, validations, learning rate scheduling
    and early stopping."""
    def __init__(self, agent, sample, loss, config: dict) -> None:
        """
        Creates a new TrainManager for a model, specified as in configuration.

        :param agent: torch module defining the agent
        :param sample: Sampler used for sampling the actions. To be passed from
                       build_sampler
        :param loss: Loss function. To be passed from build_loss
        :param config: dictionary containing the training configurations
        """
        self.config = config
        train_config = config["training"]

        # files for logging and storing
        self.model_dir = make_model_dir(train_config["model_dir"],
                                        overwrite=train_config.get(
                                            "overwrite", False))
        self.logger = make_logger("{}/train.log".format(self.model_dir))
        self.logging_freq = train_config.get("logging_freq", 100)
        self.valid_report_file = "{}/validations.txt".format(self.model_dir)
        self.tb_writer = SummaryWriter(log_dir=self.model_dir +
                                       "/tensorboard/")

        # agent
        self.agent = agent
        self._log_parameters_list()
        self.sample = sample
        # used for validation
        self.sample_greedy = build_sampler({'sampler': "greedy"})
        # objective
        self.reward_discount_gamma = train_config.get("reward_discount_gamma",
                                                      0.95)
        self.reward_whitening = train_config.get("reward_whitening", True)
        self.loss = loss

        # optimization
        self.learning_rate_min = train_config.get("learning_rate_min", 1.0e-8)

        self.clip_grad_fun = build_gradient_clipper(config=train_config)
        self.optimizer = build_optimizer(config=train_config,
                                         parameters=list(agent.parameters()) +
                                         list(self.loss.parameters()))

        # validation & early stopping
        self.validation_freq = train_config.get("validation_freq", 1000)
        self.log_valid_sents = train_config.get("print_valid_sents", [0, 1, 2])
        self.ckpt_queue = queue.Queue(
            maxsize=train_config.get("keep_last_ckpts", 5))
        self.eval_metric = train_config.get("eval_metric", "reward")

        if self.eval_metric not in ['loss', 'reward']:
            raise ConfigurationError("Invalid setting for 'eval_metric', "
                                     "valid options: 'loss', 'reward'.")
        self.early_stopping_metric = train_config.get("early_stopping_metric",
                                                      "eval_metric")

        # if we schedule after reward, we want to maximize it, else minimize
        # early_stopping_metric decides on how to find the early stopping point:
        # ckpts are written when there's a new high/low score for this metric
        if self.early_stopping_metric == "loss":
            self.minimize_metric = True
        elif self.early_stopping_metric == "reward":
            self.minimize_metric = False
            # eval metric that has to get minimized (not yet implemented)
        else:
            raise ConfigurationError(
                "Invalid setting for 'early_stopping_metric', "
                "valid options: 'loss', 'reward'.")

        # learning rate scheduling
        self.scheduler, self.scheduler_step_at = build_scheduler(
            config=train_config,
            scheduler_mode="min" if self.minimize_metric else "max",
            optimizer=self.optimizer,
            hidden_size=config["model"]["encoder"]["hidden_size"])

        self.epochs = train_config["epochs"]
        self.episodes_per_epoch = train_config["episodes_per_epoch"]
        self.max_steps_per_episode = train_config["max_steps_per_episode"]
        self.update_every_episodes = train_config.get("update_every_episodes",
                                                      64)
        self.update_every_steps = train_config.get("update_every_steps", 5)
        self.loss_during_episodes = train_config.get("loss_during_episodes",
                                                     False)
        if self.loss_during_episodes == True:
            if self.update_every_steps < 2:
                raise ConfigurationError(
                    "Invalid setting for "
                    "'update_every_steps'. When 'loss_during_episodes' is "
                    "True, 'update_every_steps' must be at least 2.")

        # CPU / GPU
        self.use_cuda = train_config["use_cuda"]
        if self.use_cuda:
            self.agent.cuda()
            self.loss.cuda()

        # initialize training statistics
        self.total_episodes = 0
        # stop training if this flag is True by reaching learning rate minimum
        self.stop = False
        self.best_ckpt_iteration = 0
        # initial values for best scores
        self.best_ckpt_score = np.inf if self.minimize_metric else -np.inf
        # comparison function for scores
        self.is_best = lambda score: score < self.best_ckpt_score \
            if self.minimize_metric else score > self.best_ckpt_score

        # model parameters
        if "load_agent" in train_config.keys():
            model_load_path = train_config["load_agent"]

            if "load_pretrained" in train_config.keys():
                self.logger.info("Loading pretrained agent from %s",
                                 model_load_path)
                reset_best_ckpt = train_config.get("reset_best_ckpt", True)
                reset_scheduler = train_config.get("reset_scheduler", True)
                reset_optimizer = train_config.get("reset_optimizer", True)
            else:
                self.logger.info("Loading model from %s", model_load_path)
                reset_best_ckpt = train_config.get("reset_best_ckpt", False)
                reset_scheduler = train_config.get("reset_scheduler", False)
                reset_optimizer = train_config.get("reset_optimizer", False)

            self.init_from_checkpoint(model_load_path,
                                      reset_best_ckpt=reset_best_ckpt,
                                      reset_scheduler=reset_scheduler,
                                      reset_optimizer=reset_optimizer)

    def _save_checkpoint(self) -> None:
        """
        Save the model's current parameters and the training state to a
        checkpoint.

        The training state contains the total number of training steps,
        the total number of training tokens,
        the best checkpoint score and iteration so far,
        and optimizer and scheduler states.

        """
        model_path = "{}/{}.ckpt".format(self.model_dir, self.total_episodes)
        state = {
            "total_episodes":
            self.total_episodes,
            "best_ckpt_score":
            self.best_ckpt_score,
            "best_ckpt_iteration":
            self.best_ckpt_iteration,
            "model_state":
            self.agent.state_dict(),
            "optimizer_state":
            self.optimizer.state_dict(),
            "scheduler_state":
            self.scheduler.state_dict()
            if self.scheduler is not None else None,
            "loss":
            self.loss.state_dict()
        }
        torch.save(state, model_path)
        if self.ckpt_queue.full():
            to_delete = self.ckpt_queue.get()  # delete oldest ckpt
            try:
                os.remove(to_delete)
            except FileNotFoundError:
                self.logger.warning(
                    "Wanted to delete old checkpoint %s but "
                    "file does not exist.", to_delete)

        self.ckpt_queue.put(model_path)

        best_path = "{}/best.ckpt".format(self.model_dir)
        try:
            # create/modify symbolic link for best checkpoint
            symlink_update("{}.ckpt".format(self.total_episodes), best_path)
        except OSError:
            # overwrite best.ckpt
            torch.save(state, best_path)

    def init_from_checkpoint(self,
                             path: str,
                             reset_best_ckpt: bool = False,
                             reset_scheduler: bool = False,
                             reset_optimizer: bool = False) -> None:
        """
        Initialize the trainer from a given checkpoint file.

        This checkpoint file contains not only model parameters, but also
        scheduler and optimizer states, see `self._save_checkpoint`.

        :param path: path to checkpoint
        :param reset_best_ckpt: reset tracking of the best checkpoint,
                                use for domain adaptation with a new dev
                                set or when using a new metric for fine-tuning.
        :param reset_scheduler: reset the learning rate scheduler, and do not
                                use the one stored in the checkpoint.
        :param reset_optimizer: reset the optimizer, and do not use the one
                                stored in the checkpoint.
        """
        model_checkpoint = load_checkpoint(path=path, use_cuda=self.use_cuda)

        # restore model and optimizer parameters
        self.agent.load_state_dict(model_checkpoint["model_state"])

        if not reset_optimizer:
            self.optimizer.load_state_dict(model_checkpoint["optimizer_state"])
        else:
            self.logger.info("Reset optimizer.")

        if not reset_scheduler:
            if model_checkpoint["scheduler_state"] is not None and \
                    self.scheduler is not None:
                self.scheduler.load_state_dict(
                    model_checkpoint["scheduler_state"])
        else:
            self.logger.info("Reset scheduler.")

        # restore counts
        # self.steps = model_checkpoint["steps"]
        self.total_episodes = model_checkpoint["total_episodes"]

        if not reset_best_ckpt:
            self.best_ckpt_score = model_checkpoint["best_ckpt_score"]
            self.best_ckpt_iteration = model_checkpoint["best_ckpt_iteration"]
        else:
            self.logger.info("Reset tracking of the best checkpoint.")

        # move parameters to cuda
        if self.use_cuda:
            self.agent.cuda()

    # pylint: disable=unnecessary-comprehension
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    def train_and_validate(self, train_env, valid_env,
                           num_valid_episodes) -> None:
        """
        Train the model and validate it from time to time on the validation set.

        :param train_env: training environment
        :param valid_env: validation environment
        """
        self.train_env = train_env
        self.valid_env = valid_env

        for epoch_no in range(self.epochs):
            self.logger.info("EPOCH %d", epoch_no + 1)

            if self.scheduler is not None and self.scheduler_step_at == "epoch":
                self.scheduler.step(epoch=epoch_no)

            self.agent.train()

            # Reset statistics for each epoch.
            start = time.time()
            total_valid_duration = 0
            start_episodes = self.total_episodes
            # count = self.update_every_episodes - 1
            self.update_episode_countdown = self.update_every_episodes
            epoch_loss = 0

            # buffer the losses until update condition. Initialize with 0.0
            self.episode_loss_accumulated = 0.0

            for _ in range(self.episodes_per_epoch):
                # reactivate training
                self.agent.train()

                # print(count, update, self.steps)

                episode_loss, episode_reward = self._train_episode()

                update = self.update_episode_countdown == 0

                # count = self.update_every_episodes if update else count
                # count -= 1

                # Only save finaly computed batch_loss of full batch
                if update:
                    self.tb_writer.add_scalar("train/train_batch_loss",
                                              episode_loss,
                                              self.total_episodes)

                epoch_loss += episode_loss.detach().cpu().numpy()

                if self.scheduler is not None and \
                    self.scheduler_step_at == "step":
                    self.scheduler.step()

                # log learning progress
                if self.total_episodes % self.logging_freq == 0 \
                        and self.total_episodes != 0:
                    elapsed = time.time() - start - total_valid_duration
                    elapsed_episodes = self.total_episodes - start_episodes
                    self.logger.info(
                        "Epoch %3d Episode: %8d Episode Loss: %12.6f "
                        "Episode Reward: %.3d, "
                        "Episodes per Sec: %8.0f, Lr: %.6f", epoch_no + 1,
                        self.total_episodes, episode_loss, episode_reward,
                        elapsed_episodes / elapsed,
                        self.optimizer.param_groups[0]["lr"])
                    start = time.time()
                    total_valid_duration = 0
                    start_episodes = self.total_episodes

                if self.total_episodes % self.validation_freq == 0 \
                        and self.total_episodes != 0:

                    valid_start_time = time.time()
                    valid_losses = []
                    valid_rewards = []

                    for _ in range(num_valid_episodes):
                        valid_loss, valid_reward = \
                                rl_validate(self.agent, self.valid_env,
                                            self.sample_greedy, self.loss,
                                            self.config, mode="training")
                        valid_losses.append(valid_loss.item())
                        valid_rewards.append(valid_reward)

                    # print("valid_losses:", valid_losses)
                    valid_loss = np.average(valid_losses)
                    valid_reward = np.average(valid_rewards)

                    # self.episode_loss_accumulated = valid_loss

                    self.tb_writer.add_scalar("valid/valid_loss", valid_loss,
                                              self.total_episodes)
                    self.tb_writer.add_scalar("valid/valid_reward",
                                              valid_reward,
                                              self.total_episodes)

                    if self.early_stopping_metric == "loss":
                        ckpt_score = valid_loss
                    elif self.early_stopping_metric == "reward":
                        ckpt_score = valid_reward

                    new_best = False
                    if self.is_best(ckpt_score):
                        self.best_ckpt_score = ckpt_score
                        self.best_ckpt_iteration = self.total_episodes
                        self.logger.info(
                            'Hooray! New best validation result [%s]!',
                            self.early_stopping_metric)
                        if self.ckpt_queue.maxsize > 0:
                            self.logger.info("Saving new checkpoint.")
                            new_best = True
                            self._save_checkpoint()

                    # append to validation report
                    self._add_report(valid_loss=valid_loss,
                                     valid_reward=valid_reward,
                                     eval_metric=self.eval_metric,
                                     new_best=new_best)

                    # log examples?
                    # # append to validation report

                    # self._log_examples(
                    #     sources_raw=[v for v in valid_sources_raw],
                    #     sources=valid_sources,
                    #     hypotheses_raw=valid_hypotheses_raw,
                    #     hypotheses=valid_hypotheses,
                    #     references=valid_references
                    # )

                    valid_duration = time.time() - valid_start_time
                    total_valid_duration += valid_duration
                    self.logger.info(
                        'Validation result (greedy) at epoch %3d, '
                        'Episode %8d: loss: %8.4f, reward: %8.4f, '
                        'duration: %.4fs', epoch_no + 1, self.total_episodes,
                        valid_loss, valid_reward, valid_duration)

                # reset update countdown
                if update:
                    # print(torch.cuda.memory_summary(device=torch.device('cuda')))
                    self.update_episode_countdown = self.update_every_episodes

                    # # store validation set outputs
                    # self._store_outputs(valid_hypotheses)

                else:  # if not update yet:
                    self.update_episode_countdown -= 1

                if self.stop:
                    break

            if self.stop:
                self.logger.info(
                    'Training ended since minimum lr %f was reached.',
                    self.learning_rate_min)
                break

            self.logger.info('Epoch %3d: total training loss %.2f',
                             epoch_no + 1, epoch_loss)
        else:
            self.logger.info('Training ended after %3d epochs.', epoch_no + 1)
        self.logger.info(
            'Best validation result (greedy) at episode '
            '%8d: %6.2f %s.', self.best_ckpt_iteration, self.best_ckpt_score,
            self.early_stopping_metric)

        self.tb_writer.close()  # close Tensorboard writer

    def _train_episode(self) -> Tensor:
        """
        Train the model on one episode: Compute the loss, make a gradient step.

        :return: loss for batch (sum) and reward for batch
        """
        self.agent.reset()
        obs = self.train_env.reset()
        done = False
        episode_info = EpisodeInfo()

        total_reward = 0.0
        steps_this_episode = 0
        episode_loss = 0.0

        if self.loss_during_episodes:
            update_step_countdown = self.update_every_steps

            while not done and steps_this_episode < self.max_steps_per_episode:
                obs, episode_info, reward, done = evaluate_step(
                    self.agent, self.train_env, self.sample, episode_info, obs)

                total_reward += reward
                steps_this_episode += 1

                if update_step_countdown == 0 and not done:
                    step_loss = self.episode_loss_accumulated + \
                        self.loss(episode_info)
                    # print("step_loss", type(step_loss))
                    self._update(step_loss)
                    self.episode_loss_accumulated = 0.0

                    episode_info.clear(keep_last=False)
                    # episode_loss += step_loss.item()
                    episode_loss += step_loss.detach()
                    # episode_loss += step_loss.detach().cpu().numpy()

                    update_step_countdown = self.update_every_steps

                else:
                    update_step_countdown -= 1

        else:
            while not done and steps_this_episode < self.max_steps_per_episode:
                obs, episode_info, reward, done = evaluate_step(
                    self.agent, self.train_env, self.sample, episode_info, obs)

                total_reward += reward
                steps_this_episode += 1

        if len(episode_info) >= 2:
            # print("episode_loss", type(episode_loss))
            episode_loss += self.loss(episode_info)

        # self.update_episode_countdown -= 1
        update = self.update_episode_countdown == 0

        if update and len(episode_info) >= 2:
            # print("episode_info:", episode_info)
            # add all previous losses and current loss together
            loss = self.episode_loss_accumulated + episode_loss
            self._update(loss)
            self.episode_loss_accumulated = 0.0  # reset loss

        else:
            # if we've just started the new update cycle:
            if self.update_episode_countdown == self.update_every_episodes - 1:
                self.episode_loss_accumulated = 0.0  # reset loss

            # accumulate loss
            else:
                self.episode_loss_accumulated += episode_loss

        # increment episode counter
        self.total_episodes += 1

        return episode_loss, total_reward

    def _update(self, loss):
        loss.backward()

        if self.clip_grad_fun is not None:
            # clip gradients (in-place)
            self.clip_grad_fun(params=self.agent.parameters())

        # make gradient step
        self.optimizer.step()
        self.optimizer.zero_grad()

    def _add_report(self,
                    valid_loss: float,
                    valid_reward: float,
                    eval_metric: str,
                    new_best: bool = False) -> None:
        """
        Append a one-line report to validation logging file.

        :param valid_loss: validation loss (sum over whole validation set)
        :param valid_reward: validation reward
        :param eval_metric: evaluation metric, e.g. "bleu"
        :param new_best: whether this is a new best model
        """
        current_lr = -1
        # ignores other param groups for now
        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']

        if current_lr < self.learning_rate_min:
            self.stop = True

        with open(self.valid_report_file, 'a') as opened_file:
            opened_file.write("Episode: {}\tLoss: {:.5f}\tReward: {:.5f}\t"
                              "LR: {:.8f}\t{}\n".format(
                                  self.total_episodes, valid_loss,
                                  valid_reward, current_lr,
                                  "*" if new_best else ""))

    def _log_parameters_list(self) -> None:
        """
        Write all model parameters (name, shape) to the log.
        """
        model_parameters = filter(lambda p: p.requires_grad,
                                  self.agent.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info("Total params: %d", n_params)
        trainable_params = [
            n for (n, p) in self.agent.named_parameters() if p.requires_grad
        ]
        self.logger.info("Trainable parameters: %s", sorted(trainable_params))
        assert trainable_params

    def _log_examples(self,
                      sources: List[str],
                      hypotheses: List[str],
                      references: List[str],
                      sources_raw: List[List[str]] = None,
                      hypotheses_raw: List[List[str]] = None,
                      references_raw: List[List[str]] = None) -> None:
        """
        Log a the first `self.log_valid_sents` sentences from given examples.

        :param sources: decoded sources (list of strings)
        :param hypotheses: decoded hypotheses (list of strings)
        :param references: decoded references (list of strings)
        :param sources_raw: raw sources (list of list of tokens)
        :param hypotheses_raw: raw hypotheses (list of list of tokens)
        :param references_raw: raw references (list of list of tokens)
        """
        for p in self.log_valid_sents:

            if p >= len(sources):
                continue

            self.logger.info("Example #%d", p)

            if sources_raw is not None:
                self.logger.debug("\tRaw source:     %s", sources_raw[p])
            if references_raw is not None:
                self.logger.debug("\tRaw reference:  %s", references_raw[p])
            if hypotheses_raw is not None:
                self.logger.debug("\tRaw hypothesis: %s", hypotheses_raw[p])

            self.logger.info("\tSource:     %s", sources[p])
            self.logger.info("\tReference:  %s", references[p])
            self.logger.info("\tHypothesis: %s", hypotheses[p])

    def _store_outputs(self, hypotheses: List[str]) -> None:
        """
        Write current validation outputs to file in `self.model_dir.`

        :param hypotheses: list of strings
        """
        current_valid_output_file = "{}/{}.hyps".format(
            self.model_dir, self.total_episodes)
        with open(current_valid_output_file, 'w') as opened_file:
            for hyp in hypotheses:
                opened_file.write("{}\n".format(hyp))


def train(cfg_file: str) -> None:
    """
    Main training function. After training, also test on test data if given.

    :param cfg_file: path to configuration yaml file
    :type cfg_file: string
    """
    cfg = load_config(cfg_file)

    # set the random seed
    set_seed(seed=cfg["training"].get("random_seed", 42))
    warmstart = cfg["training"].get("warmstart", False)

    # load the data
    print("Building environments and loading data. This will take some time.")
    train_env, valid_env = build_environment(cfg)
    print("Environments built.")

    # build an agent
    agent = build_agent(cfg, train_env)
    sampler = build_sampler(cfg['training'])
    loss = build_loss(cfg, agent)

    # for training management, e.g. early stopping and model selection
    trainer = TrainManager(agent=agent, sample=sampler, loss=loss, config=cfg)

    # store copy of original training config in model dir
    shutil.copy2(cfg_file, trainer.model_dir + "/config.yaml")

    # log all entries of config
    log_cfg(cfg, trainer.logger)

    # log_data_info(train_data=train_data, valid_data=dev_data,
    #              test_data=test_data, src_vocab=src_vocab, trg_vocab=trg_vocab,
    #              logging_function=trainer.logger.info)

    trainer.logger.info(str(agent))

    num_valid_episodes = cfg["training"].get("num_valid_episodes", 10)
    # train the model
    trainer.train_and_validate(train_env, valid_env, num_valid_episodes)

    # predict with the best model on validation and test
    # (if test data is available)
    ckpt = "{}/{}.ckpt".format(trainer.model_dir, trainer.best_ckpt_iteration)
    output_name = "{:08d}.hyps".format(trainer.best_ckpt_iteration)
    output_path = os.path.join(trainer.model_dir, output_name)
    # test(cfg_file, ckpt=ckpt, output_path=output_path, logger=trainer.logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Joey-NMT')
    parser.add_argument("config",
                        default="configs/default.yaml",
                        type=str,
                        help="Training configuration file (yaml).")
    args = parser.parse_args()
    train(cfg_file=args.config)
