# coding: utf-8
"""
This modules holds methods for generating predictions from a model.
"""

import os
import sys
from typing import List, Optional
from logging import Logger
import numpy as np

import torch
from torchtext.data import Dataset, Field
from torch import Tensor

from joeynmt.helpers import bpe_postprocess, load_config, make_logger,\
    get_latest_checkpoint, load_checkpoint, store_attention_plots
from joeynmt.metrics import bleu, chrf, token_accuracy, sequence_accuracy
from joeynmt.model import build_model, Model
from joeynmt.batch import Batch
from joeynmt.data import load_data, make_data_iter, MonoDataset
from joeynmt.constants import UNK_TOKEN, PAD_TOKEN, EOS_TOKEN
from joeynmt.vocabulary import Vocabulary

# RL:
# from joeynmt.hrl_agent import build_agent
# from joeynmt.rl_builders import build_environment, build_sampler, EpisodeInfo, \
#     build_loss
from joeynmt.rl_builders import build_agent, build_environment, build_sampler, \
        build_loss
from joeynmt.rl_helpers import evaluate_step, EpisodeInfo


# pylint: disable=too-many-arguments,too-many-locals,no-member
def validate_on_data(model: Model, data: Dataset,
                     logger: Logger,
                     batch_size: int,
                     use_cuda: bool, max_output_length: int,
                     level: str, eval_metric: Optional[str],
                     loss_function: torch.nn.Module = None,
                     beam_size: int = 1, beam_alpha: int = -1,
                     batch_type: str = "sentence"
                     ) \
        -> (float, float, float, List[str], List[List[str]], List[str],
            List[str], List[List[str]], List[np.array]):
    """
    Generate translations for the given data.
    If `loss_function` is not None and references are given,
    also compute the loss.

    :param model: model module
    :param logger: logger
    :param data: dataset for validation
    :param batch_size: validation batch size
    :param use_cuda: if True, use CUDA
    :param max_output_length: maximum length for generated hypotheses
    :param level: segmentation level, one of "char", "bpe", "word"
    :param eval_metric: evaluation metric, e.g. "bleu"
    :param loss_function: loss function that computes a scalar loss
        for given inputs and targets
    :param beam_size: beam size for validation.
        If <2 then greedy decoding (default).
    :param beam_alpha: beam search alpha for length penalty,
        disabled if set to -1 (default).
    :param batch_type: validation batch type (sentence or token)

    :return:
        - current_valid_score: current validation score [eval_metric],
        - valid_loss: validation loss,
        - valid_ppl:, validation perplexity,
        - valid_sources: validation sources,
        - valid_sources_raw: raw validation sources (before post-processing),
        - valid_references: validation references,
        - valid_hypotheses: validation_hypotheses,
        - decoded_valid: raw validation hypotheses (before post-processing),
        - valid_attention_scores: attention scores for validation hypotheses
    """
    if batch_size > 1000 and batch_type == "sentence":
        logger.warning(
            "WARNING: Are you sure you meant to work on huge batches like "
            "this? 'batch_size' is > 1000 for sentence-batching. "
            "Consider decreasing it or switching to"
            " 'eval_batch_type: token'.")
    valid_iter = make_data_iter(dataset=data,
                                batch_size=batch_size,
                                batch_type=batch_type,
                                shuffle=False,
                                train=False)
    valid_sources_raw = data.src
    pad_index = model.src_vocab.stoi[PAD_TOKEN]
    # disable dropout
    model.eval()
    # don't track gradients during validation
    with torch.no_grad():
        all_outputs = []
        valid_attention_scores = []
        total_loss = 0
        total_ntokens = 0
        total_nseqs = 0
        for valid_batch in iter(valid_iter):
            # run as during training to get validation loss (e.g. xent)

            batch = Batch(valid_batch, pad_index, use_cuda=use_cuda)
            # sort batch now by src length and keep track of order
            sort_reverse_index = batch.sort_by_src_lengths()

            # run as during training with teacher forcing
            if loss_function is not None and batch.trg is not None:
                batch_loss = model.get_loss_for_batch(
                    batch, loss_function=loss_function)
                total_loss += batch_loss
                total_ntokens += batch.ntokens
                total_nseqs += batch.nseqs

            # run as during inference to produce translations
            output, attention_scores = model.run_batch(
                batch=batch,
                beam_size=beam_size,
                beam_alpha=beam_alpha,
                max_output_length=max_output_length)

            # sort outputs back to original order
            all_outputs.extend(output[sort_reverse_index])
            valid_attention_scores.extend(
                attention_scores[sort_reverse_index]
                if attention_scores is not None else [])

        assert len(all_outputs) == len(data)

        if loss_function is not None and total_ntokens > 0:
            # total validation loss
            valid_loss = total_loss
            # exponent of token-level negative log prob
            valid_ppl = torch.exp(total_loss / total_ntokens)
        else:
            valid_loss = -1
            valid_ppl = -1

        # decode back to symbols
        decoded_valid = model.trg_vocab.arrays_to_sentences(arrays=all_outputs,
                                                            cut_at_eos=True)

        # evaluate with metric on full dataset
        join_char = " " if level in ["word", "bpe"] else ""
        valid_sources = [join_char.join(s) for s in data.src]
        valid_references = [join_char.join(t) for t in data.trg]
        valid_hypotheses = [join_char.join(t) for t in decoded_valid]

        # post-process
        if level == "bpe":
            valid_sources = [bpe_postprocess(s) for s in valid_sources]
            valid_references = [bpe_postprocess(v) for v in valid_references]
            valid_hypotheses = [bpe_postprocess(v) for v in valid_hypotheses]

        # if references are given, evaluate against them
        if valid_references:
            assert len(valid_hypotheses) == len(valid_references)

            current_valid_score = 0
            if eval_metric.lower() == 'bleu':
                # this version does not use any tokenization
                current_valid_score = bleu(valid_hypotheses, valid_references)
            elif eval_metric.lower() == 'chrf':
                current_valid_score = chrf(valid_hypotheses, valid_references)
            elif eval_metric.lower() == 'token_accuracy':
                current_valid_score = token_accuracy(valid_hypotheses,
                                                     valid_references,
                                                     level=level)
            elif eval_metric.lower() == 'sequence_accuracy':
                current_valid_score = sequence_accuracy(
                    valid_hypotheses, valid_references)
        else:
            current_valid_score = -1

    return current_valid_score, valid_loss, valid_ppl, valid_sources, \
        valid_sources_raw, valid_references, valid_hypotheses, \
        decoded_valid, valid_attention_scores


# pylint: disable-msg=logging-too-many-args
def test(cfg_file,
         ckpt: str,
         output_path: str = None,
         save_attention: bool = False,
         logger: Logger = None) -> None:
    """
    Main test function. Handles loading a model from checkpoint, generating
    translations and storing them and attention plots.

    :param cfg_file: path to configuration file
    :param ckpt: path to checkpoint to load
    :param output_path: path to output
    :param save_attention: whether to save the computed attention weights
    :param logger: log output to this logger (creates new logger if not set)
    """

    if logger is None:
        logger = make_logger()

    cfg = load_config(cfg_file)

    if "test" not in cfg["data"].keys():
        raise ValueError("Test data must be specified in config.")

    # when checkpoint is not specified, take latest (best) from model dir
    if ckpt is None:
        model_dir = cfg["training"]["model_dir"]
        ckpt = get_latest_checkpoint(model_dir)
        if ckpt is None:
            raise FileNotFoundError(
                "No checkpoint found in directory {}.".format(model_dir))
        try:
            step = ckpt.split(model_dir + "/")[1].split(".ckpt")[0]
        except IndexError:
            step = "best"

    batch_size = cfg["training"].get("eval_batch_size",
                                     cfg["training"]["batch_size"])
    batch_type = cfg["training"].get(
        "eval_batch_type", cfg["training"].get("batch_type", "sentence"))
    use_cuda = cfg["training"].get("use_cuda", False)
    level = cfg["data"]["level"]
    eval_metric = cfg["training"]["eval_metric"]
    max_output_length = cfg["training"].get("max_output_length", None)

    # load the data
    _, dev_data, test_data, src_vocab, trg_vocab = load_data(
        data_cfg=cfg["data"])

    data_to_predict = {"dev": dev_data, "test": test_data}

    # load model state from disk
    model_checkpoint = load_checkpoint(ckpt, use_cuda=use_cuda)

    # build model and load parameters into it
    model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)
    model.load_state_dict(model_checkpoint["model_state"])

    if use_cuda:
        model.cuda()

    # whether to use beam search for decoding, 0: greedy decoding
    if "testing" in cfg.keys():
        beam_size = cfg["testing"].get("beam_size", 1)
        beam_alpha = cfg["testing"].get("alpha", -1)
    else:
        beam_size = 1
        beam_alpha = -1

    for data_set_name, data_set in data_to_predict.items():

        #pylint: disable=unused-variable
        score, loss, ppl, sources, sources_raw, references, hypotheses, \
        hypotheses_raw, attention_scores = validate_on_data(
            model, data=data_set, batch_size=batch_size,
            batch_type=batch_type, level=level,
            max_output_length=max_output_length, eval_metric=eval_metric,
            use_cuda=use_cuda, loss_function=None, beam_size=beam_size,
            beam_alpha=beam_alpha, logger=logger)
        #pylint: enable=unused-variable

        if "trg" in data_set.fields:
            decoding_description = "Greedy decoding" if beam_size < 2 else \
                "Beam search decoding with beam size = {} and alpha = {}".\
                    format(beam_size, beam_alpha)
            logger.info("%4s %s: %6.2f [%s]", data_set_name, eval_metric,
                        score, decoding_description)
        else:
            logger.info("No references given for %s -> no evaluation.",
                        data_set_name)

        if save_attention:
            if attention_scores:
                attention_name = "{}.{}.att".format(data_set_name, step)
                attention_path = os.path.join(model_dir, attention_name)
                logger.info(
                    "Saving attention plots. This might take a while..")
                store_attention_plots(attentions=attention_scores,
                                      targets=hypotheses_raw,
                                      sources=data_set.src,
                                      indices=range(len(hypotheses)),
                                      output_prefix=attention_path)
                logger.info("Attention plots saved to: %s", attention_path)
            else:
                logger.warning("Attention scores could not be saved. "
                               "Note that attention scores are not available "
                               "when using beam search. "
                               "Set beam_size to 1 for greedy decoding.")

        if output_path is not None:
            output_path_set = "{}.{}".format(output_path, data_set_name)
            with open(output_path_set, mode="w", encoding="utf-8") as out_file:
                for hyp in hypotheses:
                    out_file.write(hyp + "\n")
            logger.info("Translations saved to: %s", output_path_set)


# def rl_validate(agent, env, sampler, loss, max_steps_per_episode,
#         reward_for_correct, num_subtasks) -> Tensor:
def rl_validate(agent,
                env,
                sampler,
                loss,
                cfg: dict,
                mode: str = "training") -> Tensor:
    """
    Validate the model: Compute loss and return loss and reward.
    :param agent: Agent with which to validate
    :param sampler: Either "training" or "testing"
    :param loss: Loss function. To be passed from build_loss
    :param cfg: Config file
    :type cfg: dict
    :param mode: Either "training" or "testing"
    :type mode: string
    :return: Returns accuracy additionally when testing
    """
    max_steps_per_episode = cfg[mode].get("max_steps_per_episode", 250)
    reward_for_correct = cfg["data"].get("reward_for_correct", 1)
    reward_for_incorrect = cfg["data"].get("reward_for_incorrect", -1)
    reward_for_ask_user = cfg["data"].get("beta", 0.3)
    num_subtasks = cfg["model"].get("num_subtasks", None)
    render = cfg["testing"].get("render", False)
    agent.reset()
    obs = env.reset()
    done = False
    episode_info = EpisodeInfo()

    total_reward = 0.0
    steps_this_episode = 0
    total_correct = 0
    num_ask_user = 0
    correct_per_subtask = {}
    # key: subtask_id. value: 1 or 0 for correct/incorrect respectively

    with torch.no_grad():
        while not done and steps_this_episode < max_steps_per_episode:

            obs, episode_info, reward, done = \
                evaluate_step(agent, env, sampler, episode_info, obs)
            
            if render:
                env.render()

            total_reward += reward
            steps_this_episode += 1
            if reward == reward_for_correct:
                total_correct += 1
                # print('l. 332 total_correct:', total_correct)
                if num_subtasks is not None:
                    subtask_id = episode_info[-1]['agent_id']
                    # last timestep -> subtask_id
                    # print(subtask_id)
                    correct_per_subtask[subtask_id] = 1
            if reward == reward_for_incorrect and num_subtasks is not None:
                subtask_id = episode_info[-1]['agent_id']
                correct_per_subtask[subtask_id] = 0
            
            if reward == -reward_for_ask_user and num_subtasks is not None:
                num_ask_user += 1

        if mode == "training":
            episode_loss = loss(episode_info)
            return episode_loss, total_reward
        else:  # mode == "testing"
            if num_subtasks is None:
                accuracy_this_episode = total_correct  # is either 0 or 1 if normal RL
                return total_reward, accuracy_this_episode

            else:  # i.e. Hierarchical Reinforcement Learning with subtasks
                accuracy_this_episode = total_correct / num_subtasks
                # print('l. 352 total_correct:', total_correct)
                # print('l. 352 num_subtasks:', num_subtasks)
                # print('l. 352 accuracy_this_episode', accuracy_this_episode)

                return total_reward, \
                        accuracy_this_episode, \
                        correct_per_subtask, \
                        num_ask_user


# pylint: disable-msg=logging-too-many-args
def rl_test(cfg_file: str,
            ckpt: str,
            logger: Logger = None,
            output_path: str = None) -> None:
    """
    Main test function for reinforcement learning. 

    :param cfg_file: Path to configuration file
    :type cfg_file: str
    :param ckpt: Path to checkpoint to load
    :type ckpt: str
    :param logger: Log output to this logger (creates new logger if not set)
    :param output_path: Where to save the results
    :type output_path: str
    """

    if logger is None:
        logger = make_logger()

    cfg = load_config(cfg_file)

    # if "test" not in cfg["data"].keys():
    #     raise ValueError("Test data must be specified in config.")

    # when checkpoint is not specified, take latest (best) from model dir
    if ckpt is None:
        model_dir = cfg["training"]["model_dir"]
        ckpt = get_latest_checkpoint(model_dir)
        if ckpt is None:
            raise FileNotFoundError(
                "No checkpoint found in directory {}.".format(model_dir))
        try:
            step = ckpt.split(model_dir + "/")[1].split(".ckpt")[0]
        except IndexError:
            step = "best"

    use_cuda = cfg["training"].get("use_cuda", False)
    eval_metric = cfg["training"]["eval_metric"]

    agent_checkpoint = load_checkpoint(ckpt, use_cuda=use_cuda)

    # build model and load parameters into it
    test_env = build_environment(cfg, test=True)
    agent = build_agent(cfg, test_env)
    agent.load_state_dict(agent_checkpoint["model_state"])
    sampler = build_sampler({'sampler': "greedy"})
    loss = build_loss(cfg, agent)
    max_steps_per_episode = cfg["training"].get("max_steps_per_episode", 250)
    num_episodes = cfg["testing"].get("num_episodes", 128)
    num_subtasks = cfg["model"].get("num_subtasks", None)
    calculate_accuracy = cfg["testing"].get("calculate_accuracy", False)
    render = cfg['testing'].get('render', False)

    if use_cuda:
        agent.cuda()

    accuracies = []  # all accuracies summed up for macro-averaging
    rewards = []
    nums_ask_user = []

    if num_subtasks is not None:
        total_correct_per_subtask = {}

    for _ in range(num_episodes):

        if num_subtasks is None:
            reward, accuracy_this_episode = rl_validate(agent,
                                                        test_env,
                                                        sampler,
                                                        loss,
                                                        cfg,
                                                        mode="testing")
            # accuracy_this_episode is 1 in the case of normal RL. In the case 
            # of Hierarchical RL, it's the accuracy from the episode with 
            # respect to all subtasks.
            accuracies.append(accuracy_this_episode)
            rewards.append(reward)

        else:
            reward, accuracy_this_episode, correct_per_subtask, num_ask_user \
                    = rl_validate(agent,
                                  test_env,
                                  sampler,
                                  loss,
                                  cfg,
                                  mode="testing")
            # accuracy_this_episode is 1 in the case of normal RL. In the case 
            # of Hierarchical RL, it's the accuracy from the episode with 
            # respect to all subtasks.
            accuracies.append(accuracy_this_episode)
            rewards.append(reward)
            nums_ask_user.append(num_ask_user)

            if render:
                test_env.render()

            for subtask, acc in correct_per_subtask.items():
                # acc is the accuracy for this subtask (1 or 0)
                try:
                    total_correct_per_subtask[subtask].append(acc)
                except KeyError:
                    total_correct_per_subtask[subtask] = [acc]
        if render:
            test_env.render()

    average_reward = np.average(rewards)
    macro_averaged_accuracy = np.average(accuracies)

    results1 = 'TEST RESULTS:\n\nAverage reward: {:.4}\n'.format(average_reward) 

    if calculate_accuracy:
        results1 += 'Macro averaged accuracy: {:.2}\n'.format(macro_averaged_accuracy)

    logger.info(results1)

    if output_path is not None:
        with open(output_path, mode="w", encoding="utf-8") as out_file:
            out_file.write(results1)

    if num_subtasks is not None and calculate_accuracy:
        # strict accuracy is 0 if not all subtasks were correct
        strict_accuracies = [0 if acc != 1 else 1 for acc in accuracies]
        average_strict_accuracy = np.average(strict_accuracies)
        average_num_asks = np.average(nums_ask_user)
        results2 = 'Average strict accuracy: {:.2}\n'.format(
            average_strict_accuracy)
        results2 += '# Asks: {:.2f}\n'.format(average_num_asks)

        results2 += 'Accuracies per subtask:\n'
        accuracy_per_subtask = {}
        for subtask, results in total_correct_per_subtask.items():
            acc = np.average(results)
            accuracy_per_subtask[subtask] = acc

        for subtask, acc in accuracy_per_subtask.items():
            results2 += 'Subtask: {:>8}     '.format(str(subtask))
            results2 += 'Accuracy: {:>8.2}\n'.format(acc)

        logger.info(results2)

        if output_path is not None:
            with open(output_path, mode="a", encoding="utf-8") as out_file:
                out_file.write(results2)


def translate(cfg_file, ckpt: str, output_path: str = None) -> None:
    """
    Interactive translation function.
    Loads model from checkpoint and translates either the stdin input or
    asks for input to translate interactively.
    The input has to be pre-processed according to the data that the model
    was trained on, i.e. tokenized or split into subwords.
    Translations are printed to stdout.

    :param cfg_file: path to configuration file
    :param ckpt: path to checkpoint to load
    :param output_path: path to output file
    """
    def _load_line_as_data(line):
        """ Create a dataset from one line via a temporary file. """
        # write src input to temporary file
        tmp_name = "tmp"
        tmp_suffix = ".src"
        tmp_filename = tmp_name + tmp_suffix
        with open(tmp_filename, "w") as tmp_file:
            tmp_file.write("{}\n".format(line))

        test_data = MonoDataset(path=tmp_name, ext=tmp_suffix, field=src_field)

        # remove temporary file
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)

        return test_data

    logger = make_logger()

    def _translate_data(test_data):
        """ Translates given dataset, using parameters from outer scope. """
        # pylint: disable=unused-variable
        score, loss, ppl, sources, sources_raw, references, hypotheses, \
        hypotheses_raw, attention_scores = validate_on_data(
            model, data=test_data, batch_size=batch_size,
            batch_type=batch_type, level=level,
            max_output_length=max_output_length, eval_metric="",
            use_cuda=use_cuda, loss_function=None, beam_size=beam_size,
            beam_alpha=beam_alpha, logger=logger)
        return hypotheses

    cfg = load_config(cfg_file)

    # when checkpoint is not specified, take oldest from model dir
    if ckpt is None:
        model_dir = cfg["training"]["model_dir"]
        ckpt = get_latest_checkpoint(model_dir)

    batch_size = cfg["training"].get("eval_batch_size",
                                     cfg["training"].get("batch_size", 1))
    batch_type = cfg["training"].get(
        "eval_batch_type", cfg["training"].get("batch_type", "sentence"))
    use_cuda = cfg["training"].get("use_cuda", False)
    level = cfg["data"]["level"]
    max_output_length = cfg["training"].get("max_output_length", None)

    # read vocabs
    src_vocab_file = cfg["data"].get(
        "src_vocab", cfg["training"]["model_dir"] + "/src_vocab.txt")
    trg_vocab_file = cfg["data"].get(
        "trg_vocab", cfg["training"]["model_dir"] + "/trg_vocab.txt")
    src_vocab = Vocabulary(file=src_vocab_file)
    trg_vocab = Vocabulary(file=trg_vocab_file)

    data_cfg = cfg["data"]
    level = data_cfg["level"]
    lowercase = data_cfg["lowercase"]

    tok_fun = lambda s: list(s) if level == "char" else s.split()

    src_field = Field(init_token=None,
                      eos_token=EOS_TOKEN,
                      pad_token=PAD_TOKEN,
                      tokenize=tok_fun,
                      batch_first=True,
                      lower=lowercase,
                      unk_token=UNK_TOKEN,
                      include_lengths=True)
    src_field.vocab = src_vocab

    # load model state from disk
    model_checkpoint = load_checkpoint(ckpt, use_cuda=use_cuda)

    # build model and load parameters into it
    model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)
    model.load_state_dict(model_checkpoint["model_state"])

    if use_cuda:
        model.cuda()

    # whether to use beam search for decoding, <2: greedy decoding
    if "testing" in cfg.keys():
        beam_size = cfg["testing"].get("beam_size", 1)
        beam_alpha = cfg["testing"].get("alpha", -1)
    else:
        beam_size = 1
        beam_alpha = -1

    if not sys.stdin.isatty():
        # input file given
        test_data = MonoDataset(path=sys.stdin, ext="", field=src_field)
        hypotheses = _translate_data(test_data)

        if output_path is not None:
            # write to outputfile if given
            output_path_set = "{}".format(output_path)
            with open(output_path_set, mode="w", encoding="utf-8") as out_file:
                for hyp in hypotheses:
                    out_file.write(hyp + "\n")
            logger.info("Translations saved to: %s.", output_path_set)
        else:
            # print to stdout
            for hyp in hypotheses:
                print(hyp)

    else:
        # enter interactive mode
        batch_size = 1
        batch_type = "sentence"
        while True:
            try:
                src_input = input("\nPlease enter a source sentence "
                                  "(pre-processed): \n")
                if not src_input.strip():
                    break

                # every line has to be made into dataset
                test_data = _load_line_as_data(line=src_input)

                hypotheses = _translate_data(test_data)
                print("JoeyNMT: {}".format(hypotheses[0]))

            except (KeyboardInterrupt, EOFError):
                print("\nBye.")
                break
