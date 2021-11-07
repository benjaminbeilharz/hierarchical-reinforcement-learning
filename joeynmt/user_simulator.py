# -*- coding: utf-8 -*-
# Author: Blanca Birn
"""
User Simulator for Interactive Semantic Parsing with Hierarchical Reinforcement Learning
"""

import pickle as pk
from collections import defaultdict
import random, warnings
from nltk import word_tokenize, pos_tag

# paraphrase data extracted from http://paraphrase.org/
# all functions with a * are originally taken from https://github.com/LittleYUYU/Interactive-Semantic-Parsing

TRIGGER_EXPRESS_TO_REMOVE = "this trigger fires"
ACTION_EXPRESS_TO_REMOVE = "this action will"


# *
def extract_fn(words):
    """ Collects the trigger/action functions
        from a string, by extracting the if/then phrase.
        Templates:
        (1) if <t>, (then) <a>
        (2) <a> every time/year/month/week/day/hour <t>
        (3) <a> when <t>
        (4) if <t> then <a>
        (5) <a> if <t>
        (6) when <t>, <a>
        :returns: String of trigger/action descriptions
        """
    tf_desc = []
    af_desc = []

    for word in words:
        words[words.index(word)] = word.lower()

    # check the templates one by one
    if words[0] == "if" and words.count("if") == 1:  # template (1),(4)
        if "," in words:  # template (1)
            tf_desc = words[1:words.index(",")]
            af_desc = words[words.index(",") + 1:]
            if af_desc and af_desc[0] == "then":  # remove the redundant "then"
                af_desc = af_desc[1:]
        elif "then" in words:  # template (4)
            tf_desc = words[1:words.index("then")]
            af_desc = words[words.index("then") + 1:]
    elif words.count("if") == 1:  # template (5)
        tf_desc = words[words.index("if") + 1:]
        af_desc = words[:words.index("if")]
    elif "if" not in words:  # others
        if words.count("when") == 1:  # template (3),(6)
            if words[0] == "when" and "," in words:
                tf_desc = words[1:words.index(",")]
                af_desc = words[words.index(",") + 1:]
            elif words[0] != "when" and "," not in words:
                tf_desc = words[words.index("when") + 1:]
                af_desc = words[:words.index("when")]
        elif "when" not in words:  # template (2)
            phrases = {
                "every time", "every year", "every month", "every week",
                "every day", "every hour"
            }
            picked_phrase = None
            picked_phrase_idx = None
            for word_idx in range(
                    len(words) - 2):  # at least one token following the phrase
                phrase = words[word_idx] + " " + words[word_idx + 1]
                if phrase in phrases:
                    picked_phrase = phrase
                    picked_phrase_idx = word_idx
                    break
            if picked_phrase and picked_phrase_idx:  # idx must > 0
                tf_desc = words[picked_phrase_idx:]
                af_desc = words[:picked_phrase_idx]
            else:  # none of the templates fit
                tf_desc = words
                af_desc = words
    return " ".join(tf_desc), " ".join(af_desc)
    
def extract_fn_user_descriptions(data):
    """Extracts the paraphrases of the trigger/action functions from the user data.
    :returns: function to description mapping
    :rtype: 2 defaultdict(set)
    """
    af2desc = defaultdict(set)
    tf2desc = defaultdict(set)
    source_data = data['train'] + data['dev']
    
    for item in source_data:
        words = item['words']
        tc, tf, ac, af = item['label_names']
        tf_desc = None
        af_desc = None
    
        if len(words) > 4:
            extract_fn(words)
    
        if tf_desc and af_desc:
            tf_fn = "%s.%s" % (tc.lower().strip().decode("utf-8"),
                               tf.lower().strip().decode("utf-8"))
            af_fn = "%s.%s" % (ac.lower().strip().decode("utf-8"),
                               af.lower().strip().decode("utf-8"))
            tf2desc[tf_fn].add(" ".join(tf_desc))
            af2desc[af_fn].add(" ".join(af_desc))
    return tf2desc, af2desc
    
def clean_function_text(fn2text):
    """ Clean function name or description: removing TRIGGER_EXPRESS_TO_REMOVE and ACTION_EXPRESS_TO_REMOVE
    and revise to first person angle.
    :param fn2text: function to description mapping
    :type: defaultdict(set)
    :return: """
    new_fn2text = defaultdict(set)
    for fn, desc in fn2text.items():
        for text in desc:
    
            # remove template words
            if text.startswith(TRIGGER_EXPRESS_TO_REMOVE):
                text = text[len(TRIGGER_EXPRESS_TO_REMOVE) + 1:]
            if text.startswith(ACTION_EXPRESS_TO_REMOVE):
                text = text[len(ACTION_EXPRESS_TO_REMOVE) + 1:]
                # revise to first-person angle
                text = text.replace("your", "my")
    
            if "you" in text:
                tag_output = pos_tag(word_tokenize(text))
                temp = []
                for idx, (word, tag) in enumerate(tag_output):
                    if word != "you":
                        temp.append(word)
                    else:
                        if idx + 1 < len(tag_output) and tag_output[
                                idx + 1][1].startswith('VB'):
                            temp.append("I")
                        else:
                            temp.append("me")
                text = " ".join(temp)
            new_fn2text[fn].add(text)
    return new_fn2text
    
    # *
def gen_paraphrase_for_text(text, paraphrase_dict):
    """ This function replace several words/phrases in a sentence at once
    for generating paraphrases.
    :param text: Text as string
    :type: String
    :param paraphrase_dict: Words and their equivalent paraphrases
    :type: defaultdict(set)
    :returns: Set of paraphrases"""
    paraphrases = set()
    tokens = word_tokenize(text)
    replaced = []
    replacement = []
    token_idx = 0
    while token_idx < len(tokens):
        unigram = tokens[token_idx]
        if token_idx < len(tokens) - 1:
            bigram = tokens[token_idx] + " " + tokens[token_idx]
        else:
            bigram = None

        if bigram and bigram in paraphrase_dict:
            replaced.append(bigram)
            replacement.append(paraphrase_dict[bigram])
            token_idx += 1
        elif unigram in paraphrase_dict:
            replaced.append(unigram)
            replacement.append(paraphrase_dict[unigram])
        token_idx += 1

    # generate token possibilities
    num_paraphrases = min([len(replaced)] +
                          [len(item) for item in replacement])
    for item_idx in range(len(replacement)):
        token_para = random.sample(replacement[item_idx], num_paraphrases)
        replacement[item_idx] = token_para

    # generate paraphrases
    for paraphrase_idx in range(num_paraphrases):
        new_text = text
        for token_replaced, token_para in zip(replaced, replacement):
            new_text = new_text.replace(token_replaced,
                                        token_para[paraphrase_idx])
        paraphrases.add(new_text)
    if len(paraphrases) < 1:
        paraphrases.add(text)
    return paraphrases
    
def gen_paraphrase(fn2text, paraphrase_dict):
    """Paraphrases the description of a function.
    :param fn2text: Function to descriptions mapping
    :type: Dict[str][List]
    :param paraphrase_dict: Words and their equivalent paraphrases
    :type: defaultdict(set)
    :returns: Function and their paraphrases
    :rtype: defaultdict(set)
    """
    fn2paraphrases = dict()
    for fn, desc in fn2text.items():
        for text in desc:
            paraphrases = gen_paraphrase_for_text(text, paraphrase_dict)
            if len(paraphrases):
                fn2paraphrases[fn] = paraphrases
    return fn2paraphrases


class UserSimulator:
    """Simulates a user that answers the agents follow up question to specific subtasks."""
    def __init__(self, dyn_answer):
        self._user_answer = None
        self.dyn_answer = dyn_answer
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore")  # sklearn warns about outdated label encoders
            self.paraphrase_dict = pk.load(
                open("data/equivalent_paraphrase_dict_clean.pkl", "rb"))
            self.data = pk.load(open("data/data_with_noisy_user_ans.pkl",
                                     "rb"),
                                encoding="latin1")
        tf2desc, af2desc = extract_fn_user_descriptions(self.data)
        self.tf_answers = gen_paraphrase(clean_function_text(tf2desc),
                                         self.paraphrase_dict)
        self.af_answers = gen_paraphrase(clean_function_text(af2desc),
                                         self.paraphrase_dict)

    def ask_user_answer(self, recipe_description, ground_truth_names,
                        subtask_index):
        """Returns an answer specific to the subtask.
        :param recipe_description: Description of the if-Then recipe
        :type: List[str]
        :param ground_truth_names: Names of the true Trigger/Action Channel/Function
        :type: List[str]
        :param subtask_index: current subtask index
        :type: List[int]
        :param dyn_answer: If the method can't find an answer in the answer_pool,
        if True paraphrase the recipe description,
        if False return the original description
        :returns: Simulated answer
        :rtype: List[str]"""
        
        self._user_answer = recipe_description
        if len(recipe_description) == 0:
            return self._user_answer
        if subtask_index == 0:  # Trigger Function
            self._user_answer = word_tokenize(ground_truth_names[0].strip())
        elif subtask_index == 2:  # Action Function
            self._user_answer = word_tokenize(ground_truth_names[2].strip())
        elif subtask_index == 1:  # Trigger Channel
            item = "%s.%s" % (ground_truth_names[0].lower().strip(),
                              ground_truth_names[1].lower().strip())
            if item in self.tf_answers:
                self._user_answer = word_tokenize(
                    random.choice(tuple(self.tf_answers[item])))
            elif self.dyn_answer:
                tf_desc, af_desc = extract_fn(recipe_description)
                self._user_answer = random.choice(
                    tuple(
                        gen_paraphrase_for_text(tf_desc,
                                                self.paraphrase_dict)))
        elif subtask_index == 3:  # Action Channel
            item = "%s.%s" % (ground_truth_names[2].lower().strip(),
                              ground_truth_names[3].lower().strip())
            if item in self.af_answers:
                self._user_answer = word_tokenize(
                    random.choice(tuple(self.af_answers[item])))
            elif self.dyn_answer:
                tf_desc, af_desc = extract_fn(recipe_description)
                self._user_answer = random.choice(
                    tuple(
                        gen_paraphrase_for_text(af_desc,
                                                self.paraphrase_dict)))

        if isinstance(self._user_answer, str):
            self._user_answer = word_tokenize(self._user_answer)

        self._user_answer = [word.lower() for word in self._user_answer]
        return self._user_answer

    def get_current_user_answer(self):
        return self._user_answer
