import argparse
import os
import re
import string

import joblib
import torch
from IPython import embed
from keybert import KeyBERT
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from cbr_analyser.consts import *

kw_model = KeyBERT()

parser = argparse.ArgumentParser(
    description='Functionalities related to prompt generation using gpt-j')
parser.add_argument('--task', help='the task that you want to do', type=str)
parser.add_argument(
    '--input_file', help='input_file that is probbaly a file containing the sentences with amr_objects', type=str)
parser.add_argument(
    '--output_file', help='output_file the the output probably containg the generated beliefs and arguments will be saved in', type=str)


args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


def get_belief_argument_from_sentence(sentence: str):

    sentence = sentence.replace('\n', '. ')

    filled_in_prompt = f"""
    I hear the rain falling outside my window therefore the sun is not shining
    belief: I hear the rain falling outside my window
    argument: Therefore the sun is not shining
    ###
    If Mom didn't turn off the air conditioner then clearly she must be too hot
    belief: Mom didn't turn off the air conditioner
    argument: then clearly she must be too hot
    ###
    The bigger a child's shoe size, the better the child's handwriting
    belief: The bigger a child's shoe size
    argument: The better the child's handwriting
    ###
    If Joe eats greasy food he will feel sick; Joe feels sick; Therefore Joe ate greasy food
    belief: If Joe eats greasy food he will feel sick; Joe feels sick
    argument: Therefore Joe ate greasy food
    ###
    Students should not be allowed to park in lots now reserved for faculty because those lots should be for faculty only
    belief: Students should not be allowed to park in lots now reserved for faculty
    argument: those lots should be for faculty only
    ###
    Senator Randall isn't lying when she says she cares about her constituents; she wouldn't lie to people she cares about
    belief: Senator Randall isn't lying when she says she cares about her constituents
    argument: she wouldn't lie to people she cares about
    ###
    If we ban Hummers because they are bad for the environment eventually the government will ban all cars, so we should not ban Hummers
    belief: If we ban Hummers because they are bad for the environment eventually the government will ban all cars
    argument: so we should not ban Hummers
    ###
    Mayor Blake wants to create more bicycle lanes in Lowell. Why is he forcing us to give up our cars and bike everywhere?
    belief: Mayor Blake wants to create more bicycle lanes in Lowell
    argument: Why is he forcing us to give up our cars and bike everywhere?
    ###
    Students never used to have cell phones so they don't need them now
    belief: Students never used to have cell phones
    argument: they don't need them now
    ###
    Nature gives people diseases and sickness therefore it is morally wrong to interfere with nature and treat sick people with medicine
    belief: Nature gives people diseases and sickness
    argument: it is morally wrong to interfere with nature and treat sick people with medicine
    ###
    Debate moderator: ""What will you do to fix the economic crisis?"" Candidate: ""It's important to focus on what started this crisis, to begin with, My opponent""
    belief: Debate moderator: ""what will you do to fix the economic crisis?""
    argument: Candidate: ""It's important to focus on what started this crisis, to begin with, My opponent""
    ###
    You should give me a promotion, I have a lot of debt and am behind on my rent
    belief: You should give me a promotion
    argument: I have a lot of debt and am behind on my rent
    ###
    Those who believe in behavior modification obviously want to try to control everyone by subjecting them to rewards and punishments
    belief: Those who believe in behavior modification
    argument: obviously want to try to control everyone by subjecting them to rewards and punishments
    ###
    A mother tells her children not to leave the yard because there might be wild animals in the woods
    belief: A mother tells her children not to leave the yard
    argument: there might be wild animals in the woods
    ###
    You oppose a senator's proposal to extend government-funded health care to poor minority children because that senator is a liberal Democrat
    belief: You oppose a senator's proposal to extend government-funded health care to poor minority children
    argument: that senator is a liberal Democrat
    ###
    This herbal supplement is made from a plant that grows in Zambia. It must be healthier than taking that medication which is full of chemicals I can't pronounce
    belief: This herbal supplement is made from a plant that grows in Zambia
    argument: It must be healthier than taking that medication which is full of chemicals I can't pronounce
    ###
    Since many people believe this then it must be true
    belief: Since many people believe this
    argument: then it must be true
    ###
    Mayor: During my previous term as mayor my staff and I spent a great deal of time focusing on our city's economy and unemployment reached an all time high as a result. So clearly you should support my re-election campaign if you are among those still looking for a job
    belief: Mayor: During my previous term as mayor my staff and I spent a great deal of time focusing on our city's economy and unemployment reached an all time high as a result
    argument: So clearly you should support my re-election campaign if you are among those still looking for a job
    ###
    very time Joe goes swimming he is wearing his Speedos. Something about wearing that Speedo must make him want to go swimming
    belief: very time Joe goes swimming he is wearing his Speedos
    argument: Something about wearing that Speedo must make him want to go swimming
    ###
    Every time I go to sleep, the sun goes down. Therefore, my sleeping causes the sun to set
    belief: Every time I go to sleep, the sun goes down.
    argument: therefore, my sleeping causes the sun to set
    ###
    Child: This fish tastes funny. I don't want to eat this. Parent: There are children starving in Africa. Eat your dinner.
    belief: Child: This fish tastes funny. I don't want to eat this.
    argument: Parent: There are children starving in Africa. Eat your dinner.
    ###
    A mother is telling her daughter that she went over her data for the month, and the daughter begins telling her mother about getting an A on a math test.
    belief: A mother is telling her daughter that she went over her data for the month
    argument: and the daughter begins telling her mother about getting an A on a math test
    ###
    Ms. Baker assigned me a lot of homework because she's a witch!
    belief: Ms. Baker assigned me a lot of homework
    argument: she's a witch!
    ###
    Three congressional representatives have had affairs. Members of Congress are adulterers.
    belief: Three congressional representatives have had affairs.
    argument: Members of Congress are adulterers.
    ###
    I lost my phone in the living room, so it will always be in the living room when it is lost.
    belief: I lost my phone in the living room
    argument: so it will always be in the living room when it is lost.
    ###
    {sentence}
    belief: """

    inputs = tokenizer(filled_in_prompt, return_tensors='pt').to(device)

    greedy_output = model.generate(
        **inputs, max_new_tokens=200, top_p=0.9, temperature=0.8)

    outputs = tokenizer.decode(greedy_output[0], skip_special_tokens=True)

    belief_locs = [(m.start(), m.end())
                   for m in re.finditer('belief:', outputs)]
    arguments_locs = [(m.start(), m.end())
                      for m in re.finditer('argument:', outputs)]
    hashtag_locs = [(m.start(), m.end()) for m in re.finditer('###', outputs)]

    extracted_belief = outputs[belief_locs[25][0]: arguments_locs[25][0]][7:]
    extracted_argument = outputs[arguments_locs[25]
                                 [0]: hashtag_locs[25][0]][9:]

    return extracted_belief.strip(), extracted_argument.strip()


def augment_sentences_with_amr_objects_with_belief_arguments(input_file_path: str or Path, output_file_path: str or Path) -> None:
    if os.path.exists(output_file_path):
        return
    sentences_with_amr_objects = joblib.load(input_file_path)

    for obj in tqdm(sentences_with_amr_objects, leave=False):
        try:
            belief_argument = get_belief_argument_from_sentence(obj[0])
            obj[1].add_belief_argument(belief_argument)
        except Exception as e:
            obj[1].add_belief_argument(None)
            continue

    joblib.dump(
        sentences_with_amr_objects,
        output_file_path
    )


def is_empty(input):
    return input.isspace() or input == ""


if __name__ == "__main__":
    if args.task == "generate":
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        print('loading the model!')

        model = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/gpt-j-6B").to(device)

        print('loaded the model!')

        augment_sentences_with_amr_objects_with_belief_arguments(
            input_file_path=args.input_file,
            output_file_path=args.output_file
        )

    elif args.task == "output_explagraph":
        sentences_with_amr_objects = joblib.load(args.input_file)
        with open(args.output_file, 'w') as f:
            for index, obj in enumerate(sentences_with_amr_objects):
                belief_arguments = obj[1].belief_argument
                if belief_arguments is not None and not is_empty(belief_arguments[0]) and not is_empty(belief_arguments[1]) and \
                    len(belief_arguments[0].split()) < 50 and len(belief_arguments[1].split()) < 50 and \
                        len(belief_arguments[0].split()) > 5 and len(belief_arguments[1].split()) > 5:
                    belief_arguments = (belief_arguments[0].replace(
                        '\n', '; '), belief_arguments[1].replace('\n', '; '))
                    if index == len(sentences_with_amr_objects) - 1:
                        f.write(
                            f"{belief_arguments[0]}\t{belief_arguments[1]}\tsupport\t(marriage; capable of; deceiving)"
                        )
                    else:
                        f.write(
                            f"{belief_arguments[0]}\t{belief_arguments[1]}\tsupport\t(marriage; capable of; deceiving)\n"
                        )

    elif args.task == "external_node_generation":
        sentences_with_amr_objects = joblib.load(args.input_file)
        with open(args.output_file, 'w') as f:
            for index, obj in enumerate(sentences_with_amr_objects):
                belief_arguments = obj[1].belief_argument
                if belief_arguments is not None and not is_empty(belief_arguments[0]) and not is_empty(belief_arguments[1]) and \
                    len(belief_arguments[0].split()) < 50 and len(belief_arguments[1].split()) < 50 and \
                        len(belief_arguments[0].split()) > 5 and len(belief_arguments[1].split()) > 5:
                    keywords = kw_model.extract_keywords(obj[0], keyphrase_ngram_range=(
                        1, 2), use_mmr=True, diversity=0.7, stop_words='english', top_n=5)
                    keywords = [keyword[0] for keyword in keywords]
                    if index == len(sentences_with_amr_objects) - 1:
                        f.write(
                            f"{', '.join(keywords)}"
                        )
                    else:
                        f.write(
                            f"{', '.join(keywords)}\n"
                        )

    else:
        raise NotImplementedError()
