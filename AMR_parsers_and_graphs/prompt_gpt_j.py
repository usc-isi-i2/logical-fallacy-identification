from ctypes import Union
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")


def get_belief_argument_from_sentence(sentence: str):

    sentence = re.sub(r"[\.,]", " ", sentence)

    filled_in_prompt = f"""
    I hear the rain falling outside my window; therefore the sun is not shining 
    belief: I hear the rain falling outside my window
    argument: Therefore the sun is not shining
    ###
    If Mom didn't turn off the air conditioner then clearly she must be too hot
    belief: Mom didn't turn off the air conditioner
    argument: then clearly she must be too hot
    ###
    The bigger a child's shoe size the better the child's handwriting
    belief: The bigger a child's shoe size
    argument: The better the child's handwriting
    ###
    If Joe eats greasy food he will feel sick Joe feels sick Therefore Joe ate greasy food
    belief: If Joe eats greasy food he will feel sick Joe feels sick
    argument: Therefore Joe ate greasy food
    ###
    Students should not be allowed to park in lots now reserved for faculty because those lots should be for faculty only
    belief: Students should not be allowed to park in lots now reserved for faculty
    argument: those lots should be for faculty only
    ###
    Senator Randall isn't lying when she says she cares about her constituents she wouldn't lie to people she cares about
    belief: Senator Randall isn't lying when she says she cares about her constituents
    argument: she wouldn't lie to people she cares about
    ###
    If we ban Hummers because they are bad for the environment eventually the government will ban all cars so we should not ban Hummers
    belief: If we ban Hummers because they are bad for the environment eventually the government will ban all cars
    argument: so we should not ban Hummers
    ###
    Mayor Blake wants to create more bicycle lanes in Lowell Why is he forcing us to give up our cars and bike everywhere?
    belief: Mayor Blake wants to create more bicycle lanes in Lowell
    argument: Why is he forcing us to give up our cars and bike everywhere?
    ###
    Students never used to have cell phones so they don't need them now
    belief: Students never used to have cell phones
    argument: they don't need them now
    ###
    Nature gives people diseases and sickness; therefore it is morally wrong to interfere with nature and treat sick people with medicine
    belief: Nature gives people diseases and sickness;
    argument: it is morally wrong to interfere with nature and treat sick people with medicine
    ###
    {sentence}
    belief:"""


    inputs = tokenizer(filled_in_prompt, return_tensors = 'pt')

    greedy_output = model.generate(**inputs, max_new_tokens = 150, top_p = 0.9, temperature = 0.8)

    outputs = tokenizer.decode(greedy_output[0], skip_special_tokens=True)

    belief_locs = [(m.start(), m.end()) for m in re.finditer('belief:', outputs)]
    arguments_locs = [(m.start(), m.end()) for m in re.finditer('argument:', outputs)]
    hashtag_locs = [(m.start(), m.end()) for m in re.finditer('###', outputs)]

    extracted_belief = outputs[belief_locs[10][0]: arguments_locs[10][0]][7:]
    extracted_argument = outputs[arguments_locs[10][0]: hashtag_locs[10][0]][9:]

    return extracted_belief, extracted_argument


if __name__ == "__main__":
    pass
