from ctypes import Union
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from IPython import embed
import joblib
from consts import *
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")


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
    belief: those lots should be for faculty only
    argument: Students should not be allowed to park in lots now reserved for faculty
    ###
    Senator Randall isn't lying when she says she cares about her constituents; she wouldn't lie to people she cares about
    belief: Senator Randall isn't lying when she says she cares about her constituents
    argument: she wouldn't lie to people she cares about
    ###
    If we ban Hummers because they are bad for the environment eventually the government will ban all cars so we should not ban Hummers
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
    belief: ""what will you do to fix the economic crisis?""
    argument: ""It's important to focus on what started this crisis, to begin with, My opponent""
    ###
    You should give me a promotion, I have a lot of debt and am behind on my rent
    belief: I have a lot of debt and am behind on my rent
    argument: You should give me a promotion
    ###
    Those who believe in behavior modification obviously want to try to control everyone by subjecting them to rewards and punishments
    belief: Those who believe in behavior modification
    argument: obviously want to try to control everyone by subjecting them to rewards and punishments
    ###
    A mother tells her children not to leave the yard because there might be wild animals in the woods
    belief: there might be wild animals in the woods
    argument: A mother tells her children not to leave the yard
    ###
    You oppose a senator's proposal to extend government-funded health care to poor minority children because that senator is a liberal Democrat
    belief: that senator is a liberal Democrat
    argument: You oppose a senator's proposal to extend government-funded health care to poor minority children
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
    belief: This fish tastes funny. I don't want to eat this.
    argument: Parent: There are children starving in Africa. Eat your dinner.
    ###
    A mother is telling her daughter that she went over her data for the month, and the daughter begins telling her mother about getting an A on a math test.
    belief: A mother is telling her daughter that she went over her data for the month
    argument: and the daughter begins telling her mother about getting an A on a math test
    ###
    {sentence}
    belief: """


    inputs = tokenizer(filled_in_prompt, return_tensors = 'pt')

    greedy_output = model.generate(**inputs, max_new_tokens = 150, top_p = 0.9, temperature = 0.8)

    outputs = tokenizer.decode(greedy_output[0], skip_special_tokens=True)

    belief_locs = [(m.start(), m.end()) for m in re.finditer('belief:', outputs)]
    arguments_locs = [(m.start(), m.end()) for m in re.finditer('argument:', outputs)]
    hashtag_locs = [(m.start(), m.end()) for m in re.finditer('###', outputs)]

    embed()

    extracted_belief = outputs[belief_locs[22][0]: arguments_locs[22][0]][7:]
    extracted_argument = outputs[arguments_locs[22][0]: hashtag_locs[22][0]][9:]

    return extracted_belief, extracted_argument


if __name__ == "__main__":
    sentences_with_amr_objects = joblib.load(PATH_TO_MASKED_SENTENCES_AMRS_DEV)
    sent = sentences_with_amr_objects[111][0]
    get_belief_argument_from_sentence(sent)

    
