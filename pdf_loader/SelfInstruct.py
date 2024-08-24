from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector, LengthBasedExampleSelector
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings

from langchain.prompts.example_selector.base import BaseExampleSelector
from typing import Dict, List
import numpy as np
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import os, openai

# os.environ['OPENAI_API_BASE'] = "https://api.emabc.xyz/v1"
api_key = "sk-BFHheN895eZkdU5n54A02b0b581540618f30B748C51e18E7"
openai.api_key = api_key

SEEK_TASKS = [
    "Try not to repeat the verb for each instruction to maximize diversity.",
    "The language used for the instruction also should be diverse. For example, you should combine questions with imperative instrucitons.",
    "The type of instructions should be diverse. The list should include diverse types of tasks like open-ended generation, classification, editing, etc.",
    "A GPT language model should be able to complete the instruction. For example, do not ask the assistant to create any visual or audio output. For another example, do not ask the assistant to wake you up at 5pm or set a reminder because it cannot perform any action.",
    "The instructions should be in English.",
    "The instructions should be 1 to 2 sentences long. Either an imperative sentence or a question is permitted.",
    "You should generate an appropriate input to the instruction. The input field should contain a specific example provided for the instruction. It should involve realistic data and should not contain simple placeholders. The input should provide substantial content to make the instruction challenging but should ideally not exceed 100 words.",
    "Not all instructions require input. For example, when a instruction asks about some general information, 'what is the highest peak in the world', it is not necssary to provide a specific context. In this case, we simply put '<noinput>' in the input field.",
    "The output should be an appropriate response to the instruction and the input. Make sure the output is less than 100 words.",
    "Make sure the output is gramatically correct with punctuation if needed.",

]

classification_examples=[
    {
        "task": "Given my personality and the job, tell me if I would be suitable.",
        "label": "Yes"
    },
    {
        "task": "Give me an example of a time when you had to use your sense of humor.",
        "label": "No"
    },
    {
        "task": "Replace the placeholders in the given text with appropriate named entities.",
        "label": "No"
    },
    {
        "task": "Fact checking - tell me if the statement is true, false, or unknown, based on your knowledge and common sense.",
        "label": "Yes"
    },
    {
        "task": "Return the SSN number for the person.",
        "label": "No"
    },
    {
        "task": "Detect if the Reddit thread contains hate speech.",
        "label": "Yes"
    },
    {
        "task": "Analyze the sentences below to identify biases.",
        "label": "Yes"
    }
]


none_classify_examples=[
    {
        "task": "Which exercises are best for reducing belly fat at home?",
        "input": "",
        "output": '''- Lying Leg Raises
- Leg In And Out
- Plank
- Side Plank
- Sit-ups'''
    },
    {
        "task": "Extract all the country names in the paragraph, list them separated by commas.",
        "input": "Paragraph: Dr. No is the sixth novel by the English author Ian Fleming to feature his British Secret Service agent James Bond. Written at Fleming's Goldeneye estate in Jamaica, it was first published in the United Kingdom by Jonathan Cape in 1958. In the novel Bond looks into the disappearance in Jamaica of two fellow MI6 operatives who had been investigating Doctor No. Bond travels to No's Caribbean island and meets Honeychile Rider, who is there to collect shells. They are captured and taken to a luxurious facility carved into a mountain. The character of Doctor No, the son of a German missionary and a Chinese woman, was influenced by Sax Rohmer's Fu Manchu stories. Dr. No was the first of Fleming's novels to face widespread negative reviews in Britain, but it was received more favourably in the United States.",
        "output": "English, British, Jamaica, the United Kingdom, German, Chinese, Britain, the United States."
    },
    {
        "task": "Suggest a better and more professional rephrasing of the following sentence.",
        "input": "Sentence: This house is surprisingly not constructed very well, and you probably need more money to fix it after you buy it. If you ask me, I would suggest you to consider other candidates.",
        "output": "This house does not seem to be constructed well, so you may need to spend more money to fix it after you purchase it. I would suggest that you look at other properties."
    },
    {
        "task": "Read the following paragraph and answer a math question about the paragraph. You need to write out the calculation for getting the final answer.",
        "input": '''Paragraph: Gun violence in the United States results in tens of thousands of deaths and injuries annually, and was the leading cause of death for children 19 and younger in 2020. In 2018, the most recent year for which data are available as of 2021, the Centers for Disease Control and Prevention's (CDC) National Center for Health Statistics reports 38,390 deaths by firearm, of which 24,432 were by suicide. The rate of firearm deaths per 100,000 people rose from 10.3 per 100,000 in 1999 to 12 per 100,000 in 2017, with 109 people dying per day or about 14,542 homicides in total, being 11.9 per 100,000 in 2018. In 2010, there were 19,392 firearm-related suicides, and 11,078 firearm-related homicides in the U.S. In 2010, 358 murders were reported involving a rifle while 6,009 were reported involving a handgun; another 1,939 were reported with an unspecified type of firearm. In 2011, a total of 478,400 fatal and nonfatal violent crimes were committed with a firearm.
Question: How many more firearm-related deaths were there in 2018 compared to 2010?''',
        "output": '''38390 - (19392 + 11078) = 38390 - 30470 = 7920. 
So, in 2018, there were 7920 more deaths by firearm than in 2010.'''
    },
    {
        "task": "Solve the equation and find the value of X. Show your steps.",
        "input": "Equation: 10X + 5 = 10",
        "output": "Output: 10X = 5,  X = 0.5"
    },
    {
        "task": "Select the oldest person from the given list.",
        "input": "List: George Washington, Confucius, Michael Jordan, Michelangelo",
        "output": "Confucious"
    },
]

classify_examples=[
    {
        "task": "Classify the sentiment of the sentence into positive, negative, or mixed.",
        "input": "mixed",
        "output": "I enjoy the flavor of the restaurant but their service is too slow."
    },
    {
        "task": '''Given a dialogue, classify whether the user is satisfied with the service. You should respond with "Satisfied" or "Unsatisfied".''',
        "input": "Satisfied",
        "output": '''- Agent: Thank you for your feedback. We will work to improve our service in the future.
- Customer: I am happy with the service you provided. Thank you for your help.'''
    },
    {
        "task": "Given some political opinions, classify whether the person belongs to Democrats or Republicans.",
        "input": "Democrats",
        "output": "I believe that everyone should have access to quality healthcare regardless of their income level."
    },
    {
        "task": "Tell me if the following email is a promotion email or not.",
        "input": "Promotion",
        "output": "Check out our amazing new sale! We've got discounts on all of your favorite products."
    },
    {
        "task": "Detect if the Reddit thread contains hate speech.",
        "input": "Hate Speech",
        "output": "All people of color are stupid and should not be allowed to vote."
    },
    
]
class CustomExampleSelector(BaseExampleSelector):
    
    def __init__(self, examples: List[Dict[str, str]], top_k):
        self.examples = examples
        self.k = top_k
    
    def add_example(self, example: Dict[str, str]) -> None:
        """Add new example to store for a key."""
        self.examples.append(example)

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Select which examples to use based on the inputs."""
        selected_examples = np.random.choice(self.examples, size=self.k, replace=False)
        return [{"id":i+1, "example": selected_examples[i]} for i in range(len(selected_examples))]


class SelfInstruct:
    def __init__(self) -> None:
        self.llm = ChatOpenAI(model_name='gpt-3.5-turbo', openai_api_key="sk-BFHheN895eZkdU5n54A02b0b581540618f30B748C51e18E7", max_retries=20)
    
    def init_selector(self, seed_tasks):
        self.seed_task_example_selector = CustomExampleSelector(seed_tasks, 10)
        # self.classfication_example_seleector = CustomExampleSelector(classfication_examples, 10)
    
    def init_prompt_template(self):
        example_prompt = PromptTemplate(
            input_variables=["id", "example"],
            template="Task{id}: {example}",
        )
        # 生成新指令的prompt
        self.instruction_generation_prompt = FewShotPromptTemplate(
            example_selector=self.seed_task_example_selector,
            # examples=examples, 
            example_prompt=example_prompt, 
            prefix="Come up with a series of tasks:",
            suffix="Task 11:{inp}", 
            input_variables=["inp"]
        )
        classification_example_prompt = PromptTemplate(
            input_variables=["task", "label"],
            template="Task: {task}\nIs it classification?{label}",
        )
        # 判断指令是否为分类任务
        self.classification_identify_prompt = FewShotPromptTemplate(
            examples=classification_examples, 
            example_prompt=classification_example_prompt, 
            prefix="Come up with a series of tasks:",
            suffix="Task :{input}\nIs it classification?", 
            input_variables=["input"]
        )
        #非分类任务实例生成
        none_classify_example_prompt = PromptTemplate(
            input_variables=["task", "input", "output"],
            template="Task: {task}\n{input}\nOutput:{output}",
        )
        self.none_classify_instance_generation_prompt = FewShotPromptTemplate(
            examples=none_classify_examples, 
            example_prompt=none_classify_example_prompt, 
            prefix="Come up with examples for the following tasks. Try to generate multiple examples when possible. If the task doesn't require additional input, you can generate the output directly.",
            suffix="Task :{task}\ninput: \nOutput:", 
            input_variables=["task"]
        )
        classify_example_prompt = PromptTemplate(
            input_variables=["task", "input", "output"],
            template="Task: {task}\nClass label:{output}\nInput{input}",
        )
        # 分类任务实例生成
        self.classify_instance_generation_prompt = FewShotPromptTemplate(
            examples=classify_examples, 
            example_prompt=classify_example_prompt, 
            prefix="Given the classification task definition and the class labels, generate an input that corresponds to each of the class labels. If the task doesn't require input, just generate possible class labels.",
            suffix="Task :{task}\nOutput: \nInput:", 
            input_variables=["task"]
        )
    
    # 生成一个新的指令
    def generate_new_instruction(self):
        # generate_prompt = self.instruction_generation_prompt.format()
        llm_chain = LLMChain(llm=self.llm, prompt=self.instruction_generation_prompt)
        new_instruction = llm_chain.run({"inp": ""})
        print(new_instruction)
        return new_instruction
    
    # 判断指令是否为分类任务
    def classification_identify(self, task):
        llm_chain = LLMChain(llm=self.llm, prompt=self.classification_identify_prompt)
        while True:
            is_classify = llm_chain.run({"input": task})
            if is_classify=="Yes": return True
            elif is_classify=="No": return False
    
    # 生成新的指令实例
    def classify_instance_generation(self, task, is_classify):
        if is_classify:
            llm_chain = LLMChain(llm=self.llm, prompt=self.classify_instance_generation_prompt)
            instance = llm_chain.run({"task": task})
        else:
            llm_chain = LLMChain(llm=self.llm, prompt=self.none_classify_instance_generation_prompt)
            instance = llm_chain.run({"task": task})
        return instance
    
if __name__=="__main__":
    self_instruct = SelfInstruct()
    self_instruct.init_selector(seed_tasks=SEEK_TASKS)
    self_instruct.init_prompt_template()
    # self_instruct.instruction_generation_prompt.format(inp="")
    new_instruction = self_instruct.generate_new_instruction()
    is_classify = self_instruct.classification_identify(task=new_instruction)
    print(f"is_classify:{is_classify}")
    new_instance = self_instruct.classify_instance_generation(task=new_instruction, is_classify=is_classify)
    print(f"new_instance:{new_instance}")