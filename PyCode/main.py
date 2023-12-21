from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from dotenv import load_dotenv
import argparse


# Load environment variables
load_dotenv()

# Create the CLI parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--task", help="Task to perform", type=str, default="return a list of numbers")
parser.add_argument("--language", help="Language to use", type=str, default="python")
args = parser.parse_args()

# Create the LLM object
llm = OpenAI()

# Create the prompt template
code_prompt = PromptTemplate(
    input_variables=["task", "language"],
    template="Write a very short {language} function that will {task}."
)

test_prompt = PromptTemplate(
    input_variables=["code", "language"],
    template="Generate a test for the following {language} function:\n\n{code}"
)

# Create the chain
code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt,
    output_key="code",
)

test_chain = LLMChain(
    llm=llm,
    prompt=test_prompt,
    output_key="test",
)

# Create the sequential chain
chain = SequentialChain(
    chains=[code_chain, test_chain],
    input_variables=["task", "language"],
    output_variables=["code", "test"],
)

# Generate the code
result = chain({
    "language": args.language,
    "task": args.task,
})

# Print the result
print(">>>>>>>>>>>>> GENERATED CODE <<<<<<<<<<<<<<")
print(result["code"],'\n')

print(">>>>>>>>>>>>> GENERATED TEST <<<<<<<<<<<<<<")
print(result["test"])

