import os

import dspy

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
dspy_lm = dspy.LM(model="openai/gpt-4.1") # Support by LiteLLM
dspy.configure(lm=dspy_lm)

# Calling the LM directly
dspy_lm("Say this is a test!", temperature=0.7)  # => ['This is a test!']
dspy_lm(
    messages=[{"role": "user", "content": "Say this is a test!"}]
)  # => ['This is a test!']

# Using The LM with DSPy modules

# Define a module (ChainOfThought) and assign it a signature (return an answer, given a question).
qa = dspy.ChainOfThought("question -> answer")

# Run with the default LM configured with `dspy.configure` above.
# response = qa(question="How many floors are in the castle David Gregory inherited?")
# print(response.answer)

# What's Signature in DSPy? -> It is a declarative dpecification of i/o behavior of DSPy module. Tell the LM "What needs to do" !
"""Inine DSPy Signatures
1. Question Answering : "question: type -> answer: type
2. Sentiment Classification: "sentence -> sentiment:bool

Multiple Inline:
1. RAG Question Answering: context: list[str], question: str -> answer: str
2. Multi Choices QA and Reasoning: question, choices: list[str] -> reasoning: str, selection: int

More:
"document -> summary", "text -> gist", or "long_context -> tldr"
"""

# toxicity = dspy.Predict(
#     dspy.Signature(
#         "comment -> toxic: bool",
#         instructions="Mark as 'toxic' if the comment includes insults, harassment, or sarcastic derogatory remarks.",
#     )
# )
