import os

import dspy

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
lm = dspy.LM("openai/gpt-4.1", OPENAI_API_KEY)
dspy.configure(lm=lm)

# Calling the LM directly
lm("Say this is a test!", temperature=0.7)

