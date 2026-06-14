# Real-World Case Study

This directory contains artifacts for a real-world case study evaluating the practicality of AREA on representative LLM-based applications.

## Evaluated Applications

We evaluate three representative agents reflecting common real-world deployment patterns:

- `persona-oriented_ReAct_agent/`  
  A persona-oriented ReAct agent with a short system prompt focusing on style consistency.

- `rule-intensive_ReAct_agent/`  
  A rule-intensive ReAct agent with a long system prompt encoding complex rules.

- `retrieval-augmented_agent/`  
  A retrieval-augmented agent integrating an external knowledge base.

## Defense Settings

For each agent, we evaluate three defense configurations:

- **without_defense** (`without_defense.mp4`)  
  The agent is deployed without any prompt-leakage defense.

- **with_promptObfuscation** (`with_promptObfuscation.mp4`)  
  The agent is protected using the PromptObfuscation defense.

- **with_AREA** (`with_AREA.mp4`)  
  The agent is protected using AREA (Ours).

## Video Demonstrations

Each video demo corresponds to one agent under one defense setting and consists of **six interaction rounds**:

- **Rounds 1–5:** benign queries that reflect normal usage.
- **Round 6:** a single adversarial query designed to induce prompt leakage.

The adversarial queries are selected from real-world, carefully crafted attack prompts collected from an open-source security community:
https://github.com/LouisShark/chatgpt_system_prompt/tree/066b8f9a6db9dce64f2d5d36d91f9e87d8ca2530

## Summary of Observations

From the video demonstrations, we observe the following:

1. **Without defense**, all agents are vulnerable to adversarial queries and leak system prompt content.
2. **PromptObfuscation** effectively prevents prompt leakage across all three agents; however, it preserves usability only for the persona-oriented agent, while significantly degrading usability for the rule-intensive and retrieval-augmented agents.
3. **AREA** consistently achieves strong leakage resistance while preserving normal functionality across all three agents.

These observations qualitatively complement the quantitative evaluation results reported in the paper.
