# LLM Evaluation Pipeline

## Overview
This repository contains an LLM evaluation pipeline designed to assess AI chatbot responses against a ground-truth context using an **LLM-as-Judge** approach.

The pipeline evaluates:
- Response relevance
- Factual accuracy
- Hallucination detection
- End-to-end latency
- Token-based cost estimation

It supports both **single-turn evaluation** and **batch evaluation** across all AI responses in a conversation.

## Local Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/iamAni9/LLM-Evaluation-Pipeline.git
cd <repo-name>
```
### 2. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Prepare input data
Ensure the following files exist in the project root:
#### 1. A timestamped conversation between a user and an AI system
```bash
#check this one for refernece (format should remain same)- 
sample-chat-conversation-01.json
```
#### 2. Context vectors retrieved during the original AI response generation
```bash
#check this one for refernece (format should remain same)- 
sample_context_vectors-01.json
```
### 5. Configure API key
```bash
Refer the .env.example file
```
### 6. Run the evaluation
```bash
python main.py
```
The evaluation results will be written to: **evaluation.json**




## Architecture Overview

```md
## Architecture Overview

The evaluation pipeline follows a deterministic, modular flow:

Conversation + Context  
→ Context Filtering  
→ Latency Calculation  
→ LLM-as-Judge Evaluation  
→ Token & Cost Estimation  
→ Structured JSON Output
```

## Core Components

### 1. Context Filtering
Only context vectors actually used during AI inference are included in evaluation.

This prevents:
- Prompt inflation
- False hallucination flags
- Unnecessary token usage

---

### 2. Latency Calculation
Latency is computed using timestamps from the conversation log:

- User message creation time
- AI response creation time

This captures **end-to-end system latency**, not just model inference time.

---

### 3. LLM-as-Judge
A separate LLM evaluates the AI response using:
- A strict system prompt
- A fixed JSON output schema
- Explicit rules for factual accuracy and hallucinations

This enables consistent, model-agnostic evaluation.

---

### 4. Token & Cost Estimation
Token usage is computed explicitly using `tiktoken` (`cl100k_base`).

Costs are estimated for:
- The original AI response
- The evaluation LLM call

This ensures full cost transparency.



### 5. Flexible Evaluation Modes
The pipeline supports:
- Evaluating a single AI turn by ID
- Evaluating all AI turns in a conversation

This allows both debugging and batch audits.

## PART 3 — Design Decisions
### Q: Why did you decide to build the solution this way, and not some other way?
``` bash
The solution is intentionally designed using a "class-based paradigm" to ensure modularity, extensibility, and clear separation of concerns, that is the one thing.

My key idea was to make the "evaluation prompt" configurable. By allowing the system prompt to be injected during initialization, the pipeline can easily adapt to:
- Different evaluation criteria
- Different scoring rubrics
- Different domains (e.g., finance, healthcare, customer support)

This avoids rewriting core logic when evaluation requirements change and enables rapid iteration on evaluation behavior.

I have used the "LLM-as-Judge" approach over rule-based or heuristic systems because it scales better to nuanced language understanding tasks such as relevance, factual grounding, and hallucination detection—areas where static rules typically fail, even we can allow llm to use these rules by injecting the custom prompt, proving flexibility over different approaches.

Also, there is flexibility to use any llm model for evaluation that allow to evaluate the same response over more across multiple judges. 
```
### Q: If we run your script at scale (millions of daily conversations), how are latency and costs minimized?
```bash
The pipeline is designed with scale as a first-class concern.
Not every conversation or turn must be evaluated. The pipeline supports:
- Evaluating a single AI turn by ID
- Evaluating all AI turns when needed

This dramatically reduces unnecessary evaluation calls at scale.

Also only "context vectors that were actually used during inference" are included in the evaluation prompt. This prevents:
- Prompt bloat
- Inflated token usage
- Increased latency and cost

Reducing prompt size has a direct impact on both response time and billing.

Token usage is computed explicitly using a tokenizer initialized once per pipeline instance. Costs are estimated for:
- The original AI response
- The evaluation LLM call

This makes cost transparent and enables:
- Budget monitoring
- Cost-aware throttling
- Data-driven optimization decisions

The evaluation logic is stateless and self-contained, making it easy to:
- Run evaluations asynchronously
- Scale horizontally using worker pools
- Integrate with message queues or batch processors

This allows millions of evaluations per day without architectural changes. Thus reduces long-term maintenance costs and enables safe scaling.
```
