# Generative AI with Large Language Models — Hands-on Labs

A collection of hands-on Jupyter notebooks completed as part of the **Generative AI with Large Language Models** course by [DeepLearning.AI](https://www.deeplearning.ai/).

This repository covers practical exercises involving Large Language Models (LLMs), prompt engineering, model fine-tuning, Parameter-Efficient Fine-Tuning (PEFT), Low-Rank Adaptation (LoRA), model evaluation, and Reinforcement Learning from Human Feedback (RLHF) concepts.

The labs use the FLAN-T5 model from Hugging Face and the DialogSum dataset to explore dialogue summarisation and model alignment.

---

## Repository Overview

The repository is organised into three labs:

| Lab | Topic | Key Concepts |
|---|---|---|
| 01 | Dialogue Summarisation and Prompt Engineering | Zero-shot, one-shot, few-shot prompting, generation configuration |
| 02 | Fine-Tuning a Generative AI Model | Full fine-tuning, PEFT, LoRA, Hugging Face Trainer, ROUGE |
| 03 | Detoxifying Summaries using Reinforcement Learning | Reward models, PPO, KL divergence, toxicity evaluation |

## 1. Dialogue Summarisation and Prompt Engineering

**Notebook:** `Lab_1_summarize_dialogue.ipynb`

### Objective

Explore how prompt design influences the output of a pretrained generative language model when performing dialogue summarisation.

The experiments use Google's FLAN-T5 model and the DialogSum dataset.

### Implementation

- Load the DialogSum dataset from Hugging Face.
- Load the pretrained `google/flan-t5-base` model and its tokenizer.
- Generate baseline summaries without explicit task instructions.
- Apply instruction-based prompting to guide the model towards dialogue summarisation.
- Compare different prompting strategies:
  - Zero-shot inference
  - One-shot inference
  - Few-shot inference
- Experiment with generation configuration parameters to understand their impact on generated outputs.

### Concepts Covered

- Prompt engineering
- In-context learning
- Zero-shot, one-shot and few-shot inference
- Tokenisation and decoding
- Encoder-decoder language models
- Text generation using Hugging Face Transformers
- Generation parameters such as `max_new_tokens`, sampling and decoding configurations

### Key Learning

Understand how providing task instructions and examples in the input prompt can influence the behaviour of a pretrained LLM without modifying its model parameters.

---

## 2. Fine-Tuning a Generative AI Model for Dialogue Summarisation

**Notebook:** `Lab_2_fine_tune_generative_ai_model.ipynb`

### Objective

Fine-tune a pretrained FLAN-T5 model for dialogue summarisation and compare full fine-tuning with Parameter-Efficient Fine-Tuning (PEFT).

### Implementation

**A. Baseline model evaluation**

- Load the DialogSum dataset.
- Load the pretrained FLAN-T5 model and tokenizer.
- Generate baseline summaries using zero-shot inference.
- Compare model-generated summaries with reference summaries.

**B. Dataset preprocessing**

- Prepare dialogue-summary pairs for supervised fine-tuning.
- Add task instructions to dialogue inputs.
- Tokenise input dialogues and target summaries.
- Prepare input IDs and labels for model training.

**C. Full fine-tuning**

- Configure the model for supervised fine-tuning.
- Use the Hugging Face `Trainer` API and `TrainingArguments`.
- Fine-tune the model on the dialogue summarisation task.
- Evaluate generated summaries qualitatively and quantitatively.

**D. Parameter-Efficient Fine-Tuning (PEFT)**

- Configure LoRA using the PEFT library.
- Train LoRA adapters instead of updating all model parameters.
- Examine the number and proportion of trainable parameters.
- Compare the PEFT model with the original model and full fine-tuning results.

**E. Model evaluation**

- Generate summaries using the fine-tuned models.
- Compare outputs against reference summaries.
- Use ROUGE metrics to evaluate summarisation quality.

### Concepts Covered

- Transfer learning
- Supervised fine-tuning (SFT)
- Full fine-tuning
- Parameter-Efficient Fine-Tuning (PEFT)
- Low-Rank Adaptation (LoRA)
- Hugging Face Trainer
- Model parameter efficiency
- Tokenisation and dataset preprocessing
- ROUGE evaluation metrics
- Qualitative and quantitative model evaluation

### Key Learning

Understand the trade-offs between updating all model parameters and using parameter-efficient adaptation techniques such as LoRA, particularly in terms of trainable parameters, computational requirements and evaluation performance.

---

## 3. Fine-Tuning FLAN-T5 with Reinforcement Learning (PPO) to Generate Less-Toxic Summaries

**Notebook:** `Lab_3_fine_tune_model_to_detoxify_summaries.ipynb`

### Objective

Explore reinforcement learning-based optimisation of a language model to generate less-toxic dialogue summaries.

The lab uses a pretrained FLAN-T5 model, a reward model based on a hate-speech classifier, and Proximal Policy Optimization (PPO).

### Implementation

**A. Dataset and model preparation**

- Load the DialogSum dataset.
- Prepare dialogue inputs using summarisation instructions.
- Load the FLAN-T5 model and the summarisation model checkpoint used in the exercise.
- Tokenise and prepare the dataset for PPO training.

**B. Reward model preparation**

- Load a pretrained hate-speech classification model.
- Use the classifier to estimate whether generated text is toxic.
- Use the classifier's output as a reward signal for model optimisation.

**C. Baseline toxicity evaluation**

- Generate summaries before reinforcement learning-based optimisation.
- Evaluate generated text using a toxicity evaluation pipeline.
- Establish a baseline for comparison.

**D. PPO-based optimisation**

- Configure the PPO training process using the TRL library.
- Prepare the policy model with a value head.
- Create a reference model to constrain changes to the policy.
- Generate summaries from dialogue prompts.
- Obtain reward signals from the reward model.
- Optimise the policy using Proximal Policy Optimization.
- Use KL-divergence-based constraints to discourage excessive deviation from the reference model.

**E. Post-training evaluation**

- Evaluate the model after PPO optimisation.
- Compare toxicity scores before and after optimisation.
- Inspect generated summaries qualitatively to understand the effect of the reward signal.

### Concepts Covered

- Reinforcement Learning from Human Feedback (RLHF) concepts
- Reward models
- Proximal Policy Optimization (PPO)
- Policy and reference models
- Value heads
- KL divergence
- Reward-based optimisation
- Toxicity evaluation
- Parameter-efficient fine-tuning concepts

### Key Learning

Understand how a reward model can provide feedback to a language model and how PPO can use that feedback to optimise model behaviour while a reference model and KL-divergence penalty help constrain policy updates.

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| LLM Framework | Hugging Face Transformers |
| Model | Google FLAN-T5 |
| Dataset | Hugging Face DialogSum |
| Dataset Processing | Hugging Face Datasets |
| Fine-Tuning | Hugging Face Trainer, PEFT |
| Parameter-Efficient Adaptation | LoRA |
| Reinforcement Learning | TRL, PPO |
| Evaluation | ROUGE, toxicity classification |
| Environment | Jupyter Notebook |

## Repository Structure

```text
generative-ai-with-llms/
│
├── README.md
│
├── 01-prompt-engineering/
│   └── Lab_1_summarize_dialogue.ipynb
│
├── 02-fine-tuning-and-peft/
│   └── Lab_2_fine_tune_generative_ai_model.ipynb
│
└── 03-rlhf-and-detoxification/
    └── Lab_3_fine_tune_model_to_detoxify_summaries.ipynb
```

## Setup and Execution

These notebooks were designed around the course's lab environment. Running them independently may require dependency adjustments.

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/generative-ai-with-llms.git

cd generative-ai-with-llms
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**macOS / Linux**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

### 3. Install dependencies

The original notebooks contain package versions used in the course environment. Refer to their dependency cells when recreating that environment.

Core packages include:

```bash
pip install torch transformers datasets
pip install accelerate evaluate rouge_score
pip install peft trl
pip install jupyter notebook pandas numpy tqdm
```

Some package versions may need adjustment because the original labs use older versions of Transformers, PEFT and TRL.

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the notebook you want to explore and execute its cells in sequence.

### Compute Requirements

- Model downloads require an internet connection.
- Fine-tuning and PPO experiments can be computationally intensive.
- A GPU-enabled environment such as Google Colab is recommended for training exercises.
- CPU execution may be possible for some inference and preprocessing tasks, but training can be slow or memory-intensive.

---

## Important Notes and Limitations

- This repository contains guided educational labs completed as part of a DeepLearning.AI course. It is intended to document learning and experimentation, not to represent three independently developed production applications.
- Training configurations in the original notebooks are exercise-scale and should not be interpreted as large-scale model training.
- The PPO exercise uses a limited training configuration to demonstrate the workflow; it is not a comprehensive RLHF training run.
- Model checkpoints and datasets may need to be downloaded separately.
- The notebooks may require dependency or environment changes to run outside the original course environment.
- Generated summaries and evaluation results depend on the model checkpoints, dataset samples and runtime configuration.

## Learning Outcomes

Through these exercises, I gained practical exposure to:

1. Prompt engineering and in-context learning strategies for LLMs.
2. Loading and running pretrained language models using Hugging Face Transformers.
3. Tokenisation, generation and decoding workflows.
4. Supervised fine-tuning of pretrained generative models.
5. Full fine-tuning versus Parameter-Efficient Fine-Tuning.
6. LoRA-based model adaptation.
7. Evaluation of summarisation models using ROUGE metrics.
8. Reward modelling and PPO-based language model optimisation.
9. KL divergence as a constraint during policy optimisation.
10. Toxicity evaluation and reward-based alignment concepts.

## Acknowledgements

These notebooks are based on the practical labs from [DeepLearning.AI's Generative AI with Large Language Models](https://www.deeplearning.ai/courses/generative-ai-with-llms/) course.

The original course materials, instructional content, datasets and model checkpoints belong to their respective authors and providers. Please refer to the course and associated resources for their original context and licensing terms.

---

**Purpose of this repository:** To document hands-on learning and practical experimentation with the core techniques used in Generative AI and Large Language Model development.
