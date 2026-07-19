#import "../template/lib.typ" as template

= Experiment

The Universal Dependencies (UD) parsing pipeline was evaluated in both a direct setting (without tools) and an agentic setting, in which the model could invoke external tools when required. This section details the data collection and methodology of the NLP pipeline.


== Data collection and processing

// Fix two citations
Five Universal Dependencies (UD) treebanks were chosen to span a range of resource levels: English, Marathi, Taiwanese, Low_Saxon, and Vietnamese. Although the project initially had only 4 lower-resource languages, English as a benchmark for the evaluation was also added at a later stage. Taiwanese, being a lower-resource language, did not have the official UD site, so the converted version of Sinica Treebank, one of the major Mandarin Chinese treebanks, to Universal Dependencies was implemented (Hsieh et al., 2022). The columns were rearranged so that the data matched the official UD format and could be called by the stanza NLP toolkit (Qi et al. 2020).

For each language, ten sentences were drawn from the official test split: sentences were sorted by character length, the list was divided into twenty equal-sized bins, and the sentence at the middle ten cut points were selected. This picks sentences at approximately the 30th, 35th, 40th, ..., 75th percentiles of the length distribution. This yields a length-stratified sample rather than a random one, so that both very short and comparatively long sentences are represented while avoiding the extremes.


== Platform and dependencies

The following models were used in the experiment:
1. GGUF-quantized *Qwen3.5-9B* (Q4_K_M) (see HF model card: unsloth/Qwen3.5-9B-GGUF)
2. BitsAndBytes 4-bit NF4 quantization *Qwen3.5-9B* (see HF model card: Qwen/Qwen3.5-9B)

The reason for using two almost identical but ultimately different models was that, due to limited computational resources, the experiments had to be run in parallel on dissimilar hardware. To accommodate both inference modalities, the pipeline has two model wrappers:
- *LocalModel:* loads the model directly using the transformers library. This wrapper was used for running Qwen HF-compatible model. 
- *LlamaCppModel:* This wrapper sends requests to a running llama.cpp server. The GGUF-quantized model was served locally using llama.cpp. The wrapper communicated with the llama.cpp server through its OpenAI-compatible interface.


== Inference

// add system prompt to appendix
In the experiment, the model is given a reduced parsing task: for every token in a sentence, predict `HEAD` (the id of its syntactic governor or 0 if the token is the root) following UD v2 conventions. The model receives the sentence text and a JSON array of `{id, form}` token objects, and is required to return a JSON array of `{id, head}` objects with no additional text. There are two inference modalities that are compared to one another:

*Direct prompting* (baseline): The model receives the `DIRECT_PARSING_SYSTEM_PROMPT` and no tools. It must produce the head assignment for the whole sentence in a single pass of generation, relying only on its parametric knowledge.

*Agentic parsing:* The model receives the `AGENTIC_PARSING_SYSTEM_PROMPT`, which is identical to the direct-prompting prompt except that it instructs the model to prefer tool calls over extended chain-of-thought reasoning - and is given access to the four tools described in the previous section.

In both conditions, both system prompts explicitly discourage the model from restating or re-deriving its analysis and instruct it not to hedge ("wait", "actually", "let me reconsider"), to keep generations concise; the agentic prompt additionally frames tool use as a substitute for, rather than a supplement to, long chains of thought.

The agent is allowed up to 10 iterations of generation/tool-execution per sentence and up to 10,000 new tokens per generation call (`max_iters_per_call`, `max_new_tokens`); sampling is enabled with temperature of 1, top_p of 0.95, top_k of 20 and repetition penalty of 1.1.

Tool calls are parsed from the model output in either JSON or a pseudo-XML (`<tool_call>`/`<function=...>`) format, executed against the corresponding Python implementation, and the result is appended to the conversation as a `tool` message before the next generation step. The full message trace for every sentence—system prompt, user turn, all assistant/tool turns, and the final parse—is logged to a per-sentence JSON file for later analysis.

The entire agentic NLP pipeline is run on 50 sentences, 10 from each language. For every language and inference modality it took approximately 1 hour to run the HF Qwen model using a A100 GPU. Using consumer hardware (Intel Core Ultra 7 H155) and the unsloth version of the that model, the same process took approximately 4 hours.

== Evaluation Method

Predicted and gold trees are compared for each sentence across all languages using unlabeled attachment score (UAS): the proportion of tokens whose predicted head matches the gold head, computed per sentence and aggregated per language. Four additional diagnostics are recorded alongside UAS:
- whether the sentence was parsed successfully at all (i.e. the model returned a well-formed head assignment rather than leaving heads unset),
- whether the gold tree is projective (to allow later analysis of whether projectivity predicts parsing difficulty),
- the number of tool calls made during parsing (always 0 for the direct-prompting condition) and the number of calls to each of the four tools,
- how many tokens the model generated to produce the final output.
