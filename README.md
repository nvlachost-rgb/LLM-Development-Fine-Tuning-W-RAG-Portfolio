# LLM-Development-Fine-Tuning-W-RAG-Portfolio
This is a repository which shows the basic LLM Development from scratch and the standard process of fine tuning an open weights model for industrial production use.

# AI-Powered Agricultural ERP: Automated Order Extraction & Hybrid RAG Pipeline

An end-to-end Artificial Intelligence ecosystem designed to automate order entry and process unstructured communications across the agricultural supply chain, covering fertilizers, seeds, animal feed, pesticides, and machinery spare parts[cite: 19, 22, 24, 28].

This project bridges the gap between foundational Deep Learning research and enterprise production engineering through two distinct development pillars[cite: 20, 21, 26]:

* **Enterprise ERP Order Extraction Engine:** A fine-tuned Qwen 2.5 7B model integrated with a Chunked Hybrid RAG pipeline and deterministic Python guardrails[cite: 20, 21].
* **Foundational Custom SLM:** A causal Transformer model and custom Greek/English Byte-Level BPE Tokenizer built and pre-trained from scratch in PyTorch[cite: 25, 26].

---

## Scope & Design Philosophy

In real-world ERP environments, incoming orders arrive as unstructured text—hasty sales rep notes, emails, or messages containing typos and Greeklish[cite: 24, 28]. Deploying a zero-shot LLM or a naive vector search pipeline in production fails for three primary reasons:

* **SKU & Price Hallucinations:** Probabilistic language models frequently invent non-existent warehouse codes, barcodes, or unit prices when queried directly.
* **Dense Retrieval Dilution:** Searching a single vector embedding for a multi-item order across hundreds of similar agricultural products retrieves noisy matches[cite: 19, 21, 22].
* **Arithmetic Instability:** Generative models cannot be trusted to execute financial multiplications or invoice totals reliably.

To solve these challenges, this system implements a neuro-symbolic hybrid architecture:

* **QLoRA Instruction Tuning:** The model is fine-tuned on domain-specific ERP data to extract entities, handle packaging unit conversions, and enforce strict JSON schemas[cite: 20, 21, 24].
* **Chunked Hybrid Retrieval:** Incoming orders are split into sub-clauses, combining FAISS dense vector search with Greek lexical stemming and synonym normalization[cite: 21, 23].
* **Deterministic Python Guardrails:** Extracted SKUs, barcodes, tax IDs (AFM), and prices are verified against retrieved database records, locking unknown items to `N/A` and zero value.

---

## System Architecture

### Pillar 1: Production ERP Order Extraction (Qwen 2.5 7B + Hybrid RAG)

The production pipeline transforms raw Greek order text into validated, ERP-ready JSON payloads.

* **Master ERP Database:** A deterministic Python generator builds `erp_master_database.json` with 250 unique clients and 262 unique agricultural products.
* **Vector Index:** The RAG pipeline embeds all 512 client and product records into FAISS using the `paraphrase-multilingual-MiniLM-L12-v2` model[cite: 21, 23].
* **Synthetic Training Dataset:** A Gemini API pipeline generates 1,471 structured instruction-tuning examples across 50 batches in `erp_training_dataset.json`[cite: 20, 24].
* **QLoRA Fine-Tuning:** `Qwen2.5-7B-Instruct` is trained in 4-bit precision via Unsloth on an NVIDIA L4 GPU, updating 40,370,176 parameters (0.53% of 7,655,986,688).
* **Training Convergence:** Across 184 training steps, the fine-tuning loss drops from 1.2968 at step 10 to 0.2445 at step 180 (reaching 0.225851 at step 170)[cite: 20].
* **Edge & On-Premise Export:** The merged weights are exported to GGUF format (`q4_k_m` quantization) with an Ollama Modelfile for local, privacy-preserving deployment[cite: 20].

### Pillar 2: Custom Foundational Transformer (From Scratch in PyTorch)

To demonstrate low-level mastery of language model internals, a custom architecture was designed and trained without pre-existing weights[cite: 25, 26].

* **Bilingual & Noisy Corpus:** A synthetic pipeline generates 500 batches (49,665 lines) of realistic ERP logs containing Greek, English, abbreviations, and Greeklish[cite: 27, 28].
* **Catastrophic Forgetting Mitigation:** The ERP logs are merged and shuffled with 104,243 lines from 2,500 Greek Wikipedia articles, yielding 153,908 training lines[cite: 27, 29].
* **Custom BPE Tokenizer:** A Byte-Level BPE Tokenizer is trained from scratch with a vocabulary of 8,192 tokens to encode Greek agricultural terms efficiently[cite: 25, 26].
* **Pre-LN Transformer:** Built in PyTorch with 33,746,944 trainable parameters (`d_model=512`, `n_heads=8`, `n_layers=8`, `max_seq_len=256`, GELU activation, Pre-Layer Normalization, and Embedding Scaling)[cite: 25, 26].
* **Mixed-Precision Pre-Training:** Trained for 3 epochs using AdamW, AMP `GradScaler`, and gradient clipping, reducing average loss from 7.8656 to 3.9544 and 3.5787.

---

## Live Execution Examples

### Scenario A: Verified Client & Catalog Products (Exact Match)

**Input Order:**
> *"Από τον Αγροτικό Συνεταιρισμό Θεσσαλίας, θέλουμε 100 τσουβάλια νιτρική αμμωνία και 50 κιλά σπόρο για καλαμπόκι. Στείλτε τα Λάρισα."*[cite: 21]

**Generated Output:**[cite: 21]
```json
{
  "customer": {
    "name": "Αγροτικός Συνεταιρισμός Θεσσαλίας",
    "afm": "100200300"
  },
  "order_items": [
    {
      "category": "λιπάσματα",
      "item_description": "Λίπασμα Νιτρική Αμμωνία 34.4%",
      "sku": "FER-001",
      "barcode": "5201000000001",
      "quantity": 100,
      "unit": "τσουβάλια",
      "unit_price": 28.5,
      "total_price": 2850.0
    },
    {
      "category": "σπόροι",
      "item_description": "Σπόρος Καλαμποκιού Υβρίδιο F1",
      "sku": "SEED-001",
      "barcode": "5201000000006",
      "quantity": 50,
      "unit": "κιλά",
      "unit_price": 85.5,
      "total_price": 4275.0
    }
  ],
  "total_order_value": 7125.0,
  "delivery_notes": "Από τον Αγροτικό Συνεταιρισμό Θεσσαλίας, θέλουμε 100 τσουβάλια νιτρική αμμωνία και 50 κιλά σπόρο για καλαμπόκι. Στείλτε τα Λάρισα."
}
