# LLM-Development-Fine-Tuning-W-RAG-Portfolio
This is a repository which shows the basic LLM Development from scratch and the standard process of fine tuning an open weights model for industrial production use.

# AI-Powered Agricultural ERP: Automated Order Extraction & Hybrid RAG Pipeline

An end-to-end Artificial Intelligence ecosystem designed to automate order entry and process unstructured communications across the agricultural supply chain, covering fertilizers, seeds, animal feed, pesticides, and machinery spare parts.

This project bridges the gap between foundational Deep Learning research and enterprise production engineering through two distinct development pillars:

* **Enterprise ERP Order Extraction Engine:** A fine-tuned Qwen 2.5 7B model integrated with a Chunked Hybrid RAG pipeline and deterministic Python guardrails.
* **Foundational Custom SLM:** A causal Transformer model and custom Greek/English Byte-Level BPE Tokenizer built and pre-trained from scratch in PyTorch.

---

## Scope & Design Philosophy

In real-world ERP environments, incoming orders arrive as unstructured text—hasty sales rep notes, emails, or messages containing typos and Greeklish. Deploying a zero-shot LLM or a naive vector search pipeline in production fails for three primary reasons:

* **SKU & Price Hallucinations:** Probabilistic language models frequently invent non-existent warehouse codes, barcodes, or unit prices when queried directly.
* **Dense Retrieval Dilution:** Searching a single vector embedding for a multi-item order across hundreds of similar agricultural products retrieves noisy matches.
* **Arithmetic Instability:** Generative models cannot be trusted to execute financial multiplications or invoice totals reliably.

To solve these challenges, this system implements a neuro-symbolic hybrid architecture:

* **QLoRA Instruction Tuning:** The model is fine-tuned on domain-specific ERP data to extract entities, handle packaging unit conversions, and enforce strict JSON schemas.
* **Chunked Hybrid Retrieval:** Incoming orders are split into sub-clauses, combining FAISS dense vector search with Greek lexical stemming and synonym normalization.
* **Deterministic Python Guardrails:** Extracted SKUs, barcodes, tax IDs (AFM), and prices are verified against retrieved database records, locking unknown items to `N/A` and zero value.

---

## System Architecture

### Pillar 1: Production ERP Order Extraction (Qwen 2.5 7B + Hybrid RAG)

The production pipeline transforms raw Greek order text into validated, ERP-ready JSON payloads.

* **Master ERP Database:** A deterministic Python generator builds `erp_master_database.json` with 250 unique clients and 262 unique agricultural products.
* **Vector Index:** The RAG pipeline embeds all 512 client and product records into FAISS using the `paraphrase-multilingual-MiniLM-L12-v2` model.
* **Synthetic Training Dataset:** A Gemini API pipeline generates 1,471 structured instruction-tuning examples across 50 batches in `erp_training_dataset.json`.
* **QLoRA Fine-Tuning:** `Qwen2.5-7B-Instruct` is trained in 4-bit precision via Unsloth on an NVIDIA L4 GPU, updating 40,370,176 parameters (0.53% of 7,655,986,688).
* **Training Convergence:** Across 184 training steps, the fine-tuning loss drops from 1.2968 at step 10 to 0.2445 at step 180 (reaching 0.225851 at step 170).
* **Edge & On-Premise Export:** The merged weights are exported to GGUF format (`q4_k_m` quantization) with an Ollama Modelfile for local, privacy-preserving deployment.

### Pillar 2: Custom Foundational Transformer (From Scratch in PyTorch)

To demonstrate low-level mastery of language model internals, a custom architecture was designed and trained without pre-existing weights.

* **Bilingual & Noisy Corpus:** A synthetic pipeline generates 500 batches (49,665 lines) of realistic ERP logs containing Greek, English, abbreviations, and Greeklish.
* **Catastrophic Forgetting Mitigation:** The ERP logs are merged and shuffled with 104,243 lines from 2,500 Greek Wikipedia articles, yielding 153,908 training lines.
* **Custom BPE Tokenizer:** A Byte-Level BPE Tokenizer is trained from scratch with a vocabulary of 8,192 tokens to encode Greek agricultural terms efficiently.
* **Pre-LN Transformer:** Built in PyTorch with 33,746,944 trainable parameters (`d_model=512`, `n_heads=8`, `n_layers=8`, `max_seq_len=256`, GELU activation, Pre-Layer Normalization, and Embedding Scaling).
* **Mixed-Precision Pre-Training:** Trained for 3 epochs using AdamW, AMP `GradScaler`, and gradient clipping, reducing average loss from 7.8656 to 3.9544 and 3.5787.

---

## Live Execution Examples

### Scenario A: Verified Client & Catalog Products (Exact Match)

**Input Order:**
> *"Από τον Αγροτικό Συνεταιρισμό Θεσσαλίας, θέλουμε 100 τσουβάλια νιτρική αμμωνία και 50 κιλά σπόρο για καλαμπόκι. Στείλτε τα Λάρισα."*

**Generated Output:**
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
}
```
---

### Scenario B: Unknown Client & Out-of-Catalog Item (Safe Failure Mode)

**Input Order:**
> *"Η Μαρία Παπαδοπούλου ζήτησε 3 τσουβάλια τροφή για κουνέλια."*

**Generated Output:**
 
 ```json
 {
  "customer": {
    "name": "Μαρία Παπαδοπούλου",
    "afm": "N/A"
  },
  "order_items": [
    {
      "category": "ζωοτροφές",
      "item_description": "Τροφή για κουνέλια",
      "sku": "N/A",
      "barcode": "N/A",
      "quantity": 3,
      "unit": "τσουβάλια",
      "unit_price": 0.0,
      "total_price": 0.0
    }
  ],
  "total_order_value": 0.0,
  "delivery_notes": "Η Μαρία Παπαδοπούλου ζήτησε 3 τσουβάλια τροφή για κουνέλια."
}
```

---

## 📂 Repository Structure

* **`RAG_Dataset.ipynb`:** Deterministic Python generator for the master ERP database of clients and products.
* **`erp_master_database.json`:** Master catalog containing client tax records, SKUs, barcodes, and unit prices.
* **`RAG_Creation.ipynb`:** Vector embedding pipeline that builds and saves the local FAISS index.
* **`Output Format Fine Tuning Dataset_2.ipynb`:** Synthetic instruction dataset generator via the Gemini API.
* **`Qwen Fine Tuning_2.ipynb`:** QLoRA fine-tuning script with Unsloth and GGUF quantization export.
* **`Qwen_Fine_Tuned_w_Customers_&_Products_RAG_Inference_W_HuggingFace_.ipynb`:** Production inference engine featuring Chunked Hybrid RAG and a 10-scenario evaluation suite.
* **`Synthetic_Dataset_Generation_2.ipynb`:** Generator for noisy bilingual ERP system logs and support tickets.
* **`Greek_Wikipedia_Training_Dataset__2.ipynb`:** Extraction pipeline for the Greek Wikipedia corpus subset.
* **`Wikipedia_&_Synthetic_Dataset_Merge_&_Shuffle_2.ipynb`:** Corpus merging and shuffling pipeline to prevent catastrophic forgetting.
* **`Development and Training for 30M SLM_2.ipynb`:** From-scratch BPE tokenizer training and PyTorch Pre-LN Transformer pre-training.
* **`Test Model Inference W_ Hugging Face_2.ipynb`:** Standalone inference script loading the custom Transformer from Hugging Face.

---

## 🤗 Hugging Face Hub Artifacts

* **Production Qwen 2.5 7B Model & FAISS Index:** [deanwashburn/Qwen2.5-7B-ERP-Order-Extraction](https://huggingface.co/deanwashburn/Qwen2.5-7B-ERP-Order-Extraction)
* **Custom PyTorch Transformer & BPE Tokenizer:** [deanwashburn/Agro_30M](https://huggingface.co/deanwashburn/Agro_30M)
