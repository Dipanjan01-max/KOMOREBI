# Project KOMOREBI (木漏れ日)

**Multimodal Acoustic-Linguistic Fusion for Geriatric Clinical Diagnostics**

Project KOMOREBI is an end-to-end PyTorch pipeline engineered to detect early markers of Mild Cognitive Impairment (MCI) from native Japanese speech. Traditional diagnostic AI often suffers from "Western hallucination" when applied to Japanese geriatric care due to cultural and linguistic disconnects. This architecture addresses this by fusing raw acoustic prosody with semantic hesitation patterns, grounded by a culturally specific Retrieval-Augmented Generation (RAG) module.

## 🧠 System Architecture

The pipeline processes multimodal inputs (16kHz audio and text) through a custom cross-attention fusion engine:

1. **Acoustic Encoding:** Utilizes `reazon-research/japanese-wav2vec2-base` to extract prosodic features and detect micro-hesitations in speech waveforms.
2. **Linguistic Encoding:** Transcribes audio via `openai/whisper-small` and extracts semantic embeddings using `cl-tohoku/bert-base-japanese-v3` to identify empty-speech markers (e.g., word-finding difficulty).
3. **Cross-Attention Fusion:** A Multihead Attention layer mathematically aligns the acoustic keys/values with linguistic queries, fusing the modalities before classification.
4. **Cultural RAG Grounding:** A vector-search module (cosine similarity) queries a curated Showa-era (1950s) knowledge base to provide historically accurate context to patient utterances.
5. **Bilingual Logging:** Integrates offline machine translation (`Helsinki-NLP/opus-mt-ja-en`) to provide real-time English subtitles for diagnostic logs and RAG retrievals.

## 📂 Repository Structure

```text
KOMOREBI/
├── checkpoints/         # Model weights and fine-tuned states (Local)
├── data/                # Clinical audio samples (16kHz .wav) and transcripts
├── knowledge_graphs/    # RAG vector databases and Showa-era text corpora
├── logs/                # Training metrics, loss curves, and evaluation outputs
├── transcripts/         # Whisper ASR outputs and Helsinki-NLP translations
├── KOMOREBI.ipynb       # Main execution pipeline and fusion model architecture
└── README.md            # Project documentation
