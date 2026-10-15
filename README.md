# Emotion-Aware Personalized Voice Messaging

A notebook-based NLP and speech-synthesis project that explores a simple question: **can a written message be spoken in a personal voice while reflecting the emotion expressed in the text?**

The notebook combines a pretrained text emotion classifier with several text-to-speech systems, progressing from a basic spoken message to voice cloning and manually tuned emotional delivery.

**Course:** IST 664 Natural Language Processing  
**Authors:** Eric Lee, Gautam Balgi, and Jarrett Markman

## Start here

- [Main notebook](Main_Python_Code_File/IST_664_NLP_Final_Project.ipynb): all code, experiments, observations, and embedded outputs.
- [Final report](Final_report.docx): accompanying project write-up.

This is an experimental notebook, with separate model trials and environment changes. It is not packaged as a messaging app or a one-command service.

## How it works

```text
Written message → emotion classification → speech-generation experiment → audio playback
                                                 ↑
                                 reference recording for voice cloning
```

1. The classifier `j-hartmann/emotion-english-distilroberta-base` scores the message and selects its highest-scoring emotion.
2. Depending on the experiment, that label is displayed, added to a speech prompt, or converted into a conditioning vector.
3. A speech model generates audio. Voice-cloning experiments also use a speaker recording.
4. The notebook saves the result and plays it with `IPython.display.Audio`.

The notebook loads the `go_emotions` dataset in an early cell, but it does **not** train or fine-tune its classifier on that dataset.

## Experiments included

| System | What the notebook does | What to expect |
|---|---|---|
| gTTS | Predicts an emotion, then reads the input text aloud | A baseline; the emotion is printed and used in the filename, but is not passed to gTTS as expressive control |
| Coqui Jenny | Passes an `emotion` argument to `tts_models/en/jenny/jenny` | Notebook observations report mostly neutral speech with little clear variation |
| Bark | Prefixes text with a tag such as `[sadness]` | Explores prompt-based expression; the notebook reports somewhat clearer emotion |
| YourTTS | Uses `my_voice.wav` with `tts_models/multilingual/multi-dataset/your_tts` | Voice cloning; observations describe robotic delivery and limited emotional variation |
| Zonos | Creates a speaker embedding, conditions on text and an emotion vector, and generates audio | Combines cloning and emotional conditioning, followed by manual pitch and speaking-rate experiments |

These are qualitative experiments. The repository does not provide a controlled listening study, a quantitative speech-quality benchmark, or evidence that every predicted emotion is reproduced reliably.

## Running the notebook

### 1. Choose an experiment

Open the notebook in Google Colab. Several cells use Colab-specific upload tools, `/content`-style workflows, Linux package commands, or GPU-specific installation commands. A local Jupyter environment requires adapting those cells.

Use a fresh runtime for each major speech-model section. **Do not run every installation cell in sequence:** the Bark and Zonos sections replace shared packages, and later sections depend on variables created earlier.

### 2. Run the classifier and baseline first

For an initial Colab session, install the baseline dependencies:

```python
%pip install datasets transformers torch gtts
```

Run the dataset/classifier cells, define `get_predicted_emotion`, then run the gTTS examples. Enter a short English message when prompted. The expected outputs are a predicted label, an audio file, and notebook playback.

### 3. Explore one speech model at a time

- **Jenny / YourTTS:** follow the section's `TTS` installation and model-loading cells in a compatible environment. YourTTS expects a reference recording named `my_voice.wav`; upload that filename or edit `speaker_wav`.
- **Bark:** inspect the installation cell before running it. It uninstalls PyTorch packages and installs a nightly CUDA 11.8 build. The section also overrides `torch.load` with `weights_only=False`. These are notebook-specific compatibility workarounds, not general environment defaults; keep this experiment isolated and use trusted checkpoints.
- **Zonos:** follow the notebook's repository installation and `espeak-ng` setup. Its cells pin NumPy to `1.26.4` and Transformers to `4.41.1`. Restart the runtime if required, then rerun imports and model initialization. GPU availability and memory affect whether generation succeeds.

For cloning, use a reference recording you have permission to use. The Zonos upload cell takes the first uploaded file and uses it to build the speaker embedding.

### 4. Check the label mapping before emotion-conditioned Zonos generation

The classifier returns labels such as `joy`, `sadness`, and `anger`, but the notebook's `emotion_map` uses `happy`, `sad`, and `angry`. Unmatched labels fall back to a neutral vector. To exercise the intended mapping, normalize those labels before calling `get_emotion_vector`, for example:

```python
label_aliases = {"joy": "happy", "sadness": "sad", "anger": "angry"}
mapped_label = label_aliases.get(predicted_emotion, predicted_emotion)
emotion_vec = get_emotion_vector(mapped_label)
```

This is a setup correction for your notebook session; the checked-in notebook retains its original code. The final tuning cells also demonstrate manually supplied happy/surprise mixtures rather than automatically predicted vectors.

## Outputs

Examples include `output.mp3`, `bark_output_<Emotion>.wav`, `output_cloned.wav`, `output.wav`, `emotion_voice.wav`, and `happy_output.wav`. Repeated runs can overwrite files with the same name. Some Coqui examples use an `.mp3` filename; verify the actual encoding supported by the installed backend and use a matching extension when adapting them.

## Limitations and troubleshooting

- **Package installation fails:** the repository has no unified requirements file or tested environment lock. Use the model section's compatible Python and dependency environment; one environment may not support every backend.
- **Undefined variables after restarting:** rerun classifier definitions, imports, model loading, and reference-audio cells required by the chosen section.
- **Neutral output despite emotional text:** check the label aliases above. Speech-model support and prompt tags also limit expressive control.
- **Out-of-memory errors:** use a GPU runtime with sufficient memory and avoid keeping multiple speech models loaded together.
- **Interpretation:** a text label is a model prediction about language, not a measurement of the author's internal emotional state.

The repository contains one experiment notebook and one report. It does not include a deployed UI, messaging integration, packaged API, or automated test suite.
