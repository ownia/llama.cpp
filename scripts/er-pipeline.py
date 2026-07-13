# Evaluate TTS/ASR models and report WER/CER

import logging
import subprocess
import unicodedata
import warnings

import jiwer

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def tokenize_chinese(text):
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="pkg_resources is deprecated as an API.*",
                category=UserWarning,
            )
            import jieba
    except ImportError as exc:
        raise RuntimeError("Chinese WER requires jieba: pip install jieba") from exc

    jieba.setLogLevel(logging.WARNING)
    return " ".join(token for token in jieba.cut(text) if token.strip())


def tokenize_japanese(text):
    try:
        import fugashi
        import unidic_lite
    except ImportError as exc:
        raise RuntimeError(
            "Japanese WER requires fugashi and unidic-lite: "
            "pip install fugashi unidic-lite"
        ) from exc

    tagger = fugashi.Tagger(f"-r /dev/null -d {unidic_lite.DICDIR}")
    return " ".join(word.surface for word in tagger(text) if word.surface.strip())


def tokenize_for_wer(text, language="English"):
    if language == "Chinese":
        return tokenize_chinese(text)
    if language == "Japanese":
        return tokenize_japanese(text)

    return text


def log_tokenizer(language):
    if language == "Chinese":
        logger.info("Tokenizer: jieba for Chinese WER")
    elif language == "Japanese":
        logger.info("Tokenizer: fugashi for Japanese WER")
    else:
        logger.info("Tokenizer: whitespace for WER")


def remove_unicode_punctuation(text):
    return "".join(
        char for char in text if not unicodedata.category(char).startswith("P")
    )


def run_tts(reference_text, output_file="test.wav"):
    logger.info("TTS: generating %s", output_file)

    tts_cmd = [
        "taskset",
        "-c",
        "0-1,6-11",
        "./build_vulkan/bin/llama-qwen3tts",
        "--model-talker",
        "qwen3tts-talker-q4.gguf",
        "--model-cp",
        "qwen3tts-cp-q8.gguf",
        "--model-vocoder",
        "qwen3tts-tokenizer-f16.gguf",
        "-n-gpu-layers",
        "99",
        "--cp-n-gpu-layers",
        "99",
        "--text",
        reference_text,
        "--output",
        output_file,
        "-t",
        "8",
    ]

    subprocess.run(
        tts_cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    logger.info("TTS: done")


def run_asr(audio_file="test.wav", language="English"):
    logger.info("ASR: transcribing %s", audio_file)

    asr_cmd = [
        "setsid",
        "taskset",
        "-c",
        "0-1,6-11",
        "./build_vulkan/bin/llama-cli",
        "-m",
        "Qwen3-ASR-0.6B-Q4_0.gguf",
        "--mmproj",
        "mmproj-Qwen3-ASR-0.6B-Q8_0-convq8.gguf",
        "--audio",
        audio_file,
        "--prompt",
        f"language {language}<asr_text>",
        "-t",
        "8",
        "-ngl",
        "99",
        "--single-turn",
        "--no-display-prompt",
        "--log-disable",
    ]

    result = subprocess.run(
        asr_cmd,
        capture_output=True,
        text=True,
        check=True,
        stdin=subprocess.DEVNULL,
    )

    marker = "<asr_text>"
    if marker in result.stdout:
        asr_text = result.stdout.rsplit(marker, 1)[1].splitlines()[0].strip()
    else:
        asr_text = result.stdout.strip()

    logger.info("ASR: done")
    return asr_text


def compute_error_rates(reference_text, asr_text, language="English"):
    logger.info("Metrics: computing WER/CER")
    log_tokenizer(language)

    reference_text = remove_unicode_punctuation(reference_text)
    asr_text = remove_unicode_punctuation(asr_text)

    word_norm = jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.ReduceToListOfListOfWords(),
        ]
    )

    char_norm = jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.ReduceToListOfListOfChars(),
        ]
    )

    wer_reference = tokenize_for_wer(reference_text, language)
    wer_hypothesis = tokenize_for_wer(asr_text, language)

    wer = jiwer.wer(
        wer_reference,
        wer_hypothesis,
        reference_transform=word_norm,
        hypothesis_transform=word_norm,
    )
    cer = jiwer.cer(
        reference_text,
        asr_text,
        reference_transform=char_norm,
        hypothesis_transform=char_norm,
    )

    logger.info("Metrics: done")
    return wer, cer


def run_pipeline(reference_text, audio_file="test.wav", language="English"):
    logger.info("Pipeline: start")

    run_tts(reference_text, audio_file)
    asr_text = run_asr(audio_file, language)
    wer, cer = compute_error_rates(reference_text, asr_text, language)

    logger.info("Pipeline: done")

    return {
        "reference": reference_text,
        "hypothesis": asr_text,
        "wer": wer,
        "cer": cer,
    }


def main():
    examples = [
        {
            "language": "English",
            "reference_text": "The quick brown fox jumps over the lazy dog.",
            "audio_file": "test_en.wav",
        },
        {
            "language": "Chinese",
            "reference_text": "四是四，十是十。",
            "audio_file": "test_zh.wav",
        },
        {
            "language": "Japanese",
            "reference_text": "今夜は月が綺麗ですね",
            "audio_file": "test_ja.wav",
        },
    ]

    for example in examples:
        result = run_pipeline(
            example["reference_text"],
            example["audio_file"],
            example["language"],
        )

        print(f"language: {example['language']}")
        print(f"reference: {result['reference']}")
        print(f"hypothesis: {result['hypothesis']}")
        print(f"WER: {result['wer']:.4f}")
        print(f"CER: {result['cer']:.4f}")
        print()


if __name__ == "__main__":
    main()
