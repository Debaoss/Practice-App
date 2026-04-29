"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { env, pipeline } from "@huggingface/transformers";

const isSafari = /^((?!chrome|android).)*safari/i.test(navigator.userAgent);
const isAppleDevice = /iPad|iPhone|iPod|Macintosh|Mac OS/i.test(navigator.userAgent);
const useWebKitMemorySavings = isSafari || isAppleDevice;

if (env.backends.onnx?.wasm) {
  env.backends.onnx.wasm.proxy = !useWebKitMemorySavings;
  // Disable WASM entirely on Apple devices to avoid memory issues
  if (useWebKitMemorySavings) {
    env.backends.onnx.wasm.numThreads = 1;
  }
}

const MODEL_ID = "onnx-community/ast-finetuned-audioset-10-10-0.4593-ONNX";
const TARGET_SAMPLE_RATE = 16_000;
const WINDOW_SECONDS = 3;
const CAPTURE_INTERVAL_MS = 120;
const CLASSIFY_COOLDOWN_MS = 900;
const SILENCE_DEFAULT_MS = 60_000;
const MATCH_GRACE_MS = 700;

const TARGET_OPTIONS = [
  {
    value: "piano",
    label: "Piano",
    modelLabels: ["piano", "keyboard (musical)", "electric piano", "organ", "harpsichord"],
  },
  {
    value: "voice",
    label: "Voice",
    modelLabels: [
      "speech",
      "male speech, man speaking",
      "female speech, woman speaking",
      "child speech, kid speaking",
      "conversation",
      "narration, monologue",
      "babbling",
      "speech synthesizer",
      "shout",
      "bellow",
      "whoop",
      "yell",
      "battle cry",
      "children shouting",
      "screaming",
      "whispering",
      "laughter",
      "baby laughter",
      "giggle",
      "snicker",
      "belly laugh",
      "chuckle, chortle",
      "crying, sobbing",
      "baby cry, infant cry",
      "whimper",
      "wail, moan",
      "sigh",
      "singing",
      "choir",
      "yodeling",
      "chant",
      "mantra",
      "male singing",
      "female singing",
      "child singing",
      "synthetic singing",
      "rapping",
      "humming",
      "groan",
      "grunt",
      "whistling",
    ],
  },
] as const;

type TargetValue = (typeof TARGET_OPTIONS)[number]["value"];
type Prediction = {
  label: string;
  score: number;
};

type ClassifierInput = Float32Array | { array: Float32Array; sampling_rate: number };
type Classifier = (audio: ClassifierInput) => Promise<Prediction[]>;

type Verdict = {
  headline: string;
  detail: string;
  confidence: number;
  matched: boolean;
  topLabel: string;
};

type ActiveTab = "classifier" | "timer";

function normalizeLabel(label: string) {
  return label.toLowerCase().replace(/[_-]/g, " ").replace(/\s+/g, " ").trim();
}

function titleCase(label: string) {
  return normalizeLabel(label).replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function findTarget(value: TargetValue) {
  return TARGET_OPTIONS.find((target) => target.value === value) ?? TARGET_OPTIONS[0];
}

function getBucketKey(label: string): TargetValue | null {
  const normalized = normalizeLabel(label);

  // Try to match exactly first, then fall back to substring / token matches for robustness.
  for (const option of TARGET_OPTIONS) {
    for (const candidate of option.modelLabels) {
      const candNorm = normalizeLabel(candidate);
      if (candNorm === normalized) return option.value;
    }
  }

  // Fallback: substring or token overlap matching
  const tokens = new Set(normalized.split(/\s+/));
  for (const option of TARGET_OPTIONS) {
    for (const candidate of option.modelLabels) {
      const candNorm = normalizeLabel(candidate);
      if (normalized.includes(candNorm) || candNorm.includes(normalized)) return option.value;

      const candTokens = candNorm.split(/\s+/);
      for (const t of candTokens) {
        if (t && tokens.has(t)) return option.value;
      }
    }
  }

  return null;
}

function bucketPredictions(predictions: Prediction[]) {
  const buckets = new Map<TargetValue, Prediction>();

  for (const prediction of predictions) {
    const bucketKey = getBucketKey(prediction.label);
    if (!bucketKey) {
      continue;
    }

    const current = buckets.get(bucketKey);
    if (!current || prediction.score > current.score) {
      buckets.set(bucketKey, { label: findTarget(bucketKey).label, score: prediction.score });
    }
  }

  return TARGET_OPTIONS.map((option) => buckets.get(option.value) ?? { label: option.label, score: 0 }).sort(
    (left, right) => right.score - left.score,
  );
}

function formatDuration(totalSeconds: number) {
  const safeSeconds = Math.max(0, totalSeconds);
  const minutes = Math.floor(safeSeconds / 60);
  const seconds = safeSeconds % 60;

  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

function resampleLinear(input: Float32Array, sourceRate: number, targetRate: number) {
  if (sourceRate === targetRate) {
    return input;
  }

  const ratio = sourceRate / targetRate;
  const outputLength = Math.max(1, Math.floor(input.length / ratio));
  const output = new Float32Array(outputLength);

  for (let index = 0; index < outputLength; index += 1) {
    const position = index * ratio;
    const leftIndex = Math.floor(position);
    const rightIndex = Math.min(input.length - 1, leftIndex + 1);
    const weight = position - leftIndex;
    output[index] = input[leftIndex] * (1 - weight) + input[rightIndex] * weight;
  }

  return output;
}

function extractLatestWindow(
  buffer: Float32Array,
  writeIndex: number,
  availableSamples: number,
) {
  if (availableSamples < buffer.length) {
    return null;
  }

  const start = writeIndex;
  const end = writeIndex + buffer.length;

  if (end <= buffer.length) {
    return buffer.slice(start, end);
  }

  const wrapped = new Float32Array(buffer.length);
  wrapped.set(buffer.subarray(start));
  wrapped.set(buffer.subarray(0, end - buffer.length), buffer.length - start);
  return wrapped;
}

function buildVerdict(targetValue: TargetValue, predictions: Prediction[]): Verdict {
  const target = findTarget(targetValue);
  const top = predictions[0] ?? { label: "unknown", score: 0 };

  const topLabel = top.label;
  const confidence = Math.round(top.score * 100);
  const matched = normalizeLabel(topLabel) === normalizeLabel(target.label) && top.score > 0;

  if (matched) {
    return {
      matched: true,
      headline: `Likely ${target.label}`,
      detail: `The model is hearing ${titleCase(topLabel)}.`,
      confidence,
      topLabel,
    };
  }

  return {
    matched: false,
    headline: `More like ${titleCase(topLabel)}`,
    detail: `That window does not look like ${target.label} right now.`,
    confidence,
    topLabel,
  };
}

export default function AudioClassifier() {
  const [activeTab, setActiveTab] = useState<ActiveTab>("classifier");
  const [targetValue, setTargetValue] = useState<TargetValue>("piano");
  const [isListening, setIsListening] = useState(false);
  const [status, setStatus] = useState("Idle. Start the microphone to begin.");
  const [error, setError] = useState<string | null>(null);
  const [modelReady, setModelReady] = useState(false);
  const [bufferProgress, setBufferProgress] = useState(0);
  const [timerMinutes, setTimerMinutes] = useState(1);
  const [timerSeconds, setTimerSeconds] = useState(0);
  const [timerRemainingSeconds, setTimerRemainingSeconds] = useState(60);
  const [timerRunning, setTimerRunning] = useState(false);
  const [showComplete, setShowComplete] = useState(false);
  const [verdict, setVerdict] = useState<Verdict>({
    headline: "Waiting for audio",
    detail: "Start the microphone, then play or sing into it.",
    confidence: 0,
    matched: false,
    topLabel: "unknown",
  });
  const [predictions, setPredictions] = useState<Prediction[]>([]);

  const classifierPromiseRef = useRef<Promise<Classifier> | null>(null);
  const classifierRef = useRef<Classifier | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const isAudibleRef = useRef(true);
  const [isAudible, setIsAudible] = useState(true);
  const silentFramesRef = useRef(0);
  const rmsRef = useRef(0);
  const [rms, setRms] = useState(0);
  const timerRunningRef = useRef(timerRunning);
  const timerPausedBySilenceRef = useRef(false);
  const verdictRef = useRef<Verdict>(verdict);
  const silentSinceRef = useRef<number | null>(null);
  const defaultedToVoiceRef = useRef(false);
  const intervalRef = useRef<number | null>(null);
  const ringBufferRef = useRef<Float32Array | null>(null);
  const writeIndexRef = useRef(0);
  const bufferedSamplesRef = useRef(0);
  const inFlightRef = useRef(false);
  const lastClassifiedAtRef = useRef(0);
  const lastMatchedAtRef = useRef(0);
  const targetRef = useRef(targetValue);
  const timerDurationSeconds = Math.max(1, timerMinutes * 60 + timerSeconds);

  useEffect(() => {
    targetRef.current = targetValue;
  }, [targetValue]);

  useEffect(() => {
    if (!timerRunning) {
      return;
    }

    const timer = window.setInterval(() => {
      setTimerRemainingSeconds((current) => {
        if (current <= 0) {
          setTimerRunning(false);
          setShowComplete(true);
          return 0;
        }

        const now = performance.now();
        const recentlyMatched = verdict.matched || now - lastMatchedAtRef.current < MATCH_GRACE_MS;
        if (!isListening || !recentlyMatched) {
          return current;
        }

        const nextValue = Math.max(0, current - 1);
        if (nextValue === 0) {
          setTimerRunning(false);
          setShowComplete(true);
        }

        return nextValue;
      });
    }, 1000);

    return () => window.clearInterval(timer);
  }, [isListening, timerRemainingSeconds, timerRunning, verdict.matched]);

  useEffect(() => {
    timerRunningRef.current = timerRunning;
  }, [timerRunning]);

  useEffect(() => {
    verdictRef.current = verdict;
    // if verdict becomes matched and we were auto-paused by silence and audio is audible, resume
    if (timerPausedBySilenceRef.current && isAudibleRef.current && verdict.matched) {
      timerPausedBySilenceRef.current = false;
      setTimerRunning(true);
    }
  }, [verdict, isAudible]);

  useEffect(() => {
    classifierPromiseRef.current ??= (() => {
      const attemptLoad = async (dtype: string, attemptNum: number): Promise<Classifier> => {
        console.log(`[Model Load] Attempt ${attemptNum}: dtype=${dtype}, isSafari=${isSafari}`);
        try {
          const classifier = (await pipeline("audio-classification", MODEL_ID, {
            dtype: dtype as "q4" | "q8" | "fp16",
          })) as Classifier;
          console.log(`[Model Load] Success with dtype=${dtype}`);
          return classifier;
        } catch (err) {
          const errMsg = err instanceof Error ? err.message : String(err);
          console.log(`[Model Load] Failed attempt ${attemptNum}: ${errMsg}`);
          throw err;
        }
      };

      const loadWithFallback = async (): Promise<Classifier> => {
        // Try in order: q4 (default), q8 (more quantized), fp16 (less quantized)
        const dtypes = useWebKitMemorySavings ? ["q8", "fp16", "q4"] : ["q4", "q8"];

        for (let i = 0; i < dtypes.length; i++) {
          try {
            return await attemptLoad(dtypes[i], i + 1);
          } catch (err) {
            if (i === dtypes.length - 1) {
              // Last attempt failed, throw final error
              throw err;
            }
            // Try next dtype
            await new Promise((resolve) => setTimeout(resolve, 500));
          }
        }
        throw new Error("All load attempts failed");
      };

      return Promise.race([
        loadWithFallback(),
        new Promise<Classifier>((_, reject) =>
          setTimeout(() => {
            reject(
              new Error(
                `Model loading timeout after 60s${useWebKitMemorySavings ? " (Apple device detected)" : ""}. Try refreshing the page.`,
              ),
            );
          }, 60_000),
        ),
      ]);
    })();

    if (typeof window !== "undefined") {
      console.log("Loading audio model. useWebKitMemorySavings:", useWebKitMemorySavings);
    }

    classifierPromiseRef.current
      .then((classifier) => {
        classifierRef.current = classifier;
        setModelReady(true);
        setStatus((current) =>
          current === "Idle. Start the microphone to begin." ? "Model ready. Start the microphone to begin." : current,
        );
      })
      .catch((err: unknown) => {
        const message = err instanceof Error ? err.message : "Failed to load the instrument model.";
        setError(message);
        setStatus("Model load failed: " + message);
      });

    return () => {
      if (intervalRef.current !== null) {
        window.clearInterval(intervalRef.current);
        intervalRef.current = null;
      }

      sourceRef.current?.disconnect();
      analyserRef.current?.disconnect();
      sourceRef.current = null;
      analyserRef.current = null;

      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;

      void audioContextRef.current?.close();
      audioContextRef.current = null;

      ringBufferRef.current = null;
      writeIndexRef.current = 0;
      bufferedSamplesRef.current = 0;
    };
  }, []);

  useEffect(() => {
    if (typeof window !== "undefined") {
      console.log("Device debug info:", {
        userAgent: navigator.userAgent,
        isSafari,
        isAppleDevice,
        useWebKitMemorySavings,
        platform: navigator.platform,
        language: navigator.language,
      });
    }
  }, []);

  const classifyWindow = async () => {
    if (inFlightRef.current) {
      return;
    }

    const classifier = classifierRef.current;
    const audioContext = audioContextRef.current;
    const ringBuffer = ringBufferRef.current;

    if (!classifier || !audioContext || !ringBuffer) {
      return;
    }

    const latestWindow = extractLatestWindow(ringBuffer, writeIndexRef.current, bufferedSamplesRef.current);
    if (!latestWindow) {
      return;
    }

    inFlightRef.current = true;
    setStatus(`Analyzing ${targetRef.current.replace(/-/g, " ")}...`);

    try {
      const audio = resampleLinear(latestWindow, audioContext.sampleRate, TARGET_SAMPLE_RATE);
      let output: Prediction[];

      try {
        output = (await classifier(audio)) as Prediction[];
      } catch {
        output = (await classifier({ array: audio, sampling_rate: TARGET_SAMPLE_RATE })) as Prediction[];
      }

      const ranked = bucketPredictions(Array.isArray(output) ? output : []);
      setPredictions(ranked);
      const v = buildVerdict(targetRef.current, ranked);
      setVerdict(v);
      if (v.matched) {
        lastMatchedAtRef.current = performance.now();
      }
      lastClassifiedAtRef.current = performance.now();
      setStatus("Listening for piano or voice...");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Classification failed.");
      setStatus("Classification failed.");
    } finally {
      inFlightRef.current = false;
    }
  };

  const captureAudioTick = () => {
    const analyser = analyserRef.current;
    const ringBuffer = ringBufferRef.current;

    if (!analyser || !ringBuffer) {
      return;
    }

    const chunk = new Float32Array(analyser.fftSize);
    analyser.getFloatTimeDomainData(chunk);

    let writeIndex = writeIndexRef.current;
    let bufferedSamples = bufferedSamplesRef.current;

    for (let index = 0; index < chunk.length; index += 1) {
      ringBuffer[writeIndex] = chunk[index];
      writeIndex = (writeIndex + 1) % ringBuffer.length;
      bufferedSamples = Math.min(bufferedSamples + 1, ringBuffer.length);
    }

    writeIndexRef.current = writeIndex;
    bufferedSamplesRef.current = bufferedSamples;
    setBufferProgress(bufferedSamples / ringBuffer.length);

    // compute RMS to detect silence/low volume
    let sum = 0;
    for (let i = 0; i < chunk.length; i++) {
      const s = chunk[i];
      sum += s * s;
    }
    const rms = Math.sqrt(sum / chunk.length) || 0;
    rmsRef.current = rms;
    setRms(rms);
    const AUDIBLE_RMS_THRESHOLD = 0.01;

    if (rms < AUDIBLE_RMS_THRESHOLD) {
      silentFramesRef.current += 1;
    } else {
      silentFramesRef.current = 0;
    }

    const nowAudible = silentFramesRef.current < 3;
    if (nowAudible !== isAudibleRef.current) {
      isAudibleRef.current = nowAudible;
      setIsAudible(nowAudible);

      if (!nowAudible) {
        // mark when the silence period started
        if (silentSinceRef.current === null) {
          silentSinceRef.current = performance.now();
        }
        // auto-pause the timer due to low audio, but remember that we paused it
        if (timerRunningRef.current) {
          timerPausedBySilenceRef.current = true;
          setTimerRunning(false);
        }
      } else {
        // audio returned: clear silence start and possibly resume
        silentSinceRef.current = null;
        defaultedToVoiceRef.current = false;
        // if we auto-paused earlier and the verdict currently matches, resume
        if (timerPausedBySilenceRef.current && verdictRef.current?.matched) {
          timerPausedBySilenceRef.current = false;
          setTimerRunning(true);
        } else {
          // clear the auto-pause flag if audible but not resuming
          timerPausedBySilenceRef.current = false;
        }
      }
    }

    // if we've been quiet for long enough, default to voice
    if (!nowAudible && silentSinceRef.current !== null && !defaultedToVoiceRef.current) {
      const elapsed = performance.now() - silentSinceRef.current;
      if (elapsed >= SILENCE_DEFAULT_MS) {
        // set predictions to Voice bucket and update verdict
        const voicePrediction: Prediction = { label: findTarget("voice").label, score: 1 };
        const ranked = TARGET_OPTIONS.map((option) => (option.value === "voice" ? voicePrediction : { label: option.label, score: 0 }));
        setPredictions(ranked);
        setVerdict(buildVerdict(targetRef.current, ranked));
        defaultedToVoiceRef.current = true;
        setStatus("Quiet for 1 minute — defaulting to Voice");

        // if the user's target is voice, resume timer
        if (targetRef.current === "voice") {
          timerPausedBySilenceRef.current = false;
          setTimerRunning(true);
        }
      }
    }

    const now = performance.now();
    if (
      bufferedSamples >= ringBuffer.length &&
      now - lastClassifiedAtRef.current >= CLASSIFY_COOLDOWN_MS &&
      !inFlightRef.current
    ) {
      void classifyWindow();
    }
  };

  async function startListening() {
    try {
      setError(null);
      setStatus("Requesting microphone access...");

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const audioContextCtor = window.AudioContext ?? (window as Window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!audioContextCtor) {
        throw new Error("This browser does not support AudioContext.");
      }

      const audioContext = new audioContextCtor({ sampleRate: TARGET_SAMPLE_RATE });
      await audioContext.resume();
      audioContextRef.current = audioContext;

      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 4096;
      analyser.smoothingTimeConstant = 0.02;
      analyserRef.current = analyser;

      const source = audioContext.createMediaStreamSource(stream);
      source.connect(analyser);
      sourceRef.current = source;

      const captureWindow = new Float32Array(audioContext.sampleRate * WINDOW_SECONDS);
      ringBufferRef.current = captureWindow;
      writeIndexRef.current = 0;
      bufferedSamplesRef.current = 0;
      lastClassifiedAtRef.current = 0;

      setIsListening(true);
      setBufferProgress(0);
      setStatus(modelReady ? "Listening for piano or voice..." : "Microphone ready. Loading model...");

      intervalRef.current = window.setInterval(captureAudioTick, CAPTURE_INTERVAL_MS);

      if (!classifierRef.current) {
        setStatus("Loading instrument model...");
        const classifier = await classifierPromiseRef.current;
        classifierRef.current = classifier;
        setModelReady(true);
        setStatus("Listening for piano or voice...");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not access the microphone.");
      setStatus("Microphone unavailable.");
      void stopListening();
    }
  }

  async function stopListening() {
    if (intervalRef.current !== null) {
      window.clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    sourceRef.current?.disconnect();
    analyserRef.current?.disconnect();
    sourceRef.current = null;
    analyserRef.current = null;

    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;

    if (audioContextRef.current) {
      await audioContextRef.current.close();
      audioContextRef.current = null;
    }

    ringBufferRef.current = null;
    writeIndexRef.current = 0;
    bufferedSamplesRef.current = 0;
    setBufferProgress(0);
    setIsListening(false);
    if (!error) {
      setStatus(modelReady ? "Idle. Start the microphone to begin." : "Idle. Waiting for the model.");
    }
  }
  function startTimer() {
    const nextDuration = timerDurationSeconds;
    setTimerRemainingSeconds(nextDuration);
    setTimerRunning(true);
  }

  function pauseTimer() {
    setTimerRunning(false);
  }

  function resetTimer() {
    setTimerRunning(false);
    setTimerRemainingSeconds(timerDurationSeconds);
  }

  function adjustTimerRemaining(deltaSeconds: number) {
    setTimerRemainingSeconds((current) => Math.max(0, current + deltaSeconds));
  }

  const target = useMemo(() => findTarget(targetValue), [targetValue]);
  const timerPercent = Math.min(100, Math.max(0, (timerRemainingSeconds / timerDurationSeconds) * 100));
  const timerNote = useMemo(() => {
    if (timerRemainingSeconds <= 0) {
      return "Timer complete.";
    }

    if (timerRunning) {
      if (!isListening) return "Start the microphone first, then the timer will only move while the instrument is detected.";
      if (!isAudible) return "Paused — audio too quiet.";
      return verdict.matched ? "Instrument detected. Timer is running." : `Waiting for ${target.label.toLowerCase()} to appear before the countdown moves.`;
    }

    if (!isListening) return "Start the microphone to let the timer listen for the instrument.";
    if (!isAudible) return "Paused — audio too quiet.";
    return verdict.matched
      ? `Instrument detected. Timer will keep moving for ${target.label}.`
      : `Waiting for ${target.label.toLowerCase()} to keep the timer moving.`;
  }, [isListening, isAudible, target.label, timerRemainingSeconds, timerRunning, verdict.matched]);

  return (
    <div className="grid min-h-[calc(100vh-4rem)] gap-6 p-6 lg:grid-cols-[1.05fr_0.95fr] lg:p-10">
      <section className="flex flex-col justify-between gap-8">
        <div className="space-y-6">
          <div className="inline-flex items-center gap-3 rounded-full border border-amber-200/20 bg-amber-200/10 px-4 py-2 text-xs font-semibold uppercase tracking-[0.28em] text-amber-100/90">
            Real-time pretrained classifier
          </div>

          <div className="space-y-4">
            <h1 className="max-w-2xl text-4xl font-semibold tracking-tight text-balance text-white md:text-6xl">
              Listen to the microphone and check whether the sound matches a chosen instrument.
            </h1>
            <p className="max-w-2xl text-base leading-7 text-stone-300 md:text-lg">
              This version uses a pretrained audio-classification model in the browser. It buffers a 3-second
              window, scores the audio classes, and compares the result against piano or human voice.
            </p>
          </div>

          <div className="inline-flex rounded-full border border-white/10 bg-black/20 p-1 text-sm text-stone-300">
            <button
              type="button"
              onClick={() => setActiveTab("classifier")}
              className={`rounded-full px-4 py-2 transition ${
                activeTab === "classifier" ? "bg-amber-300 text-slate-950" : "hover:text-white"
              }`}
            >
              Classifier
            </button>
            <button
              type="button"
              onClick={() => setActiveTab("timer")}
              className={`rounded-full px-4 py-2 transition ${
                activeTab === "timer" ? "bg-amber-300 text-slate-950" : "hover:text-white"
              }`}
            >
              Timer
            </button>
          </div>

          <div className="grid gap-3 sm:grid-cols-3">
            <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
              <p className="text-xs uppercase tracking-[0.24em] text-stone-400">Status</p>
              <p className="mt-3 text-lg font-medium text-white">{status}</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
              <p className="text-xs uppercase tracking-[0.24em] text-stone-400">Target</p>
              <p className="mt-3 text-lg font-medium text-white">{target.label}</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
              <p className="text-xs uppercase tracking-[0.24em] text-stone-400">Buffer</p>
              <p className="mt-3 text-lg font-medium text-white">{Math.round(bufferProgress * 100)}%</p>
            </div>
          </div>
        </div>

        {activeTab === "classifier" ? (
          <div className="flex flex-col gap-4 sm:flex-row">
            <button
              type="button"
              onClick={isListening ? stopListening : startListening}
              className="inline-flex min-h-12 items-center justify-center rounded-full bg-amber-300 px-6 text-sm font-semibold text-slate-950 transition-transform hover:-translate-y-0.5 hover:bg-amber-200 focus:outline-none focus:ring-2 focus:ring-amber-200 focus:ring-offset-2 focus:ring-offset-slate-950"
            >
              {isListening ? "Stop microphone" : "Start microphone"}
            </button>

            <label className="flex min-h-12 items-center gap-3 rounded-full border border-white/12 bg-black/20 px-4 text-sm text-stone-200">
              <span className="whitespace-nowrap text-stone-400">Target source</span>
              <select
                value={targetValue}
                onChange={(event) => setTargetValue(event.target.value as TargetValue)}
                className="h-10 flex-1 rounded-full border border-white/10 bg-white/8 px-4 text-sm text-white outline-none transition focus:border-amber-200/70"
              >
                {TARGET_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value} className="bg-slate-900">
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
          </div>
        ) : (
          <div className="space-y-4 rounded-[1.5rem] border border-white/10 bg-black/20 p-5">
            <div className="grid gap-3 sm:grid-cols-3">
              <label className="flex items-center gap-3 rounded-full border border-white/12 bg-white/5 px-4 py-3 text-sm text-stone-200">
                <span className="whitespace-nowrap text-stone-400">Minutes</span>
                <input
                  type="number"
                  min="0"
                  max="59"
                  value={timerMinutes}
                  onChange={(event) => setTimerMinutes(Number.parseInt(event.target.value || "0", 10) || 0)}
                  className="w-full bg-transparent text-right text-white outline-none"
                />
              </label>
              <label className="flex items-center gap-3 rounded-full border border-white/12 bg-white/5 px-4 py-3 text-sm text-stone-200">
                <span className="whitespace-nowrap text-stone-400">Seconds</span>
                <input
                  type="number"
                  min="0"
                  max="59"
                  value={timerSeconds}
                  onChange={(event) => setTimerSeconds(Number.parseInt(event.target.value || "0", 10) || 0)}
                  className="w-full bg-transparent text-right text-white outline-none"
                />
              </label>
              <label className="flex items-center gap-3 rounded-full border border-white/12 bg-white/5 px-4 py-3 text-sm text-stone-200">
                <span className="whitespace-nowrap text-stone-400">Target</span>
                <span className="truncate text-right text-white">{target.label}</span>
              </label>
            </div>

            <div className="flex flex-wrap gap-3">
              <button
                type="button"
                onClick={startTimer}
                className="inline-flex min-h-12 items-center justify-center rounded-full bg-amber-300 px-6 text-sm font-semibold text-slate-950 transition-transform hover:-translate-y-0.5 hover:bg-amber-200 focus:outline-none focus:ring-2 focus:ring-amber-200 focus:ring-offset-2 focus:ring-offset-slate-950"
              >
                {timerRunning ? "Restart timer" : "Start timer"}
              </button>
              <button
                type="button"
                onClick={pauseTimer}
                className="inline-flex min-h-12 items-center justify-center rounded-full border border-white/12 bg-white/5 px-6 text-sm font-semibold text-white transition hover:bg-white/10"
              >
                Pause
              </button>
              <button
                type="button"
                onClick={resetTimer}
                className="inline-flex min-h-12 items-center justify-center rounded-full border border-white/12 bg-white/5 px-6 text-sm font-semibold text-white transition hover:bg-white/10"
              >
                Reset
              </button>
            </div>

            <p className="text-sm leading-6 text-stone-300">{timerNote}</p>
          </div>
        )}

        {error ? (
          <p className="max-w-2xl rounded-2xl border border-rose-300/20 bg-rose-400/10 px-4 py-3 text-sm text-rose-100">
            {error}
          </p>
        ) : null}
      </section>

      <aside className="flex flex-col gap-6 rounded-[1.75rem] border border-white/10 bg-slate-950/50 p-6 shadow-inner shadow-black/30">
        {activeTab === "classifier" ? (
          <>
            <div className="space-y-4">
              <p className="text-xs uppercase tracking-[0.3em] text-stone-400">Current verdict</p>
              <div>
                <h2 className="text-3xl font-semibold tracking-tight text-white">{verdict.headline}</h2>
                <p className="mt-3 text-sm leading-6 text-stone-300">{verdict.detail}</p>
              </div>
            </div>

            <div className="space-y-3">
              <div className="flex items-center justify-between text-sm text-stone-300">
                <span>Confidence</span>
                <span>{verdict.confidence}%</span>
              </div>
              <div className="h-3 overflow-hidden rounded-full bg-white/10">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-amber-300 via-orange-300 to-cyan-300 transition-all duration-300"
                  style={{ width: `${Math.max(verdict.confidence, 6)}%` }}
                />
              </div>
            </div>

            <div className="space-y-2 mt-3">
              <p className="text-xs uppercase tracking-[0.3em] text-stone-400">Volume</p>
              <div className="h-2 w-full overflow-hidden rounded-full bg-white/10">
                <div
                  className="h-full rounded-full bg-emerald-400 transition-all duration-150"
                  style={{ width: `${Math.min(100, Math.round(rms * 1000))}%` }}
                />
              </div>
              <div className="flex items-center justify-between text-xs text-stone-400">
                <span>RMS</span>
                <span>{Math.round(rms * 1000)}%</span>
              </div>
            </div>

            <div className="rounded-2xl border border-white/8 bg-white/4 p-4 text-sm leading-6 text-stone-300">
              Top prediction: <span className="font-medium text-white">{titleCase(verdict.topLabel)}</span>
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              {predictions.length > 0 ? (
                predictions.map((prediction) => (
                  <div
                    key={prediction.label}
                    className="rounded-2xl border border-white/8 bg-white/4 p-4 text-stone-300"
                  >
                    <p className="text-sm font-medium text-white">{titleCase(prediction.label)}</p>
                    <p className="mt-2 text-xs leading-5 text-inherit/75">
                      {Math.round(prediction.score * 100)}% confidence
                    </p>
                  </div>
                ))
              ) : (
                <div className="rounded-2xl border border-white/8 bg-white/4 p-4 text-sm text-stone-300 sm:col-span-2">
                  Predictions will appear once the buffer fills and the model runs.
                </div>
              )}
            </div>

            <div className="rounded-2xl border border-cyan-200/15 bg-cyan-300/5 p-4 text-sm leading-6 text-stone-300">
              The model is quantized and runs locally in the browser. Piano maps to the piano and keyboard-related
              AudioSet labels, while voice maps to speech, singing, humming, and other human-vocal classes.
            </div>
          </>
        ) : (
          <>
            <div className="space-y-4">
              <p className="text-xs uppercase tracking-[0.3em] text-stone-400">Timer</p>
              <div>
                <h2 className="text-5xl font-semibold tracking-tight text-white">{formatDuration(timerRemainingSeconds)}</h2>
                <p className="mt-3 text-sm leading-6 text-stone-300">
                  This countdown only moves while the selected instrument is being detected by the microphone.
                </p>
              </div>
            </div>

            <div className="space-y-3">
              <div className="flex items-center justify-between text-sm text-stone-300">
                <span>Progress</span>
                <span>{Math.round(timerPercent)}%</span>
              </div>
              <div className="h-3 overflow-hidden rounded-full bg-white/10">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-cyan-300 via-amber-300 to-orange-300 transition-all duration-300"
                  style={{ width: `${Math.max(timerPercent, 4)}%` }}
                />
              </div>
            </div>

            <div className="space-y-3">
              <p className="text-xs uppercase tracking-[0.24em] text-stone-400">Adjust time</p>
              <div className="grid grid-cols-3 gap-2">
                <button
                  type="button"
                  onClick={() => adjustTimerRemaining(-60)}
                  className="rounded-lg border border-white/12 bg-white/5 px-2 py-2 text-xs font-semibold text-white transition hover:bg-white/10"
                >
                  −1 min
                </button>
                <button
                  type="button"
                  onClick={() => adjustTimerRemaining(-180)}
                  className="rounded-lg border border-white/12 bg-white/5 px-2 py-2 text-xs font-semibold text-white transition hover:bg-white/10"
                >
                  −3 mins
                </button>
                <button
                  type="button"
                  onClick={() => adjustTimerRemaining(-300)}
                  className="rounded-lg border border-white/12 bg-white/5 px-2 py-2 text-xs font-semibold text-white transition hover:bg-white/10"
                >
                  −5 mins
                </button>
                <button
                  type="button"
                  onClick={() => adjustTimerRemaining(60)}
                  className="rounded-lg border border-white/12 bg-white/5 px-2 py-2 text-xs font-semibold text-white transition hover:bg-white/10"
                >
                  +1 min
                </button>
                <button
                  type="button"
                  onClick={() => adjustTimerRemaining(180)}
                  className="rounded-lg border border-white/12 bg-white/5 px-2 py-2 text-xs font-semibold text-white transition hover:bg-white/10"
                >
                  +3 mins
                </button>
                <button
                  type="button"
                  onClick={() => adjustTimerRemaining(300)}
                  className="rounded-lg border border-white/12 bg-white/5 px-2 py-2 text-xs font-semibold text-white transition hover:bg-white/10"
                >
                  +5 mins
                </button>
              </div>
            </div>

            <div className="space-y-2 mt-3">
              <p className="text-xs uppercase tracking-[0.3em] text-stone-400">Volume</p>
              <div className="h-2 w-full overflow-hidden rounded-full bg-white/10">
                <div
                  className="h-full rounded-full bg-emerald-400 transition-all duration-150"
                  style={{ width: `${Math.min(100, Math.round(rms * 1000))}%` }}
                />
              </div>
              <div className="flex items-center justify-between text-xs text-stone-400">
                <span>RMS</span>
                <span>{Math.round(rms * 1000)}%</span>
              </div>
            </div>

            <div className="rounded-2xl border border-white/8 bg-white/4 p-4 text-sm leading-6 text-stone-300">
              Listening state: <span className="font-medium text-white">{isListening ? "Microphone active" : "Microphone off"}</span>
            </div>

            <div className="rounded-2xl border border-white/8 bg-white/4 p-4 text-sm leading-6 text-stone-300">
              Detection state: <span className="font-medium text-white">{verdict.matched ? `Matching ${target.label}` : `Waiting for ${target.label}`}</span>
            </div>

            <div className="rounded-2xl border border-cyan-200/15 bg-cyan-300/5 p-4 text-sm leading-6 text-stone-300">
              Start the microphone first, then the timer will only move while the instrument is detected.
            </div>
          </>
        )}
      </aside>
      {showComplete ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div className="absolute inset-0 bg-black/50" />
          <div className="relative z-10 max-w-md rounded-lg bg-slate-900/90 p-6 text-stone-200">
            <h3 className="text-2xl font-semibold text-white">Timer complete</h3>
            <p className="mt-2 text-sm">Your countdown has finished.</p>
            <div className="mt-4 flex justify-end">
              <button
                type="button"
                onClick={() => setShowComplete(false)}
                className="rounded-full bg-amber-300 px-4 py-2 text-sm font-semibold text-slate-950"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}

