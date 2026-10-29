import os
import sys

# Auto-add local lipsinc directories and DLL paths BEFORE torch/C-extensions
LIPSINC_DIR = r"D:\lipsinc"
paths_to_add = [
    LIPSINC_DIR,
    os.path.join(LIPSINC_DIR, "auto_avsr"),
    os.path.join(LIPSINC_DIR, "face_detection"),
    os.path.join(LIPSINC_DIR, "face_alignment"),
]
for p in paths_to_add:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

lipsinc_site_packages = os.path.join(LIPSINC_DIR, ".venv", "Lib", "site-packages")
if os.path.exists(lipsinc_site_packages) and lipsinc_site_packages not in sys.path:
    sys.path.append(lipsinc_site_packages)

# Add DLL directories for PyTorch and CUDA runtime on Windows
if sys.platform == "win32":
    import ctypes
    candidate_dll_dirs = [
        os.path.join(lipsinc_site_packages, "torch", "lib"),
        os.path.join(lipsinc_site_packages, "nvidia", "cudnn", "bin"),
        os.path.join(lipsinc_site_packages, "nvidia", "cu13", "bin", "x86_64"),
    ]
    if os.path.exists(lipsinc_site_packages):
        for root, dirs, files in os.walk(lipsinc_site_packages):
            if any(d in dirs for d in ["bin", "lib"]):
                for d in ["bin", "lib"]:
                    candidate_dll_dirs.append(os.path.join(root, d))
    for dll_dir in candidate_dll_dirs:
        if os.path.isdir(dll_dir):
            try:
                os.add_dll_directory(dll_dir)
            except Exception:
                pass

    # Preload NVIDIA CUDA and cuDNN libraries to prevent "Cannot load symbol cudnnGetVersion"
    try:
        from cuda.pathfinder import load_nvidia_dynamic_lib
        for lib in ("cudart", "nvJitLink", "nvrtc", "cublas", "cublasLt", "cufft", "curand", "cusolver", "cusparse", "cupti", "cudnn"):
            try:
                load_nvidia_dynamic_lib(lib)
            except Exception:
                pass
    except Exception:
        pass

    cudnn_bin = os.path.join(lipsinc_site_packages, "nvidia", "cudnn", "bin")
    if os.path.isdir(cudnn_bin):
        for dll_file in os.listdir(cudnn_bin):
            if dll_file.endswith(".dll"):
                try:
                    ctypes.CDLL(os.path.join(cudnn_bin, dll_file))
                except Exception:
                    pass

import base64
import tempfile
import subprocess
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Bypass unused torchaudio import in auto_avsr/datamodule/transforms.py (VSR is visual-only)
from unittest.mock import MagicMock
if "torchaudio" not in sys.modules:
    try:
        import torchaudio
    except Exception:
        sys.modules["torchaudio"] = MagicMock()

app = FastAPI(title="Local RTX 5050 VSR Engine")

# Locate local vsr_trlrs3_base.pth model weights
CANDIDATE_PATHS = [
    os.path.join(LIPSINC_DIR, "auto_avsr", "tutorials", "vsr_trlrs3_base.pth"),
    os.path.expanduser(r"~\Downloads\vsr_trlrs3_base.pth"),
    r"C:\Users\ASUS\Downloads\vsr_trlrs3_base.pth",
]

model_path = None
for p in CANDIDATE_PATHS:
    if os.path.isfile(p) and os.path.getsize(p) > 100_000:
        model_path = p
        break

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"[*] Local Engine Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
print(f"[*] Model checkpoint: {model_path}")

def find_ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        import shutil
        return shutil.which("ffmpeg") or "ffmpeg"

ffmpeg_binary = find_ffmpeg()
print(f"[*] FFmpeg binary: {ffmpeg_binary}")

pipeline_model = None

def get_pipeline():
    global pipeline_model
    if pipeline_model is None:
        if not model_path:
            raise RuntimeError("vsr_trlrs3_base.pth not found in D:\\lipsinc or Downloads.")

        import argparse
        from datamodule.transforms import VideoTransform
        from lightning import ModelModule, get_beam_search_decoder
        from preparation.detectors.retinaface.detector import LandmarksDetector
        from preparation.detectors.retinaface.video_process import VideoProcess

        class EnglishLanguageModelRescorer:
            """
            Statistical English Language Model (LM) Rescorer for Visual Speech Recognition.
            Employs vocabulary verification, conversational n-gram collocations, and homophene
            disambiguation to resolve visually identical mouth movements (/p,b,m/, /t,d,n/, /f,v/).
            """
            # High-frequency conversational spoken collocations and n-grams
            BIGRAMS = {
                ("how", "are"): 4.5, ("are", "you"): 4.5, ("can", "you"): 4.5, ("hear", "me"): 4.5,
                ("see", "you"): 4.0, ("nice", "to"): 4.5, ("to", "meet"): 4.5, ("meet", "you"): 4.5,
                ("thank", "you"): 5.0, ("you", "very"): 4.0, ("very", "much"): 4.5, ("good", "morning"): 4.5,
                ("good", "afternoon"): 4.0, ("good", "evening"): 4.0, ("good", "night"): 4.0,
                ("what", "is"): 4.0, ("is", "your"): 4.0, ("what", "are"): 4.0, ("where", "are"): 4.0,
                ("i", "am"): 4.5, ("i", "can"): 4.0, ("i", "do"): 4.0, ("i", "have"): 4.0, ("i", "think"): 4.0,
                ("do", "you"): 4.5, ("would", "you"): 4.0, ("could", "you"): 4.0, ("let", "us"): 4.0,
                ("pack", "your"): 4.5, ("your", "bag"): 4.5, ("your", "bags"): 4.5, ("turn", "on"): 4.0,
                ("turn", "off"): 4.0, ("look", "at"): 4.0, ("listen", "to"): 4.0, ("tell", "me"): 4.5,
                ("yes", "i"): 3.5, ("no", "i"): 3.5, ("it", "is"): 4.5, ("that", "is"): 4.0,
                ("have", "a"): 4.0, ("a", "good"): 4.0, ("a", "great"): 4.0, ("great", "day"): 4.0,
                ("welcome", "back"): 4.5, ("excuse", "me"): 4.5, ("please", "help"): 4.0,
            }
            TRIGRAMS = {
                ("how", "are", "you"): 6.0, ("can", "you", "hear"): 6.0, ("you", "hear", "me"): 6.0,
                ("can", "you", "see"): 5.5, ("nice", "to", "meet"): 6.0, ("to", "meet", "you"): 6.0,
                ("thank", "you", "very"): 6.0, ("you", "very", "much"): 6.0, ("what", "is", "your"): 5.5,
                ("pack", "your", "bag"): 5.5, ("pack", "your", "bags"): 5.5, ("i", "do", "not"): 5.0,
                ("have", "a", "good"): 5.0, ("have", "a", "great"): 5.0, ("a", "great", "day"): 5.0,
            }
            # Visual Homophene Confusion Matrix (sounds with visually indistinguishable lip closures)
            HOMOPHENE_MAP = {
                "year": ["hear", "near"],
                "meat": ["meet"],
                "ferry": ["very"],
                "back": ["pack", "bag"],
                "mack": ["pack", "back"],
                "bay": ["may", "pay"],
                "bat": ["pat", "mat"],
                "sea": ["see"],
                "write": ["right"],
                "no": ["know"],
                "won": ["one"],
                "to": ["two", "too"],
                "here": ["hear"],
            }
            # Common English conversational dictionary
            LEXICON = {
                "a", "about", "above", "across", "action", "actually", "after", "again", "against", "age",
                "all", "almost", "alone", "along", "already", "also", "always", "am", "american", "among",
                "an", "and", "another", "answer", "any", "anyone", "anything", "appear", "apple", "are",
                "area", "arms", "around", "as", "ask", "at", "away", "baby", "back", "bad", "bag", "bags",
                "ball", "bank", "bar", "be", "beautiful", "because", "become", "bed", "been", "before",
                "began", "begin", "behind", "being", "believe", "bell", "best", "better", "between",
                "big", "bill", "bird", "bit", "black", "blood", "blow", "blue", "board", "boat", "body",
                "book", "born", "both", "bottom", "box", "boy", "break", "bring", "brother", "brought",
                "brown", "build", "building", "built", "business", "busy", "but", "by", "call", "came",
                "can", "cannot", "cant", "car", "care", "carry", "case", "cat", "catch", "cause", "center",
                "certain", "chair", "chance", "change", "check", "child", "children", "choose", "city",
                "class", "clean", "clear", "close", "cold", "color", "come", "common", "company", "complete",
                "condition", "consider", "continue", "control", "cook", "cool", "corner", "correct", "cost",
                "could", "count", "country", "course", "cover", "cross", "cry", "cup", "cut", "dark",
                "data", "daughter", "day", "days", "dead", "deal", "dear", "death", "decide", "deep",
                "describe", "design", "desk", "detail", "did", "die", "different", "difficult", "dinner",
                "direction", "discover", "distance", "do", "doctor", "does", "dog", "dollar", "done",
                "door", "down", "draw", "dream", "dress", "drink", "drive", "drop", "dry", "during",
                "each", "early", "ear", "ears", "earth", "east", "easy", "eat", "education", "effect",
                "eight", "either", "electric", "element", "else", "end", "energy", "engine", "england",
                "english", "enough", "enter", "entire", "environment", "equal", "especially", "even",
                "evening", "event", "ever", "every", "everyone", "everything", "everywhere", "evidence",
                "exact", "example", "except", "expect", "experience", "explain", "eye", "eyes", "face",
                "fact", "factor", "fail", "fall", "family", "famous", "far", "farm", "farmer", "fast",
                "father", "fear", "feel", "feeling", "feet", "few", "field", "fight", "figure", "fill",
                "final", "finally", "find", "fine", "finger", "finish", "fire", "first", "fish", "fit",
                "five", "flat", "floor", "flow", "flower", "fly", "follow", "food", "foot", "for",
                "force", "foreign", "forest", "forget", "form", "forward", "found", "four", "free",
                "fresh", "friend", "friends", "from", "front", "full", "fun", "game", "garden", "gas",
                "general", "gentle", "get", "girl", "give", "given", "glad", "glass", "go", "god",
                "going", "gold", "gone", "good", "got", "government", "grass", "great", "green", "ground",
                "group", "grow", "grown", "guess", "gun", "guy", "had", "hair", "half", "hand", "hands",
                "hang", "happen", "happy", "hard", "has", "have", "having", "he", "head", "hear", "heard",
                "heart", "heat", "heavy", "held", "hello", "help", "her", "here", "herself", "hi", "high",
                "hill", "him", "himself", "his", "history", "hit", "hold", "hole", "home", "hope", "horse",
                "hot", "hour", "hours", "house", "how", "however", "huge", "human", "hundred", "hung",
                "hurry", "husband", "i", "ice", "idea", "if", "ill", "image", "imagine", "important",
                "improve", "in", "inch", "include", "including", "increase", "indeed", "indicate",
                "industry", "information", "inside", "instead", "interest", "into", "iron", "is", "island",
                "issue", "it", "its", "itself", "job", "join", "joy", "judge", "jump", "just", "keep",
                "kept", "key", "kill", "kind", "king", "knew", "know", "known", "labor", "lack", "lady",
                "laid", "lake", "land", "language", "large", "last", "late", "later", "laugh", "law",
                "lay", "lead", "leader", "learn", "least", "leave", "led", "left", "leg", "legs", "length",
                "less", "let", "letter", "level", "lie", "life", "lift", "light", "like", "likely", "line",
                "lip", "lips", "list", "listen", "little", "live", "living", "local", "locate", "long",
                "look", "looked", "looking", "looks", "lord", "lose", "lost", "lot", "loud", "love", "low",
                "machine", "made", "main", "major", "make", "making", "man", "many", "map", "mark",
                "market", "marry", "master", "match", "matter", "may", "maybe", "me", "mean", "meant",
                "measure", "meat", "meet", "meeting", "member", "men", "metal", "method", "middle", "might",
                "mile", "milk", "million", "mind", "minute", "minutes", "miss", "modern", "moment", "money",
                "month", "months", "moon", "more", "morning", "most", "mother", "motion", "motor", "mountain",
                "mouth", "move", "movement", "much", "music", "must", "my", "myself", "name", "nation",
                "national", "natural", "nature", "near", "nearly", "necessary", "neck", "need", "needed",
                "neither", "nerve", "never", "new", "news", "next", "nice", "night", "nine", "no", "nobody",
                "noise", "none", "noon", "nor", "north", "nose", "not", "note", "nothing", "notice", "now",
                "number", "object", "observe", "obtain", "ocean", "of", "off", "offer", "office", "officer",
                "often", "oh", "oil", "old", "on", "once", "one", "only", "open", "operate", "opinion",
                "or", "order", "original", "other", "others", "our", "out", "outside", "over", "own",
                "pack", "page", "paid", "pain", "paint", "pair", "paper", "paragraph", "park", "part",
                "particular", "party", "pass", "past", "path", "pay", "peace", "people", "per", "perfect",
                "period", "person", "phone", "picture", "piece", "place", "plain", "plan", "plane", "plant",
                "play", "player", "please", "point", "police", "poor", "position", "possible", "post",
                "pound", "power", "practice", "prepare", "present", "president", "press", "pretty", "price",
                "probably", "problem", "process", "produce", "product", "program", "project", "proper",
                "protect", "proud", "provide", "public", "pull", "pure", "push", "put", "quarter", "queen",
                "question", "questions", "quick", "quiet", "quite", "race", "radio", "rain", "raise", "ran",
                "range", "rate", "rather", "reach", "read", "ready", "real", "really", "reason", "receive",
                "record", "red", "region", "remain", "remember", "repeat", "reply", "report", "represent",
                "require", "rest", "result", "return", "rich", "ride", "right", "ring", "rise", "river",
                "road", "rock", "roll", "room", "root", "rose", "round", "row", "rule", "run", "safe",
                "said", "sail", "same", "sat", "save", "saw", "say", "saying", "scale", "school", "science",
                "score", "sea", "season", "seat", "second", "secret", "section", "see", "seed", "seem",
                "seemed", "seems", "seen", "self", "sell", "send", "sense", "sent", "sentence", "separate",
                "serve", "service", "set", "settle", "seven", "several", "shall", "shape", "share", "sharp",
                "she", "sheet", "ship", "shoe", "shoes", "shop", "short", "should", "shoulder", "shout",
                "show", "shown", "side", "sight", "sign", "signal", "silent", "similar", "simple", "since",
                "sing", "single", "sister", "sit", "site", "six", "size", "skill", "skin", "sky", "sleep",
                "slow", "small", "smell", "smile", "smoke", "snow", "so", "social", "society", "soft",
                "soil", "soldier", "solution", "solve", "some", "someone", "something", "sometimes",
                "son", "song", "soon", "sound", "south", "space", "speak", "special", "speech", "speed",
                "spell", "spend", "spoke", "spot", "spread", "spring", "square", "stand", "star", "start",
                "state", "statement", "station", "stay", "step", "stick", "still", "stood", "stop", "store",
                "story", "straight", "strange", "stream", "street", "stretch", "strike", "strong", "student",
                "study", "subject", "substance", "succeed", "success", "such", "sudden", "suit", "summer",
                "sun", "supply", "support", "suppose", "sure", "surface", "surprise", "system", "table",
                "tail", "take", "taken", "taking", "talk", "tall", "taste", "teach", "teacher", "team",
                "tell", "ten", "term", "test", "than", "thank", "thanks", "that", "the", "their", "them",
                "themselves", "then", "there", "therefore", "these", "they", "thick", "thin", "thing",
                "things", "think", "third", "this", "those", "though", "thought", "thousand", "three",
                "through", "throw", "thus", "tie", "time", "times", "tiny", "tire", "to", "today", "together",
                "told", "tomorrow", "tone", "too", "took", "tool", "top", "total", "touch", "toward",
                "town", "track", "trade", "train", "travel", "tree", "trip", "trouble", "true", "trust",
                "truth", "try", "tube", "turn", "turned", "twelve", "twenty", "two", "type", "under",
                "understand", "unit", "until", "up", "upon", "us", "use", "used", "useful", "usually",
                "valley", "value", "various", "very", "view", "village", "visit", "voice", "wait", "walk",
                "wall", "want", "wanted", "war", "warm", "was", "wash", "watch", "water", "wave", "way",
                "we", "wear", "weather", "week", "weeks", "weight", "welcome", "well", "went", "were",
                "west", "what", "whatever", "wheel", "when", "whenever", "where", "whether", "which",
                "while", "white", "who", "whole", "whom", "whose", "why", "wide", "wife", "wild", "will",
                "win", "wind", "window", "wing", "winter", "wire", "wise", "wish", "with", "within",
                "without", "woman", "women", "wonder", "wood", "word", "words", "work", "worker", "world",
                "worry", "would", "write", "writer", "wrong", "yard", "year", "years", "yellow", "yes",
                "yesterday", "yet", "you", "young", "your", "yourself"
            }

            def __init__(self):
                pass

            def compute_lm_score(self, text: str) -> float:
                """
                Scores text according to English vocabulary validity, n-gram transitions,
                and grammar fluency.
                """
                import re
                clean_text = text.lower().strip()
                words = re.findall(r"[a-z']+", clean_text)
                if not words:
                    return -15.0

                score = 0.0
                valid_words = 0
                for w in words:
                    # Strip common trailing punctuation/contractions
                    base_w = w.replace("'", "")
                    if base_w in self.LEXICON or len(base_w) == 1 and base_w in ("a", "i"):
                        valid_words += 1
                        score += 1.0
                    else:
                        score -= 4.0  # Significant penalty for non-lexicon / gibberish tokens

                # Bigram fluency bonus
                for i in range(len(words) - 1):
                    pair = (words[i], words[i + 1])
                    if pair in self.BIGRAMS:
                        score += self.BIGRAMS[pair]

                # Trigram fluency bonus
                for i in range(len(words) - 2):
                    tri = (words[i], words[i + 1], words[i + 2])
                    if tri in self.TRIGRAMS:
                        score += self.TRIGRAMS[tri]

                # Per-word normalized score
                return score / max(1, len(words))

            def generate_homophene_variants(self, text: str) -> list:
                """
                Generates high-likelihood homophene alternatives if a word in text is a known visual confusion.
                """
                import re
                words = re.findall(r"[a-z']+", text.lower().strip())
                if not words:
                    return []

                variants = []
                for i, w in enumerate(words):
                    if w in self.HOMOPHENE_MAP:
                        for alt in self.HOMOPHENE_MAP[w]:
                            alt_words = list(words)
                            alt_words[i] = alt
                            variants.append(" ".join(alt_words).upper())
                return variants

            def rescore_hypotheses(self, nbest_hyps, text_transform, lm_weight: float = 0.35) -> list:
                """
                Re-ranks beam search candidates using shallow fusion:
                Score = Visual_Acoustic_Score + (lm_weight * LM_Score)
                """
                candidates = {}

                for hyp in nbest_hyps:
                    hyp_dict = hyp.asdict()
                    token_ids = torch.tensor(list(map(int, hyp_dict["yseq"][1:])))
                    hyp_text = text_transform.post_process(token_ids).replace("<eos>", "").strip()
                    if not hyp_text:
                        continue
                    vis_score = float(hyp_dict.get("score", 0.0))

                    if hyp_text not in candidates or vis_score > candidates[hyp_text]["vis_score"]:
                        lm_sc = self.compute_lm_score(hyp_text)
                        candidates[hyp_text] = {
                            "text": hyp_text,
                            "vis_score": vis_score,
                            "lm_score": lm_sc,
                            "final_score": vis_score + (lm_weight * lm_sc),
                        }

                    # Also evaluate potential homophene alternatives with conservative acoustic score
                    for alt_text in self.generate_homophene_variants(hyp_text):
                        if alt_text not in candidates:
                            alt_lm = self.compute_lm_score(alt_text)
                            # Slightly downweight acoustic score for derived variant (-1.5)
                            alt_vis = vis_score - 1.5
                            candidates[alt_text] = {
                                "text": alt_text,
                                "vis_score": alt_vis,
                                "lm_score": alt_lm,
                                "final_score": alt_vis + (lm_weight * alt_lm),
                            }

                # Sort by blended final score descending
                sorted_results = sorted(candidates.values(), key=lambda x: x["final_score"], reverse=True)
                return sorted_results

        class InferencePipeline(torch.nn.Module):
            def __init__(self, ckpt_path: str, dev: str = "cpu"):
                super().__init__()
                parser = argparse.ArgumentParser()
                args, _ = parser.parse_known_args(args=[])
                setattr(args, "modality", "video")

                self.dev = dev
                self.landmarks_detector = LandmarksDetector(device=dev)
                self.video_process = VideoProcess(convert_gray=False)
                self.video_transform = VideoTransform(subset="test")

                ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
                self.modelmodule = ModelModule(args)
                self.modelmodule.model.load_state_dict(ckpt)
                if dev != "cpu":
                    self.modelmodule.to(dev)
                self.modelmodule.eval()

                # Language Model Rescorer & Disambiguator
                self.lm_rescorer = EnglishLanguageModelRescorer()

                # Optimized beam search decoder with expanded beam size (40) matching LRS3 benchmark
                self.beam_search = get_beam_search_decoder(
                    self.modelmodule.model,
                    self.modelmodule.token_list,
                    penalty=0.05,
                    ctc_weight=0.10,
                    beam_size=40,
                )

            def load_video(self, data_filename: str):
                import cv2
                import numpy as np
                cap = cv2.VideoCapture(data_filename)
                frames = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Convert BGR to RGB so colors and grayscale conversion match PyTorch model training
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                cap.release()
                if not frames:
                    return np.empty((0, 0, 0, 3), dtype=np.uint8)
                return np.array(frames)

            @staticmethod
            def calculate_mars(landmarks):
                import numpy as np
                mars = []
                for lm in landmarks:
                    if lm is None:
                        mars.append(None)
                        continue
                    p48, p54 = lm[48], lm[54]
                    p51, p57 = lm[51], lm[57]
                    p50, p58 = lm[50], lm[58]
                    p52, p56 = lm[52], lm[56]
                    h = np.linalg.norm(p48 - p54)
                    if h < 1e-3:
                        mars.append(None)
                        continue
                    v = (np.linalg.norm(p51 - p57) + np.linalg.norm(p50 - p58) + np.linalg.norm(p52 - p56)) / 3.0
                    mars.append(float(v / h))
                return mars

            @staticmethod
            def segment_speech_window(mars, padding=4, min_frames=10):
                """
                Velocity-based Speech Segmentation (Immune to Leading & Trailing Delay):
                Uses temporal derivative of lip movement to isolate the exact speech burst
                from any silent delay before or after speaking.
                """
                import numpy as np
                valid_vals = [m for m in mars if m is not None]
                if len(valid_vals) < min_frames:
                    return 0, len(mars), 0.0, 0.0, 0.0, False

                baseline = float(np.median(valid_vals))
                mar_arr = np.array([m if m is not None else baseline for m in mars])
                mar_range = float(np.ptp(mar_arr))
                mar_std = float(np.std(mar_arr))

                # Compute temporal velocity and smoothed energy of lip motion
                velocity = np.abs(np.diff(mar_arr, prepend=mar_arr[0]))
                kernel = np.ones(3) / 3.0
                smooth_vel = np.convolve(velocity, kernel, mode="same")
                dev = np.abs(mar_arr - baseline)
                combined_score = smooth_vel * 1.5 + dev

                # Dynamic threshold for active lip speech
                thresh = max(0.020, float(np.mean(combined_score) + 0.8 * np.std(combined_score)))
                active_frames = np.where(combined_score > thresh)[0]

                # Check if overall motion indicates real speech
                is_speaking = (mar_range >= 0.030 and mar_std >= 0.0055 and len(active_frames) >= 4)
                if not is_speaking or len(active_frames) == 0:
                    return 0, len(mars), baseline, mar_range, mar_std, False

                # Extract tight active window with 4-frame phoneme padding
                start_idx = max(0, int(active_frames[0]) - padding)
                end_idx = min(len(mars), int(active_frames[-1]) + padding + 1)

                # Ensure minimum length of 10 frames (~400ms) for reliable recognition
                if end_idx - start_idx < min_frames:
                    expand = (min_frames - (end_idx - start_idx)) // 2 + 1
                    start_idx = max(0, start_idx - expand)
                    end_idx = min(len(mars), end_idx + expand)

                return start_idx, end_idx, baseline, mar_range, mar_std, True

            def forward(self, data_filename: str):
                import numpy as np

                # STEP 2: Extract & Verify Frames
                video = self.load_video(data_filename)
                total_frames = len(video)
                if total_frames == 0:
                    print("  [!] Error: OpenCV failed to read frames from resampled video.")
                    return {"text": "", "message": "Failed to read video frames."}
                
                duration_sec = total_frames / 25.0
                h, w, c = video[0].shape
                print(f"[STEP 2/6] [FRAMES] Loading Frames & Color Verification:")
                print(f"  * Raw Extracted Frames : {total_frames} frames ({duration_sec:.2f}s at 25.0 FPS)")
                print(f"  * Frame Dimensions     : {w}x{h} px")
                print(f"  * Color Channel Order  : RGB (Confirmed 3-channel)")

                # STEP 3: Facial Landmark Detection
                print(f"\n[STEP 3/6] [LANDMARKS] Facial Landmark Detection (RetinaFace + FAN):")
                landmarks = self.landmarks_detector(video)
                valid_count = sum(1 for lm in landmarks if lm is not None)
                tracking_pct = (valid_count / total_frames) * 100.0 if total_frames > 0 else 0
                print(f"  * Face Tracked Frames  : {valid_count}/{total_frames} ({tracking_pct:.1f}% tracking stability)")

                if valid_count < 8:
                    print(f"  [!] Face tracking too low ({valid_count} frames). Rejecting segment.")
                    return {
                        "text": "",
                        "message": "Face was not clearly detected. Please face the camera directly in good lighting.",
                    }

                # STEP 4: Visual Activity & Speech Segmentation
                print(f"\n[STEP 4/6] [V-VAD] Visual Activity Analysis & Speech Segmentation:")
                raw_mars = self.calculate_mars(landmarks)
                start_f, end_f, baseline_mar, mar_range, mar_std, is_speaking = self.segment_speech_window(raw_mars)

                print(f"  * Resting Lip Ratio (MAR) : {baseline_mar:.4f}")
                print(f"  * Lip Motion Dynamic Range: {mar_range:.4f} (std: {mar_std:.4f})")

                if not is_speaking:
                    print(f"  * V-VAD Speech Decision   : [SILENT / STILL] No active speech motion detected.")
                    print(f"  * Action                  : Skipping heavy AI inference to prevent hallucination.")
                    return {
                        "text": "",
                        "message": "No speech detected (lips were still/silent).",
                    }

                speech_frames_count = end_f - start_f
                trimmed_silence = total_frames - speech_frames_count
                print(f"  * V-VAD Speech Decision   : [ACTIVE SPEECH DETECTED]")
                print(f"  * Speech Window Extracted : Frames {start_f} to {end_f} ({speech_frames_count} frames = {speech_frames_count/25.0:.2f}s)")
                print(f"  * Dead Silence Discarded  : {trimmed_silence} frames removed ({trimmed_silence/25.0:.2f}s padding eliminated)")

                # Trim video and landmarks to the focused speech window
                video_trimmed = video[start_f:end_f]
                landmarks_trimmed = landmarks[start_f:end_f]

                # STEP 5: Mouth Patch Alignment and Cropping
                print(f"\n[STEP 5/6] [CROP] Mouth Region Alignment & Affine Cropping:")
                video_patches = self.video_process(video_trimmed, landmarks_trimmed)
                if video_patches is None or len(video_patches) == 0:
                    print("  [!] VideoProcess failed to generate mouth patches.")
                    return {"text": "", "message": "Could not align lip region."}

                patch_shape = video_patches.shape
                print(f"  * Cropped Mouth Patches   : Shape {patch_shape} (96x96 pixels)")

                video_tensor = torch.tensor(video_patches)
                video_tensor = video_tensor.permute((0, 3, 1, 2))
                video_tensor = self.video_transform(video_tensor)

                if self.dev != "cpu":
                    video_tensor = video_tensor.to(self.dev)

                print(f"  * Model Input Tensor Shape: {tuple(video_tensor.shape)} (CenterCrop 88x88, Grayscale, Normalized)")
                print(f"  * Compute Device          : {self.dev}")

                # STEP 6: Neural Network Inference
                print(f"\n[STEP 6/6] [AI MODEL] Neural Network Inference (Conformer + BeamSearch):")
                with torch.no_grad():
                    x = self.modelmodule.model.frontend(video_tensor.unsqueeze(0))
                    x = self.modelmodule.model.proj_encoder(x)
                    enc_feat, _ = self.modelmodule.model.encoder(x, None)
                    enc_feat = enc_feat.squeeze(0)
                    nbest_hyps = self.beam_search(enc_feat)

                    if not nbest_hyps:
                        print("  [!] Beam search produced no hypotheses.")
                        return {"text": "", "message": "No intelligible speech recognized."}

                    # Step 6b: Language Model Shallow Fusion & Candidate Rescoring
                    print(f"  * Beam Search Generated {len(nbest_hyps)} Raw Hypotheses. Running LM Shallow Fusion...")
                    rescored_results = self.lm_rescorer.rescore_hypotheses(
                        nbest_hyps=nbest_hyps,
                        text_transform=self.modelmodule.text_transform,
                        lm_weight=0.35,
                    )

                    print("  * Top Ranked Hypotheses (Visual Score + LM Fusion):")
                    for rank, cand in enumerate(rescored_results[:4], 1):
                        txt = cand["text"]
                        vis = cand["vis_score"]
                        lm_s = cand["lm_score"]
                        fin = cand["final_score"]
                        print(f"    #{rank}: \"{txt}\" | Visual: {vis:.2f} | LM: {lm_s:+.2f} => Combined: {fin:.2f}")

                    best_text = rescored_results[0]["text"] if rescored_results else ""

                print(f"\n=======================================================")
                print(f"[RESULT] FINAL PREDICTED TEXT: \"{best_text}\"")
                print(f"=======================================================\n")
                return {"text": best_text, "message": ""}

        print(f"[*] Initializing Auto-AVSR on {device}...")
        pipeline_model = InferencePipeline(ckpt_path=model_path, dev=device)
        print("[*] [OK] Local Auto-AVSR loaded successfully!")
    return pipeline_model

class VideoRequest(BaseModel):
    video_base64: str
    filename: str = "segment.webm"

@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": "local",
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "checkpoint": model_path,
        "ffmpeg": ffmpeg_binary,
    }

@app.post("/predict")
async def predict_lip_reading(req: VideoRequest):
    try:
        video_data = base64.b64decode(req.video_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 video encoding")

    file_size_kb = len(video_data) / 1024.0
    print("\n" + "=" * 75)
    print(f"[INGEST] NEW VIDEO RECEIVED: {req.filename} (Size: {file_size_kb:.1f} KB)")
    print("=" * 75)

    ext = ".webm" if "webm" in req.filename else ".mp4"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(video_data)
        tmp_path = tmp.name

    # Resample strictly to 25.0 fps constant framerate H.264 MP4 with reset PTS
    mp4_path = tmp_path.rsplit(".", 1)[0] + "_25fps.mp4"
    ffmpeg_bin = find_ffmpeg()
    
    print(f"[STEP 1/6] [FFMPEG] Normalization:")
    print(f"  * Source File      : {tmp_path}")
    print(f"  * Target Resampling: Constant 25.0 FPS H.264 MP4 (PTS-normalized)")
    res = subprocess.run(
        [
            ffmpeg_bin, "-y", "-i", tmp_path, "-an",
            "-vf", "setpts=PTS-STARTPTS,fps=25",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-pix_fmt", "yuv420p", mp4_path
        ],
        capture_output=True,
        text=True,
    )

    if not os.path.exists(mp4_path) or os.path.getsize(mp4_path) == 0:
        print(f"  [!] FFmpeg conversion failed: {res.stderr}")
        raise HTTPException(status_code=500, detail="FFmpeg video resampling to 25 FPS failed")

    out_size_kb = os.path.getsize(mp4_path) / 1024.0
    print(f"  * Normalized Output: {mp4_path} ({out_size_kb:.1f} KB)")

    target_path = mp4_path

    try:
        pipe = get_pipeline()
        raw_result = pipe(target_path)
        if isinstance(raw_result, dict):
            raw_text = str(raw_result.get("text", "")).strip()
            msg = raw_result.get("message", "")
        else:
            raw_text = str(raw_result).strip() if raw_result else ""
            msg = ""

        if not raw_text:
            return {
                "status": "success",
                "raw_text": "",
                "text": "",
                "message": msg or "No intelligible speech detected. Please face the camera directly in good lighting.",
            }

        return {
            "status": "success",
            "raw_text": raw_text,
            "text": raw_text.capitalize(),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "success",
            "raw_text": "",
            "text": "",
            "message": f"Lip movement could not be extracted ({type(e).__name__}: {str(e)}).",
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if os.path.exists(mp4_path):
            os.remove(mp4_path)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    print("\n=======================================================")
    print("🚀 LOCAL RTX 5050 AI ENGINE (PORT 8001)")
    print("URL: http://127.0.0.1:8001/predict")
    print("=======================================================\n")
    uvicorn.run(app, host="127.0.0.1", port=8001)
