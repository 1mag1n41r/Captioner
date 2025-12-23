import os
import json
from flask import Flask, request, render_template
from PIL import Image
import torch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------------------
# Load config
# -----------------------------------------
with open("config.json", "r") as f:
    cfg = json.load(f)

PHI_MODEL = cfg["phi_model"]
TOP_K     = cfg["topk_labels"]

device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------------------
# Load EfficientNet
# -----------------------------------------
weights = EfficientNet_B0_Weights.DEFAULT
effnet = efficientnet_b0(weights=weights).to(device).eval()
preprocess = weights.transforms()
categories = weights.meta["categories"]


# -----------------------------------------
# Load Phi-1.5
# -----------------------------------------
tok = AutoTokenizer.from_pretrained(PHI_MODEL)
phi = AutoModelForCausalLM.from_pretrained(
    PHI_MODEL,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device).eval()


# -----------------------------------------
# Functions
# -----------------------------------------
def get_labels(img_path, k=TOP_K):
    img = Image.open(img_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = effnet(x)
        probs = torch.softmax(logits, dim=1)[0]

    topk = torch.topk(probs, k)
    labels = [categories[i] for i in topk.indices]
    scores = topk.values.tolist()
    return labels, scores


# 6) Tag cleaning + caption generation (matching your bottom code)
DOG_BREED_KEYWORDS = [
    "retriever", "terrier", "hound", "spaniel", "shepherd", "poodle",
    "collie", "bulldog", "mastiff", "pinscher", "chihuahua", "beagle",
    "pug", "dalmatian", "rottweiler", "doberman", "labrador", "malinois",
    "ridgeback", "husky", "wolfhound", "greyhound", "setter", "pointer",
    "papillon", "spitz"
]

BAD_TAGS = {
    "shovel", "crutch", "band aid", "toilet tissue",
    "digital clock", "hair spray", "ice lolly",
    "espresso", "packet", "envelope"
}

def clean_tags(labels):
    cleaned = []
    seen = set()
    for x in labels:
        lx = x.lower().strip()
        if lx in BAD_TAGS:
            continue
        is_dog = any(k in lx for k in DOG_BREED_KEYWORDS)
        if is_dog:
            lx = "dog"
        if lx not in seen:
            seen.add(lx)
            cleaned.append(lx)
    return cleaned if cleaned else labels[:2]


@torch.no_grad()
def generate_caption(final_tags, max_new_tokens=22, temperature=0.9):
    """
    final_tags: already cleaned / edited tags (list of strings)
    """
    tags = ", ".join(final_tags)

    prompt = (
        "You write short, cozy, creative Instagram captions.\n"
        "You are soft, warm, dreamy, and gentle.\n"
        "Do NOT list the tags or describe the photo literally.\n"
        "Write ONE caption under 18 words, emotional and aesthetic.\n\n"
        f"Tags: {tags}\n"
        "Caption:"
    )

    inputs = tok(prompt, return_tensors="pt").to(device)
    out = phi.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tok.eos_token_id,
    )

    text = tok.decode(out[0], skip_special_tokens=True)

    if "Caption:" in text:
        text = text.split("Caption:", 1)[1].strip()

    if "." in text:
        text = text.split(".", 1)[0].strip() + "."

    if not text:
        text = "Soft moments in quiet light."

    return text

# -----------------------------------------
# Flask Web App
# -----------------------------------------
UPLOAD_DIR = "static/uploaded"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)


from werkzeug.utils import secure_filename
import uuid

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        stage = request.form.get("stage", "upload")  # "upload" or "caption"

        # ----------------------------
        # Stage 1: upload -> generate tags
        # ----------------------------
        if stage == "upload":
            file = request.files.get("image")
            if file is None or file.filename == "":
                return render_template("index.html", error="Please choose an image.")

            # safer unique filename
            ext = os.path.splitext(file.filename)[1].lower()
            base = secure_filename(os.path.splitext(file.filename)[0])
            filename = f"{base}_{uuid.uuid4().hex[:8]}{ext}"
            path = os.path.join(UPLOAD_DIR, filename)
            file.save(path)

            # generate tags
            raw_labels, _ = get_labels(path)
            cleaned_tags = clean_tags(raw_labels)

            # show tags page (no caption yet)
            return render_template(
                "result.html",
                image_file=filename,
                auto_tags=cleaned_tags,
                final_tags=cleaned_tags,  # prefill input
                caption=None
            )

        # ----------------------------
        # Stage 2: user edits tags -> generate caption
        # ----------------------------
        elif stage == "caption":
            filename = request.form.get("image_file")
            if not filename:
                return render_template("index.html", error="Missing image reference. Please upload again.")

            path = os.path.join(UPLOAD_DIR, filename)

            # take user tags (optional)
            user_tags = (request.form.get("custom_tags") or "").strip()
            if user_tags:
                final_tags = [t.strip() for t in user_tags.split(",") if t.strip()]
            else:
                # if user left empty, fall back to auto tags generated earlier (sent back in hidden field)
                auto_tags_str = request.form.get("auto_tags", "")
                final_tags = [t.strip() for t in auto_tags_str.split(",") if t.strip()]
                if not final_tags:
                    # last fallback
                    raw_labels, _ = get_labels(path)
                    final_tags = clean_tags(raw_labels)

            caption = generate_caption(final_tags)

            # re-display everything with caption
            auto_tags_display = request.form.get("auto_tags", "")
            auto_tags = [t.strip() for t in auto_tags_display.split(",") if t.strip()] or final_tags

            return render_template(
                "result.html",
                image_file=filename,
                auto_tags=auto_tags,
                final_tags=final_tags,
                caption=caption
            )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)

