#!/usr/bin/env python3
"""ptm-llama Gradio Space: instruction-tuned multi-PTM site prediction."""

import json
import re
from typing import List, Optional, Tuple

import gradio as gr
import requests
import spaces
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_REPO = "jbenbudd/ptm-llama"
MODEL_URL = "https://huggingface.co/jbenbudd/ptm-llama"

# Load inference-time configuration once at import (the same file produced by
# the evaluation notebook and pushed to the HF model repo). Everything below —
# window size, stride, prompt template, per-PTM instructions and calibrated
# consensus thresholds — is read from this file so app behaviour stays in sync
# with the model card without code changes.
_CFG_PATH = hf_hub_download(MODEL_REPO, "inference_config.json")
with open(_CFG_PATH) as _f:
    INFERENCE_CFG = json.load(_f)

WINDOW_SIZE: int = INFERENCE_CFG["window_size"]
STRIDE: int = INFERENCE_CFG["stride"]
MAX_NEW_TOKENS: int = INFERENCE_CFG["max_new_tokens"]
PROMPT_TEMPLATE: str = INFERENCE_CFG["prompt_template"]
PTM_INSTRUCTIONS = INFERENCE_CFG["instructions"]
CONSENSUS_THRESHOLDS = INFERENCE_CFG["consensus_thresholds"]
PTM_TYPES: List[str] = list(PTM_INSTRUCTIONS.keys())
DEFAULT_PTM = "Phosphorylation" if "Phosphorylation" in PTM_TYPES else PTM_TYPES[0]

BATCH_SIZE = 32

ESMFOLD_API_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"
ESMFOLD_MAX_LENGTH = 400

_SITE_RE = re.compile(r"^([A-Z])(\d+)$")
_SITES_RE = re.compile(r"Sites=<([^>]*)>")

# Cached on first GPU call; kept for the lifetime of the process.
_model = None
_tokenizer = None


def clean_sequence(sequence: str) -> str:
    """Uppercase, then strip anything that isn't one of the 20 canonical amino acids."""
    return re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", sequence.upper())


def make_windows(seq: str, w: int = WINDOW_SIZE, s: int = STRIDE) -> List[Tuple[int, str]]:
    """Sliding windows with a tail window to guarantee full coverage of the sequence."""
    L = len(seq)
    if L <= w:
        return [(0, seq)]
    starts = list(range(0, L - w + 1, s))
    if starts[-1] + w < L:
        starts.append(L - w)
    return [(i, seq[i:i + w]) for i in starts]


def build_prompt(window_seq: str, ptm_type: str) -> str:
    return PROMPT_TEMPLATE.format(
        instruction=PTM_INSTRUCTIONS[ptm_type],
        input=f"Seq=<{window_seq}>",
    )


def parse_window_sites(text: str) -> List[Tuple[str, int]]:
    """Parse `Sites=<K3,S12,...>` into [(letter, 1-indexed window position), ...]."""
    m = _SITES_RE.search(text)
    if not m:
        return []
    body = m.group(1).strip()
    if not body:
        return []
    out: List[Tuple[str, int]] = []
    for part in body.split(","):
        mm = _SITE_RE.match(part.strip())
        if mm:
            out.append((mm.group(1), int(mm.group(2))))
    return out


def _load_model():
    global _model, _tokenizer
    if _model is None:
        print(f"Loading tokenizer and model from {MODEL_REPO}...")
        tok = AutoTokenizer.from_pretrained(MODEL_REPO)
        if tok.pad_token is None:
            tok.pad_token = tok.unk_token
        tok.padding_side = "left"
        mdl = AutoModelForCausalLM.from_pretrained(
            MODEL_REPO,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).eval()
        mdl.config.use_cache = True
        _model, _tokenizer = mdl, tok
        print("Model loaded.")
    return _model, _tokenizer


@torch.inference_mode()
def _generate_batch(prompts: List[str]) -> List[str]:
    mdl, tok = _model, _tokenizer
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(mdl.device)
    out = mdl.generate(
        **enc,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        num_beams=1,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        use_cache=True,
    )
    new_tokens = out[:, enc.input_ids.shape[1]:]
    return tok.batch_decode(new_tokens, skip_special_tokens=True)


@spaces.GPU(duration=120)
def run_inference(sequence: str, ptm_type: str) -> List[str]:
    """Sliding-window inference + per-residue consensus voting for one full protein.

    Mirrors the reference `predict_sites` function on the model card and the eval
    notebook exactly, so results in this Space are consistent with the published
    per-PTM-type metrics.
    """
    mdl, tok = _load_model()
    L = len(sequence)
    threshold = CONSENSUS_THRESHOLDS[ptm_type]
    windows = make_windows(sequence)

    covered = [0] * L
    predicted = [0] * L

    for i in range(0, len(windows), BATCH_SIZE):
        batch = windows[i:i + BATCH_SIZE]
        prompts = [build_prompt(w_seq, ptm_type) for _, w_seq in batch]
        outs = _generate_batch(prompts)
        for (start, w_seq), comp in zip(batch, outs):
            end = min(start + len(w_seq), L)
            for k in range(start, end):
                covered[k] += 1
            for letter, pos_local in parse_window_sites(comp):
                if not (1 <= pos_local <= len(w_seq)):
                    continue
                pos_full_0 = start + pos_local - 1
                if not (0 <= pos_full_0 < L):
                    continue
                if sequence[pos_full_0] != letter:
                    continue
                predicted[pos_full_0] += 1

    sites: List[str] = []
    for i in range(L):
        if covered[i] > 0 and (predicted[i] / covered[i]) >= threshold:
            sites.append(f"{sequence[i]}{i + 1}")
    return sites


def get_pdb_from_esmfold(sequence: str) -> Optional[str]:
    """Predict 3D structure via the ESMFold public API and return the PDB string."""
    try:
        response = requests.post(
            ESMFOLD_API_URL,
            data=sequence,
            headers={"Content-Type": "text/plain"},
            timeout=120,
        )
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"ESMFold API error: {e}")
        return None


# Stock Gradio Default with no custom palette. Outfit is a light geometric sans;
# sequences stay monospace.
#
# Gradio does NOT read the Hugging Face website light/dark toggle — without an
# explicit __theme it follows the OS/browser prefers-color-scheme, which is why
# a Space can look pitch-black while HF itself is in light mode. Default to
# light when no theme is specified; ?__theme=dark still works.
THEME = gr.themes.Default(
    font=[gr.themes.GoogleFont("Outfit"), "ui-sans-serif", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
)

_THEME_JS = """
() => {
  const url = new URL(window.location);
  if (!url.searchParams.get('__theme')) {
    url.searchParams.set('__theme', 'light');
    window.location.replace(url.href);
  }
}
"""

CUSTOM_CSS = """
/* Sequence / site data: monospace for readability. */
.sequence-input textarea,
.ptm-panel-body,
.ptm-panel-footer code,
.site-chip {
    font-family: var(--font-mono) !important;
    font-variant-ligatures: none;
}

/* Result / status panels — theme tokens only so light & dark both work. */
.ptm-panel,
.ptm-status,
.ptm-unavailable,
.ptm-error {
    padding: 16px;
    border-radius: 8px;
    border: 1px solid var(--border-color-primary);
    background: var(--block-background-fill);
    color: var(--body-text-color);
}
.ptm-panel-header {
    font-weight: 600;
    margin-bottom: 8px;
}
.ptm-panel-body {
    line-height: 1.8;
    margin-bottom: 10px;
    word-break: break-word;
}
.ptm-panel-footer {
    font-size: 12px;
    color: var(--body-text-color-subdued);
}
.ptm-panel-footer code {
    color: var(--body-text-color);
}
.site-chip {
    display: inline-block;
    background: color-mix(in srgb, var(--body-text-color) 10%, transparent);
    color: var(--body-text-color);
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: 500;
}
.ptm-status {
    display: flex;
    align-items: center;
    gap: 14px;
    min-height: 60px;
}
.ptm-status-text { display: flex; flex-direction: column; gap: 4px; }
.ptm-status-text .primary { font-weight: 600; }
.ptm-status-text .secondary {
    color: var(--body-text-color-subdued);
    font-size: 13px;
}
.ptm-spinner {
    width: 22px;
    height: 22px;
    border: 3px solid var(--border-color-primary);
    border-top-color: var(--body-text-color);
    border-radius: 50%;
    animation: ptm-spin 0.9s linear infinite;
    flex-shrink: 0;
}
@keyframes ptm-spin { to { transform: rotate(360deg); } }
.ptm-error {
    background: color-mix(in srgb, #ef4444 10%, var(--block-background-fill));
    border-color: color-mix(in srgb, #ef4444 35%, var(--border-color-primary));
}
.ptm-unavailable {
    text-align: center;
    min-height: 300px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}

/* Examples table contrast + mono sequences. */
.gradio-container [data-testid="dataset"],
.gradio-container [data-testid="dataset"] *,
.gradio-container .samples-table,
.gradio-container .samples-table * {
    color: var(--body-text-color) !important;
}
.gradio-container [data-testid="dataset"] td:first-child,
.gradio-container .samples-table td:first-child {
    font-family: var(--font-mono) !important;
    font-size: 12px !important;
}
"""


def _structure_unavailable_html(sequence: str, predicted_sites: List[str], reason: str) -> str:
    sites_str = ", ".join(predicted_sites) if predicted_sites else "None"
    return f"""
    <div class="ptm-unavailable">
        <h3 style="margin:0 0 12px 0;">3D Structure Unavailable</h3>
        <p style="margin:0 0 8px 0;">{reason}</p>
        <p style="margin:0;"><strong>Predicted sites:</strong> {sites_str}</p>
    </div>
    """


def _status_html(primary: str, secondary: Optional[str] = None) -> str:
    """Loading indicator with a spinner, used while an async phase is running."""
    sec = f'<span class="secondary">{secondary}</span>' if secondary else ""
    return f"""
    <div class="ptm-status" role="status" aria-live="polite">
      <div class="ptm-spinner" aria-hidden="true"></div>
      <div class="ptm-status-text">
        <span class="primary">{primary}</span>
        {sec}
      </div>
    </div>
    """


def _error_html(message: str) -> str:
    return f'<div class="ptm-error">{message}</div>'


def create_3dmol_html(pdb_string: str, sequence: str, predicted_sites: List[str], ptm_type: str) -> str:
    """Build an HTML document embedding a 3Dmol.js viewer with highlighted PTM sites."""

    site_positions: List[int] = []
    site_labels: List[str] = []
    for site in predicted_sites:
        m = re.match(r"([A-Z])(\d+)", site)
        if m:
            site_positions.append(int(m.group(2)))
            site_labels.append(site)

    resi_js_array = "[" + ",".join(str(p) for p in site_positions) + "]"
    labels_js_array = "[" + ",".join(f'"{lbl}"' for lbl in site_labels) + "]"

    pdb_escaped = pdb_string.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #1a1a2e; font-family: Arial, sans-serif; }}
  #container {{ width: 100%; height: 600px; position: relative; }}
  #legend {{
    position: absolute; bottom: 12px; left: 12px; z-index: 10;
    background: rgba(26,26,46,0.85); color: #ccc; padding: 10px 14px;
    border-radius: 8px; font-size: 12px; line-height: 1.6;
    border: 1px solid rgba(255,255,255,0.1);
  }}
  #legend b {{ color: #fff; }}
  .dot {{
    display: inline-block; width: 10px; height: 10px;
    border-radius: 50%; margin-right: 5px; vertical-align: middle;
  }}
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.3/jquery.min.js"></script>
<script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
</head>
<body>
<div id="container"></div>
<div id="legend">
  <b>Legend</b><br>
  <span class="dot" style="background:#ef4444;"></span> {ptm_type} site<br>
  <span class="dot" style="background:linear-gradient(90deg,#3b82f6,#22c55e,#eab308,#ef4444);width:40px;border-radius:3px;height:8px;"></span> Backbone (N&rarr;C spectrum)
</div>
<script>
$(function() {{
  var pdb = `{pdb_escaped}`;
  var ptmResidues = {resi_js_array};
  var ptmLabels  = {labels_js_array};

  var viewer = $3Dmol.createViewer($("#container"), {{
    backgroundColor: "0x1a1a2e"
  }});
  viewer.addModel(pdb, "pdb");

  // Cartoon backbone coloured by residue index (spectrum)
  viewer.setStyle({{}}, {{
    cartoon: {{ color: "spectrum", opacity: 0.9 }}
  }});

  // PTM sites: add stick + sphere highlight
  if (ptmResidues.length > 0) {{
    viewer.setStyle({{ resi: ptmResidues }}, {{
      cartoon: {{ color: "spectrum", opacity: 0.9 }},
      stick:   {{ radius: 0.25, color: "#ef4444" }},
      sphere:  {{ radius: 0.7,  color: "#ef4444", opacity: 0.85 }}
    }});

    // Labels for each PTM site
    for (var i = 0; i < ptmResidues.length; i++) {{
      var atoms = viewer.getModel(0).selectedAtoms({{ resi: ptmResidues[i], atom: "CA" }});
      if (atoms.length > 0) {{
        viewer.addLabel(ptmLabels[i], {{
          position: atoms[0],
          backgroundColor: "rgba(239,68,68,0.8)",
          fontColor: "white",
          fontSize: 13,
          borderRadius: 4,
          padding: 3,
          showBackground: true
        }});
      }}
    }}
  }}

  viewer.zoomTo();
  viewer.render();
  viewer.zoom(0.85, 800);
}});
</script>
</body>
</html>"""

    return f'<iframe style="width:100%;height:620px;border:none;border-radius:8px;" srcdoc=\'{html}\'></iframe>'


def _sites_summary_html(sequence: str, ptm_type: str, sites: List[str]) -> str:
    threshold = CONSENSUS_THRESHOLDS[ptm_type]
    footer = (
        f"Consensus threshold for {ptm_type}: "
        f"<code>{threshold:.3f}</code> &middot; sequence length: <code>{len(sequence)}</code>"
    )
    if sites:
        header = f"{len(sites)} predicted {ptm_type.lower()} site(s)"
        body = ", ".join(f'<span class="site-chip">{s}</span>' for s in sites)
    else:
        header = f"No {ptm_type.lower()} sites predicted"
        body = (
            "<em>The model did not call any residues in this protein above the "
            "calibrated threshold.</em>"
        )
    return f"""
    <div class="ptm-panel">
      <div class="ptm-panel-header">{header}</div>
      <div class="ptm-panel-body">{body}</div>
      <div class="ptm-panel-footer">{footer}</div>
    </div>
    """


def predict_ptm_sites(sequence: str, ptm_type: str):
    """Orchestrator: sanitize sequence, run PTM prediction, render sites + 3D viewer.

    Implemented as a generator so the UI can stream visible loading states
    between phases (site prediction on GPU, then ESMFold structure).
    """
    if not sequence or not sequence.strip():
        yield (
            _error_html("Please enter an amino acid sequence."),
            _structure_unavailable_html("", [], "No sequence provided."),
        )
        return
    if ptm_type not in PTM_INSTRUCTIONS:
        yield (
            _error_html(f"Unknown PTM type: {ptm_type}."),
            _structure_unavailable_html("", [], "Unknown PTM type."),
        )
        return
    clean_seq = clean_sequence(sequence)
    if not clean_seq:
        yield (
            _error_html("Invalid sequence: no valid amino-acid letters found."),
            _structure_unavailable_html("", [], "Invalid sequence."),
        )
        return

    yield (
        _status_html(
            f"Running {ptm_type.lower()} site prediction on GPU…",
            f"Sliding-window inference over {len(clean_seq)} residues. "
            "This can take up to a minute — please be patient.",
        ),
        _status_html(
            "Waiting for site prediction…",
            "3D structure will be predicted once site prediction completes.",
        ),
    )
    sites = run_inference(clean_seq, ptm_type)
    sites_html = _sites_summary_html(clean_seq, ptm_type, sites)

    if len(clean_seq) > ESMFOLD_MAX_LENGTH:
        structure_html = _structure_unavailable_html(
            clean_seq, sites,
            f"Sequence too long for ESMFold structure prediction "
            f"({len(clean_seq)} residues, max {ESMFOLD_MAX_LENGTH}).",
        )
        yield sites_html, structure_html
        return

    yield (
        sites_html,
        _status_html(
            "Predicting 3D structure with ESMFold…",
            f"{len(clean_seq)} residues via the public ESMFold API.",
        ),
    )
    pdb = get_pdb_from_esmfold(clean_seq)
    if pdb:
        structure_html = create_3dmol_html(pdb, clean_seq, sites, ptm_type)
    else:
        structure_html = _structure_unavailable_html(
            clean_seq, sites,
            "The ESMFold structure prediction service is currently unavailable. "
            "Please try again later.",
        )
    yield sites_html, structure_html


ABOUT_MD = f"""
**ptm-llama** is an instruction-tuned causal protein language model that predicts
post-translational modification (PTM) sites from raw amino-acid sequence.
A single LoRA adapter was fine-tuned on top of
[ProLLaMA](https://huggingface.co/GreatCaptainNemo/ProLLaMA_Stage_1) to jointly
handle **methylation, phosphorylation, and ubiquitination** — the PTM type is
selected purely through the natural-language instruction at inference time,
demonstrating that a single generative protein LLM can be steered across PTM
prediction tasks without task-specific classification heads.

At inference the app runs sliding-window generation (window {WINDOW_SIZE},
stride {STRIDE}), aggregates per-residue consensus scores, and applies a
per-PTM-type threshold calibrated on a held-out protein-level split.

Model weights, inference config, calibrated thresholds, and full evaluation
(per-PTM metrics, per-residue-type breakdown, cross-instruction ablation) are
on Hugging Face: [{MODEL_REPO}]({MODEL_URL}).
"""


with gr.Blocks(theme=THEME, title="ptm-llama", css=CUSTOM_CSS, js=_THEME_JS) as demo:
    gr.Markdown("# 🧬 ptm-llama — multi-PTM site predictor")
    gr.Markdown(
        "Predict post-translational modification sites in a protein sequence. "
        "Choose a PTM type and paste an amino-acid sequence; the model runs "
        "sliding-window inference, aggregates per-residue consensus votes, and "
        "applies the calibrated threshold for that PTM type."
    )

    with gr.Accordion("About the model", open=False):
        gr.Markdown(ABOUT_MD)

    with gr.Row():
        with gr.Column(scale=1):
            sequence_input = gr.Textbox(
                label="Amino acid sequence",
                placeholder=(
                    "Paste a single-letter amino-acid sequence "
                    "(e.g. MASDEGKLFVGGLSFDTNEQALEQVFSKYGQ...)"
                ),
                lines=6,
                elem_classes=["sequence-input"],
            )
            ptm_dropdown = gr.Dropdown(
                choices=PTM_TYPES,
                value=DEFAULT_PTM,
                label="PTM type",
                info="The instruction handed to the model at inference time.",
            )
            predict_btn = gr.Button("Predict sites", variant="primary")
            output_sites = gr.HTML(label="Predicted sites")

            gr.Examples(
                examples=[
                    ["XRPGPRGCSAPAARRPGPRRRRSSFPPLYSSGLVECEDQDPLNPDRSFDVESVKKEIQRGRKLKCKFCHKRGATVGCDLKNCNKNYHFFCAKKDDAVPQSDGVRGIYKLLCQQHAQFPIIAQSAKFSGVKRKRGRKKPLSGNHVQPPETMKCNTFIRQVKEEHGRHTDATVKVPFLKKCKEAGLLNYLLEEILDKVHSIPEKLMDETTSESDYEEIGSALFDCRLFEDTFVNFQAAIEKKIHASQQRWQQLKEEIELLQDLKQTLCSFQENRDLMSSSTSISSLSY", "Phosphorylation"],
                    ["MILILGGGFAGVSAYNQNKENSLVVDRKDYFLLTPWIIDFICGMKKLEDIIVKYKKVILGNVQKIDFKNKKVILDNSKELTYDKLIVSLGHHQNLPRLKGAKEYAHKIETLEDAIELKRRLNEVKDITIIGGGATGVELAGNIKGKKITLVQRRNRLLPTMSTASSKKAEDLLRELGVNLMLGVEAIEIKKDSVVTSYGEIKTELTIFAGGLKGPQIVGNLHANKNHRLLVDKNLKSIEYNDVYGAGDCVTFEDKEIPMTADIAVAAGRVVMKNILGNEIEFKPKRLATTIRIRNEFFGDFGENYVEGKFAKILKDISYLESLLLPRRLRE", "Methylation"],
                    ["MDDDIAALVVDNGSGMCKAGFAGDDAPRAVFPSIVGRPRHQGVMVGMGQKDSYVGDEAQSKRGILTLKYPIEHGIVTNWD", "Ubiquitination"],
                ],
                inputs=[sequence_input, ptm_dropdown],
                label="Examples",
            )

        with gr.Column(scale=2):
            output_structure = gr.HTML(label="3D structure (ESMFold)")
            gr.Markdown(
                f"Structure predicted on demand via the "
                f"[ESMFold API](https://esmatlas.com/about) for sequences "
                f"≤ {ESMFOLD_MAX_LENGTH} residues. Model weights: "
                f"[{MODEL_REPO}]({MODEL_URL})."
            )

    predict_btn.click(
        fn=predict_ptm_sites,
        inputs=[sequence_input, ptm_dropdown],
        outputs=[output_sites, output_structure],
    )


if __name__ == "__main__":
    print("Starting ptm-llama Gradio app...")
    demo.launch(share=False)
