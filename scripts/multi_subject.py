import torch
import torch.nn.functional as F
import gradio as gr
import modules.scripts as scripts
import modules.shared as shared
from modules.processing import StableDiffusionProcessingTxt2Img, process_images
from modules.script_callbacks import on_ui_tabs, on_cfg_denoiser, remove_current_script_callbacks
from modules.sd_samplers import samplers

TAG_DB = {
    "character_type": {
        "label": "Character Type",
        "icon": "👤",
        "tags": ["1girl", "1boy", "2girls", "2boys", "1girl 1boy", "multiple girls", "multiple boys", "group"],
    },
    "hair": {
        "label": "Hair",
        "icon": "💇",
        "tags": [
            "blonde hair", "blue hair", "red hair", "black hair", "white hair",
            "pink hair", "purple hair", "green hair", "silver hair", "aqua hair",
            "long hair", "short hair", "twintails", "ponytail", "braid",
            "bangs", "ahoge", "side ponytail", "hair over one eye",
        ],
    },
    "eyes": {
        "label": "Eyes",
        "icon": "👁️",
        "tags": ["blue eyes", "red eyes", "green eyes", "golden eyes", "purple eyes", "brown eyes", "heterochromia", "glowing eyes", "closed eyes"],
    },
    "body": {
        "label": "Body",
        "icon": "🧍",
        "tags": ["tall", "short", "petite", "muscular", "slim", "large breasts", "small breasts", "flat chest"],
    },
    "outfit": {
        "label": "Outfit",
        "icon": "👗",
        "tags": [
            "white dress", "black dress", "school uniform", "maid outfit",
            "armor", "casual clothes", "formal wear", "swimsuit", "kimono",
            "gothic lolita", "military uniform", "nurse outfit", "witch hat",
            "cape", "hoodie", "suit", "lab coat", "cheerleader",
        ],
    },
    "pose_action": {
        "label": "Pose / Action",
        "icon": "🎭",
        "tags": [
            "standing", "sitting", "lying", "walking", "running",
            "looking at viewer", "looking away", "from behind", "from side",
            "hand on hip", "arms crossed", "peace sign", "stretching",
            "fighting stance", "dancing", "reading", "sleeping",
        ],
    },
    "expression": {
        "label": "Expression",
        "icon": "😊",
        "tags": ["smile", "grin", "serious", "angry", "sad", "blush", "crying", "surprised", "shy", "tsundere", "closed mouth", "open mouth", "tongue out"],
    },
    "environment": {
        "label": "Environment",
        "icon": "🌿",
        "tags": [
            "forest", "beach", "city", "night sky", "sunset",
            "indoors", "outdoors", "classroom", "castle", "ruins",
            "snow", "rain", "cherry blossoms", "space", "underwater",
            "library", "garden", "rooftop", "street",
        ],
    },
    "style": {
        "label": "Style / Quality",
        "icon": "✨",
        "tags": [
            "masterpiece", "best quality", "cinematic lighting", "dramatic lighting",
            "soft lighting", "backlighting", "lens flare", "depth of field",
            "bokeh", "film grain", "anime style", "realistic", "watercolor",
            "oil painting", "concept art", "illustration",
        ],
    },
    "camera": {
        "label": "Camera / Framing",
        "icon": "📷",
        "tags": ["close-up", "upper body", "full body", "wide shot", "cowboy shot", "portrait", "dutch angle", "bird's eye view", "worm's eye view", "panoramic"],
    },
}

REGION_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
AD_MERGE_INVERT = ["None", "Merge", "Merge and Invert"]


class _AttrReplacer:
    def __init__(self, obj, attr, original):
        self.obj = obj
        self.attr = attr
        self.original = original

    def restore(self):
        try:
            setattr(self.obj, self.attr, self.original)
        except Exception:
            pass


class MultiSubjectEngine:
    _instance = None

    def __init__(self):
        self._regional_data = None
        self._replacers = []

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def cleanup(self):
        self._regional_data = None
        for replacer in self._replacers:
            replacer.restore()
        self._replacers = []
        try:
            remove_current_script_callbacks()
        except Exception:
            pass

    def apply_weight_to_prompt(self, prompt, weight):
        if weight == 1.0:
            return prompt
        return f"({prompt}:{weight:.2f})"

    @staticmethod
    def _extract_extra_networks(text):
        import re
        if not text:
            return [], text
        pattern = r'(<(?:lora|lyco|hypernet|embedding):[^>]+>)'
        tags = re.findall(pattern, text)
        cleaned = re.sub(pattern, '', text).strip()
        cleaned = re.sub(r',\s*,', ',', cleaned)
        cleaned = cleaned.strip().strip(',').strip()
        return tags, cleaned

    def build_final_prompt(self, active_prompts, env, blend_mode):
        extra_tags = []
        clean_env = env
        if env:
            extra_tags, clean_env = self._extract_extra_networks(env)
        combined = []
        for prompt in active_prompts:
            parts = [prompt]
            if clean_env:
                parts.append(clean_env)
            combined.append(" ".join(parts))
        if blend_mode == "Simple AND":
            body = " AND ".join(combined)
        elif blend_mode in ("BREAK (Attention)", "Regional Blend (Horizontal)"):
            body = " BREAK ".join(combined)
        else:
            body = " AND ".join(combined)
        if extra_tags:
            return " ".join(extra_tags) + " " + body
        return body

    def setup_regional(self, prompts, env, ratios, base_ratio, feather_width, calc_mode):
        self._regional_data = {
            "prompts": prompts,
            "env": env,
            "num_regions": len(prompts),
            "ratios": ratios,
            "base_ratio": base_ratio,
            "feather_width": feather_width,
            "calc_mode": calc_mode,
        }

    def has_regional(self):
        return self._regional_data is not None

    def apply_regional_to_processing(self, p):
        if not self._regional_data:
            return
        data = self._regional_data
        num_regions = data["num_regions"]
        ratios = data["ratios"]
        feather = data["feather_width"]
        calc_mode = data["calc_mode"]
        latent_width = p.width // 8
        latent_height = p.height // 8
        masks = self._create_horizontal_masks(latent_width, latent_height, num_regions, ratios, feather)
        if calc_mode == "Attention":
            self._apply_attention_regional(p, masks, num_regions, latent_width, latent_height, data)
        else:
            self._apply_latent_regional(p, masks, num_regions, latent_width, latent_height, data)
        p.extra_generation_params.update({
            "ms_blend_mode": "Regional Blend (Horizontal)",
            "ms_num_regions": num_regions,
            "ms_ratios": ",".join(str(r) for r in ratios),
            "ms_base_ratio": data["base_ratio"],
            "ms_feather": feather,
            "ms_calc_mode": calc_mode,
        })

    def parse_ratios(self, ratio_str, num_regions):
        try:
            ratios = [float(x.strip()) for x in ratio_str.split(",") if x.strip()]
        except (ValueError, AttributeError):
            ratios = []
        while len(ratios) < num_regions:
            ratios.append(1.0)
        return ratios[:num_regions]

    def _create_horizontal_masks(self, width, height, num_regions, ratios, feather_width=16):
        total = sum(ratios)
        if total <= 0:
            total = 1.0
        boundaries = []
        current_x = 0.0
        for ratio in ratios:
            region_width = width * ratio / total
            boundaries.append((current_x, current_x + region_width))
            current_x += region_width
        masks = []
        for i, (start, end) in enumerate(boundaries):
            mask = torch.zeros(1, 1, height, width)
            start_idx = max(0, int(round(start)))
            end_idx = min(width, int(round(end)))
            if i == len(boundaries) - 1:
                end_idx = width
            mask[:, :, :, start_idx:end_idx] = 1.0
            if feather_width > 0 and num_regions > 1:
                left_feather_start = max(0, start_idx - feather_width)
                right_feather_end = min(width, end_idx + feather_width)
                for x in range(left_feather_start, start_idx):
                    span = start_idx - left_feather_start
                    alpha = (x - left_feather_start + 1) / (span + 1) if span > 0 else 1.0
                    current = mask[0, 0, 0, x].item()
                    mask[:, :, :, x] = max(current, alpha)
                for x in range(end_idx, right_feather_end):
                    span = right_feather_end - end_idx
                    alpha = 1.0 - (x - end_idx + 1) / (span + 1) if span > 0 else 1.0
                    current = mask[0, 0, 0, x].item()
                    mask[:, :, :, x] = max(current, alpha)
            masks.append(mask)
        return masks

    def _count_tokens(self, clip, prompt):
        try:
            tokens = clip.tokenize(prompt)
            if isinstance(tokens, dict):
                for k, v in tokens.items():
                    if hasattr(v, 'shape'):
                        return v.shape[-1]
                    elif isinstance(v, (list, tuple)):
                        return len(v)
            elif hasattr(tokens, 'shape'):
                return tokens.shape[-1]
            elif isinstance(tokens, (list, tuple)):
                return len(tokens)
            return 77
        except Exception:
            return 77

    def _apply_attention_regional(self, p, masks, num_regions, width, height, data):
        try:
            unet = p.sd_model.forge_objects.unet
            clip = p.sd_model.forge_objects.clip
            device = next(unet.model.parameters()).device
            dtype = next(unet.model.parameters()).dtype
        except Exception:
            return
        masks_on_device = [m.to(device=device, dtype=dtype) for m in masks]
        prompts = data["prompts"]
        env = data["env"]
        base_ratio = data["base_ratio"]
        token_boundaries = []
        current_pos = 0
        for prompt in prompts:
            full_prompt = f"{prompt} {env}" if env else prompt
            token_count = self._count_tokens(clip, full_prompt)
            token_boundaries.append((current_pos, current_pos + token_count))
            current_pos += token_count
        attn2_modules = []
        try:
            for name, module in unet.model.named_modules():
                if "attn2" in name and hasattr(module, "to_q"):
                    attn2_modules.append(module)
        except Exception:
            pass
        if attn2_modules:
            self._apply_cross_attn_masking(attn2_modules, masks_on_device, token_boundaries, num_regions, base_ratio, dtype)
        else:
            self._apply_output_block_masking(p, unet, masks_on_device, base_ratio)

    def _apply_cross_attn_masking(self, attn_modules, masks, token_boundaries, num_regions, base_ratio, dtype):
        for attn_module in attn_modules:
            original_forward = attn_module.forward

            def make_masked_forward(mod, orig_fwd, masks_ref, boundaries_ref, n_regions, b_ratio, dt):
                def masked_forward(x, context=None, value=None, mask=None):
                    B, N, C = x.shape
                    spatial_size = int(N ** 0.5)
                    if spatial_size * spatial_size != N or spatial_size <= 1:
                        return orig_fwd(x, context=context, value=value, mask=mask)
                    try:
                        q = mod.to_q(x)
                        ctx = context if context is not None else x
                        k = mod.to_k(ctx)
                        v = mod.to_v(ctx) if value is None else mod.to_v(value)
                        heads = mod.heads
                        head_dim = C // heads
                        q = q.view(B, N, heads, head_dim).transpose(1, 2)
                        T = k.shape[1]
                        k = k.view(B, T, heads, head_dim).transpose(1, 2)
                        v = v.view(B, T, heads, head_dim).transpose(1, 2)
                        scale = head_dim ** -0.5
                        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
                        attn_mask = torch.zeros(B, heads, N, T, device=x.device, dtype=dt)
                        for ri, (t_start, t_end) in enumerate(boundaries_ref):
                            t_start = min(t_start, T)
                            t_end = min(t_end, T)
                            region_mask = masks_ref[ri]
                            resized = F.interpolate(region_mask, size=(spatial_size, spatial_size), mode="bilinear", align_corners=False)
                            spatial_m = resized.view(1, 1, N, 1)
                            token_m = torch.zeros(1, 1, 1, T, device=x.device, dtype=dt)
                            token_m[:, :, :, t_start:t_end] = 1.0
                            attn_mask = attn_mask + spatial_m * token_m
                        attn_mask = attn_mask + b_ratio
                        neg_inf = torch.tensor(-1e9, device=x.device, dtype=dt)
                        hard_mask = torch.where(attn_mask > 1e-6, torch.zeros_like(attn_mask), neg_inf)
                        attn = attn + hard_mask
                        attn = attn.softmax(dim=-1)
                        out = torch.matmul(attn, v)
                        out = out.transpose(1, 2).contiguous().view(B, N, C)
                        return mod.to_out(out)
                    except Exception:
                        return orig_fwd(x, context=context, value=value, mask=mask)
                return masked_forward

            attn_module.forward = make_masked_forward(attn_module, original_forward, masks, token_boundaries, num_regions, base_ratio, dtype)
            self._replacers.append(_AttrReplacer(attn_module, "forward", original_forward))

    def _apply_output_block_masking(self, p, unet, masks, base_ratio):
        def output_block_patch(h, hsp, transformer_options):
            B, C, H, W = h.shape
            if H <= 1 or W <= 1:
                return h, hsp
            base_weight = 1.0 - base_ratio
            region_weight = base_ratio
            blended = torch.zeros_like(h)
            total_weight = torch.zeros(B, C, H, W, device=h.device, dtype=h.dtype)
            for mask in masks:
                resized = F.interpolate(mask, size=(H, W), mode="bilinear", align_corners=False)
                resized = resized.expand(B, C, H, W)
                weight = resized * region_weight
                blended = blended + h * weight
                total_weight = total_weight + weight
            base_mask = torch.ones(B, C, H, W, device=h.device, dtype=h.dtype) * base_weight
            blended = blended + h * base_mask
            total_weight = total_weight + base_mask
            total_weight = total_weight.clamp(min=1e-6)
            h = blended / total_weight
            return h, hsp
        m = unet.clone()
        m.set_model_output_block_patch(output_block_patch)
        p.sd_model.forge_objects.unet = m

    def _apply_latent_regional(self, p, masks, num_regions, width, height, data):
        try:
            unet = p.sd_model.forge_objects.unet
            device = next(unet.model.parameters()).device
            dtype = next(unet.model.parameters()).dtype
        except Exception:
            return
        masks_on_device = [m.to(device=device, dtype=dtype) for m in masks]
        base_ratio = data["base_ratio"]
        stored_masks = masks_on_device

        def latent_blend_callback(params):
            try:
                x = params.x
                if x is None:
                    return
                B, C, H, W = x.shape
                if H <= 1 or W <= 1:
                    return
                base_weight = 1.0 - base_ratio
                region_weight = base_ratio
                blended = torch.zeros_like(x)
                total_weight = torch.zeros(B, C, H, W, device=x.device, dtype=x.dtype)
                for mask in stored_masks:
                    resized = F.interpolate(mask, size=(H, W), mode="bilinear", align_corners=False)
                    resized = resized.expand(B, C, H, W)
                    weight = resized * region_weight
                    blended = blended + x * weight
                    total_weight = total_weight + weight
                base_mask = torch.ones(B, C, H, W, device=x.device, dtype=x.dtype) * base_weight
                blended = blended + x * base_mask
                total_weight = total_weight + base_mask
                total_weight = total_weight.clamp(min=1e-6)
                params.x = blended / total_weight
            except Exception:
                pass

        on_cfg_denoiser(latent_blend_callback)
        m = unet.clone()
        p.sd_model.forge_objects.unet = m


class _MultiSubjectScript(scripts.Script):
    sorting_priority = 15

    def title(self):
        return "Multi-Subject Regional (Backend Hook)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return []

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        engine = MultiSubjectEngine.get()
        if engine.has_regional():
            engine.apply_regional_to_processing(p)

    def postprocess(self, p, processed, *args):
        engine = MultiSubjectEngine.get()
        engine.cleanup()


def _get_sampler_names():
    try:
        return [s.name for s in samplers]
    except Exception:
        return ["Euler", "Euler a", "DPM++ 2M Karras", "DPM++ SDE Karras"]


def _render_region_preview(ratios, active_count=None):
    if not ratios:
        ratios = [1.0, 1.0]
    if active_count is not None:
        ratios = ratios[:active_count]
    total = sum(ratios)
    if total <= 0:
        total = 1.0
    svg_width = 400
    svg_height = 60
    x = 0
    rects = []
    labels = []
    for i, ratio in enumerate(ratios):
        w = (ratio / total) * svg_width
        color = REGION_COLORS[i] if i < len(REGION_COLORS) else "#888"
        rects.append(f'<rect x="{x}" y="0" width="{w}" height="{svg_height}" fill="{color}" opacity="0.6" rx="4"/>')
        label_x = x + w / 2
        pct = ratio / total * 100
        labels.append(f'<text x="{label_x}" y="{svg_height/2+5}" text-anchor="middle" fill="white" font-size="12" font-weight="bold">R{i+1}: {pct:.0f}%</text>')
        x += w
    return f'<svg width="{svg_width}" height="{svg_height}" xmlns="http://www.w3.org/2000/svg" style="border-radius:8px;display:block;margin:8px auto;">{"".join(rects)}{"".join(labels)}</svg>'


def _get_ad_model_list():
    import os
    from pathlib import Path
    ad_dirs = []
    try:
        from modules.paths import models_path
        ad_dirs.append(Path(models_path) / 'adetailer')
    except Exception:
        pass
    data_dir = getattr(shared.cmd_opts, 'datadir', None)
    if data_dir:
        ad_dirs.append(Path(data_dir) / 'models' / 'adetailer')
    ad_dirs.append(Path(shared.cmd_opts.datadir if hasattr(shared.cmd_opts, 'datadir') else '.') / 'models' / 'adetailer')
    script_dir = Path(__file__).resolve().parent.parent.parent
    ad_dirs.append(script_dir / 'models' / 'adetailer')
    for d in list(ad_dirs):
        ad_dirs.append(d.resolve())
    seen = set()
    unique_dirs = []
    for d in ad_dirs:
        r = str(d)
        if r not in seen:
            seen.add(r)
            unique_dirs.append(d)
    try:
        from adetailer import get_models
        for ad_dir in unique_dirs:
            if ad_dir.exists():
                model_mapping = get_models(ad_dir, huggingface=False)
                if model_mapping:
                    return list(model_mapping.keys())
    except Exception:
        pass
    found = []
    for ad_dir in unique_dirs:
        if not ad_dir.exists():
            continue
        for f in sorted(os.listdir(ad_dir)):
            if f.endswith(('.pt', '.pth', '.safetensors', '.onnx')):
                name = os.path.splitext(f)[0]
                if name not in found:
                    found.append(name)
    return found


def _get_cn_models():
    import os
    from pathlib import Path
    try:
        from controlnet_ext import get_cn_models
        models = get_cn_models()
        if models:
            return ["None"] + models
    except Exception:
        pass
    cn_dirs = []
    try:
        from modules.paths import models_path
        cn_dirs.append(Path(models_path) / 'ControlNet')
        cn_dirs.append(Path(models_path) / 'controlnet')
    except Exception:
        pass
    data_dir = getattr(shared.cmd_opts, 'datadir', None)
    if data_dir:
        cn_dirs.append(Path(data_dir) / 'models' / 'ControlNet')
    script_dir = Path(__file__).resolve().parent.parent.parent
    cn_dirs.append(script_dir / 'models' / 'ControlNet')
    found = []
    for cn_dir in cn_dirs:
        if not cn_dir.exists():
            continue
        for f in sorted(os.listdir(cn_dir)):
            if f.endswith(('.safetensors', '.pt', '.pth', '.ckpt', '.bin')):
                name = os.path.splitext(f)[0]
                if name not in found:
                    found.append(name)
    return ["None"] + found if found else ["None"]


def _get_checkpoint_list():
    try:
        import modules.sd_models
        checkpoints = modules.sd_models.checkpoint_tiles(use_short=True)
        return ["Use same checkpoint"] + checkpoints
    except Exception:
        return ["Use same checkpoint"]


def _interrupt():
    shared.state.interrupt()


def _skip():
    shared.state.skip()


def _on_generate_start():
    return gr.Button(visible=False), gr.Button(visible=True), gr.Button(visible=True)


def _on_generate_end():
    return gr.Button(visible=True), gr.Button(visible=False), gr.Button(visible=False)


def _build_ad_unit(d):
    unit_dict = {
        "ad_model": d.get("model", "face_yolov8n.pt"),
        "ad_prompt": d.get("prompt", ""),
        "ad_negative_prompt": d.get("negative_prompt", ""),
        "ad_confidence": d.get("confidence", 0.3),
        "ad_mask_k_largest": d.get("mask_k_largest", 0),
        "ad_mask_min_ratio": d.get("mask_min_ratio", 0.0),
        "ad_mask_max_ratio": d.get("mask_max_ratio", 1.0),
        "ad_x_offset": d.get("x_offset", 0),
        "ad_y_offset": d.get("y_offset", 0),
        "ad_dilate_erode": d.get("dilate_erode", 4),
        "ad_mask_merge_invert": d.get("mask_merge_invert", "None"),
        "ad_mask_blur": d.get("mask_blur", 4),
        "ad_denoising_strength": d.get("denoising_strength", 0.4),
        "ad_inpaint_only_masked": d.get("inpaint_only_masked", True),
        "ad_inpaint_only_masked_padding": d.get("inpaint_padding", 32),
        "ad_use_inpaint_width_height": d.get("use_inpaint_wh", False),
        "ad_inpaint_width": d.get("inpaint_width", 512),
        "ad_inpaint_height": d.get("inpaint_height", 512),
        "ad_use_steps": d.get("use_steps", False),
        "ad_steps": d.get("steps", 28),
        "ad_use_cfg_scale": d.get("use_cfg", False),
        "ad_cfg_scale": d.get("cfg_scale", 7.0),
        "ad_use_sampler": d.get("use_sampler", False),
        "ad_sampler": d.get("sampler", "DPM++ 2M Karras"),
        "ad_use_noise_multiplier": d.get("use_noise_mult", False),
        "ad_noise_multiplier": d.get("noise_multiplier", 1.0),
        "ad_use_clip_skip": d.get("use_clip_skip", False),
        "ad_clip_skip": d.get("clip_skip", 1),
        "ad_restore_face": d.get("restore_face", False),
        "ad_controlnet_model": d.get("cn_model", "None"),
        "ad_controlnet_module": d.get("cn_module", "None"),
        "ad_controlnet_weight": d.get("cn_weight", 1.0),
        "ad_controlnet_guidance_start": d.get("cn_guidance_start", 0.0),
        "ad_controlnet_guidance_end": d.get("cn_guidance_end", 1.0),
        "is_api": True,
    }
    try:
        from lib_adetailer.process import ADetailerUnit
        return ADetailerUnit(**unit_dict)
    except (ImportError, TypeError):
        pass
    try:
        from adetailer.args import ADetailerArgs
        return ADetailerArgs(**unit_dict)
    except (ImportError, TypeError):
        pass
    return unit_dict


class _ScriptProxy:
    def __init__(self, script, new_from, new_to):
        object.__setattr__(self, '_wrapped', script)
        object.__setattr__(self, 'args_from', new_from)
        object.__setattr__(self, 'args_to', new_to)

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def __setattr__(self, name, value):
        if name in ('_wrapped', 'args_from', 'args_to'):
            object.__setattr__(self, name, value)
        else:
            setattr(self._wrapped, name, value)


def _setup_scripts(p, ad1=None, ad2=None):
    try:
        enabled = (ad1 and ad1.get("enable", False)) or (ad2 and ad2.get("enable", False))
        if not enabled:
            return
        from copy import copy as shallow_copy
        original = scripts.scripts_txt2img
        ad_script = None
        for s in original.alwayson_scripts:
            if "detailer" in s.title().lower():
                ad_script = s
                break
        if not ad_script:
            return
        ad_num_args = ad_script.args_to - ad_script.args_from
        if ad_num_args <= 0:
            return
        proxy = _ScriptProxy(ad_script, 0, ad_num_args)
        runner = shallow_copy(original)
        runner.alwayson_scripts = [proxy]
        runner.selectable_scripts = []
        p.scripts = runner
        p.is_api = True
        args = [None] * ad_num_args
        if ad1 and ad1.get("enable", False):
            args[0] = True
            args[1] = _build_ad_unit(ad1)
            num_models = getattr(shared.opts, 'ad_max_models', 2)
            for i in range(1, num_models):
                idx = 1 + i
                if idx < ad_num_args:
                    if i == 1 and ad2 and ad2.get("enable", False):
                        args[idx] = _build_ad_unit(ad2)
                    else:
                        args[idx] = _build_ad_unit({"model": "None"})
        else:
            args[0] = False
            if ad_num_args > 1:
                args[1] = _build_ad_unit({"model": "None"})
        p.script_args = args
    except Exception:
        pass


def _generate(
    cp0, cp1, cp2, cp3,
    cw0, cw1, cw2, cw3,
    ce0, ce1, ce2, ce3,
    cg0, cg1, cg2, cg3,
    main_env, negative_prompt, blend_mode,
    region_ratios, base_ratio, feather_width, calc_mode,
    width, height, steps, cfg_scale, sampler_name, seed, batch_size,
    checkpoint_name,
    ad1_enable, ad1_model, ad1_prompt, ad1_neg, ad1_conf, ad1_k, ad1_min_r, ad1_max_r,
    ad1_x_off, ad1_y_off, ad1_dilate, ad1_merge, ad1_mask_blur, ad1_ds,
    ad1_iom, ad1_iom_pad, ad1_use_wh, ad1_iw, ad1_ih,
    ad1_use_steps, ad1_steps, ad1_use_cfg, ad1_cfg, ad1_use_sampler, ad1_sampler,
    ad1_use_noise, ad1_noise, ad1_use_clip, ad1_clip, ad1_restore,
    ad1_cn_model, ad1_cn_weight, ad1_cn_start, ad1_cn_end,
    ad2_enable, ad2_model, ad2_prompt, ad2_neg, ad2_conf,
    ad2_mask_blur, ad2_ds, ad2_iom, ad2_iom_pad, ad2_dilate,
):
    char_prompts = [cp0, cp1, cp2, cp3]
    char_weights = [cw0, cw1, cw2, cw3]
    char_enableds = [ce0, ce1, ce2, ce3]
    char_genders = [cg0, cg1, cg2, cg3]

    engine = MultiSubjectEngine.get()
    engine.cleanup()

    if checkpoint_name and checkpoint_name != "Use same checkpoint":
        try:
            import modules.sd_models
            shared.opts.sd_model_checkpoint = checkpoint_name
            modules.sd_models.reload_model_weights()
        except Exception:
            pass

    active_prompts = []
    girls_count = 0
    boys_count = 0
    for i in range(4):
        prompt = char_prompts[i] if i < len(char_prompts) else ""
        weight = char_weights[i] if i < len(char_weights) else 1.0
        enabled = char_enableds[i] if i < len(char_enableds) else False
        gender = char_genders[i] if i < len(char_genders) else "none"
        if enabled and prompt and prompt.strip():
            active_prompts.append(engine.apply_weight_to_prompt(prompt.strip(), weight))
            if gender == "girl":
                girls_count += 1
            elif gender == "boy":
                boys_count += 1

    if not active_prompts:
        return [], "No active characters. Please enable at least one character with a prompt."

    count_tags = []
    if girls_count == 1:
        count_tags.append("1girl")
    elif girls_count > 1:
        count_tags.append(f"{girls_count}girls")
    if boys_count == 1:
        count_tags.append("1boy")
    elif boys_count > 1:
        count_tags.append(f"{boys_count}boys")
    count_str = ", ".join(count_tags)

    env = main_env.strip() if main_env else ""

    final_prompt = engine.build_final_prompt(active_prompts, env, blend_mode)

    prompt_parts = []
    if count_str:
        prompt_parts.append(count_str)
    prompt_parts.append(final_prompt)
    final_prompt = ", ".join(prompt_parts)

    neg = negative_prompt.strip() if negative_prompt else ""

    if blend_mode == "Regional Blend (Horizontal)":
        ratios = engine.parse_ratios(region_ratios, len(active_prompts))
        engine.setup_regional(active_prompts, env, ratios, base_ratio, feather_width, calc_mode)

    outdir_samples = getattr(shared.opts, "outdir_samples", None) or getattr(shared.opts, "outdir_txt2img_samples", None) or getattr(shared.cmd_opts, "outdir_txt2img_samples", None) or getattr(shared.cmd_opts, "outdir_txt2img", None) or "outputs/txt2img-images"
    outdir_grids = getattr(shared.opts, "outdir_grids", None) or getattr(shared.opts, "outdir_txt2img_grids", None) or getattr(shared.cmd_opts, "outdir_txt2img_grids", None) or getattr(shared.cmd_opts, "outdir_grids", None) or "outputs/txt2img-grids"
    import os
    os.makedirs(outdir_samples, exist_ok=True)
    os.makedirs(outdir_grids, exist_ok=True)
    p = StableDiffusionProcessingTxt2Img(
        outpath_samples=outdir_samples,
        outpath_grids=outdir_grids,
        sd_model=shared.sd_model,
        prompt=final_prompt,
        negative_prompt=neg,
        steps=int(steps) if steps else 28,
        sampler_name=sampler_name or "Euler a",
        cfg_scale=float(cfg_scale) if cfg_scale else 7.0,
        width=int(width) if width else 1024,
        height=int(height) if height else 1024,
        seed=int(seed) if seed is not None and str(seed).strip() not in ("", "None") and int(seed) >= 0 else -1,
        batch_size=int(batch_size) if batch_size else 1,
        do_not_save_grid=True,
        n_iter=1,
    )

    ad1_params = {
        "enable": ad1_enable, "model": ad1_model, "prompt": ad1_prompt or "",
        "negative_prompt": ad1_neg or "", "confidence": ad1_conf,
        "mask_k_largest": ad1_k, "mask_min_ratio": ad1_min_r, "mask_max_ratio": ad1_max_r,
        "x_offset": ad1_x_off, "y_offset": ad1_y_off, "dilate_erode": ad1_dilate,
        "mask_merge_invert": ad1_merge, "mask_blur": ad1_mask_blur,
        "denoising_strength": ad1_ds, "inpaint_only_masked": ad1_iom,
        "inpaint_padding": ad1_iom_pad, "use_inpaint_wh": ad1_use_wh,
        "inpaint_width": ad1_iw, "inpaint_height": ad1_ih,
        "use_steps": ad1_use_steps, "steps": ad1_steps,
        "use_cfg": ad1_use_cfg, "cfg_scale": ad1_cfg,
        "use_sampler": ad1_use_sampler, "sampler": ad1_sampler,
        "use_noise_mult": ad1_use_noise, "noise_multiplier": ad1_noise,
        "use_clip_skip": ad1_use_clip, "clip_skip": ad1_clip,
        "restore_face": ad1_restore,
        "cn_model": ad1_cn_model, "cn_weight": ad1_cn_weight,
        "cn_guidance_start": ad1_cn_start, "cn_guidance_end": ad1_cn_end,
    }
    ad2_params = {
        "enable": ad2_enable, "model": ad2_model, "prompt": ad2_prompt or "",
        "negative_prompt": ad2_neg or "", "confidence": ad2_conf,
        "mask_blur": ad2_mask_blur, "denoising_strength": ad2_ds,
        "inpaint_only_masked": ad2_iom, "inpaint_padding": ad2_iom_pad,
        "dilate_erode": ad2_dilate,
    }
    if blend_mode == "Regional Blend (Horizontal)":
        engine.apply_regional_to_processing(p)

    _setup_scripts(p, ad1_params, ad2_params)

    from contextlib import closing
    try:
        with closing(p):
            processed = process_images(p)
    except Exception as e:
        engine.cleanup()
        return [], f"Generation error: {str(e)}"

    images = processed.images if hasattr(processed, 'images') else []
    info = processed.info_string if hasattr(processed, 'info_string') else "Done"

    engine.cleanup()
    return images, info


def _make_tag_append_fn(char_prompt_ref):
    def _fn(tags, current):
        if not tags:
            return current
        new_part = ", ".join(tags)
        if current and current.strip():
            return current.strip() + ", " + new_part
        return new_part
    return _fn


def _ad_ui_group(n):
    prefix = f"ad{n}"
    components = {}
    with gr.Tab(f"1st" if n == 1 else "2nd"):
        components[f'{prefix}_model'] = gr.Dropdown(
            label="Detection Model", choices=_get_ad_model_list(),
            value="face_yolov8n.pt" if n == 1 else "None",
            elem_id=f"nai_{prefix}_model",
        )
        components[f'{prefix}_prompt'] = gr.Textbox(
            label="ADetailer Prompt (blank = use main prompt)",
            placeholder="e.g., detailed face, beautiful eyes", lines=2,
            elem_id=f"nai_{prefix}_p",
        )
        components[f'{prefix}_neg'] = gr.Textbox(
            label="ADetailer Negative Prompt",
            placeholder="e.g., bad face, deformed", lines=2,
            elem_id=f"nai_{prefix}_np",
        )
        with gr.Accordion("Detection", open=False):
            components[f'{prefix}_conf'] = gr.Slider(
                label="Detection Confidence", minimum=0.0, maximum=1.0, step=0.01, value=0.3,
                elem_id=f"nai_{prefix}_conf",
            )
            components[f'{prefix}_k'] = gr.Slider(
                label="Mask only top K largest (0=disable)", minimum=0, maximum=10, step=1, value=0,
                elem_id=f"nai_{prefix}_k",
            )
            with gr.Row():
                components[f'{prefix}_min_r'] = gr.Slider(
                    label="Mask min area ratio", minimum=0.0, maximum=1.0, step=0.001, value=0.0,
                    elem_id=f"nai_{prefix}_minr",
                )
                components[f'{prefix}_max_r'] = gr.Slider(
                    label="Mask max area ratio", minimum=0.0, maximum=1.0, step=0.001, value=1.0,
                    elem_id=f"nai_{prefix}_maxr",
                )
        with gr.Accordion("Mask Preprocessing", open=False):
            with gr.Row():
                components[f'{prefix}_x_off'] = gr.Slider(
                    label="Mask X offset", minimum=-200, maximum=200, step=1, value=0,
                    elem_id=f"nai_{prefix}_xo",
                )
                components[f'{prefix}_y_off'] = gr.Slider(
                    label="Mask Y offset", minimum=-200, maximum=200, step=1, value=0,
                    elem_id=f"nai_{prefix}_yo",
                )
            components[f'{prefix}_dilate'] = gr.Slider(
                label="Dilate/Erode (-128~128)", minimum=-128, maximum=128, step=4, value=4,
                elem_id=f"nai_{prefix}_de",
            )
            components[f'{prefix}_merge'] = gr.Radio(
                label="Mask merge mode", choices=AD_MERGE_INVERT, value="None",
                elem_id=f"nai_{prefix}_merge",
            )
        with gr.Accordion("Inpainting", open=False):
            with gr.Row():
                components[f'{prefix}_mask_blur'] = gr.Slider(
                    label="Mask Blur", minimum=0, maximum=64, step=1, value=4,
                    elem_id=f"nai_{prefix}_mb",
                )
                components[f'{prefix}_ds'] = gr.Slider(
                    label="Denoising Strength", minimum=0.0, maximum=1.0, step=0.01, value=0.4,
                    elem_id=f"nai_{prefix}_ds",
                )
            with gr.Row():
                components[f'{prefix}_iom'] = gr.Checkbox(
                    label="Inpaint Only Masked", value=True,
                    elem_id=f"nai_{prefix}_iom",
                )
                components[f'{prefix}_iom_pad'] = gr.Slider(
                    label="Inpaint Padding", minimum=0, maximum=256, step=4, value=32,
                    elem_id=f"nai_{prefix}_ip",
                )
            with gr.Row():
                components[f'{prefix}_use_wh'] = gr.Checkbox(
                    label="Use separate width/height", value=False,
                    elem_id=f"nai_{prefix}_uwh",
                )
                components[f'{prefix}_iw'] = gr.Slider(
                    label="Inpaint width", minimum=64, maximum=2048, step=4, value=512,
                    elem_id=f"nai_{prefix}_iw",
                )
                components[f'{prefix}_ih'] = gr.Slider(
                    label="Inpaint height", minimum=64, maximum=2048, step=4, value=512,
                    elem_id=f"nai_{prefix}_ih",
                )
            with gr.Row():
                components[f'{prefix}_use_steps'] = gr.Checkbox(
                    label="Use separate steps", value=False,
                    elem_id=f"nai_{prefix}_ust",
                )
                components[f'{prefix}_steps'] = gr.Slider(
                    label="ADetailer steps", minimum=1, maximum=150, step=1, value=28,
                    elem_id=f"nai_{prefix}_st",
                )
            with gr.Row():
                components[f'{prefix}_use_cfg'] = gr.Checkbox(
                    label="Use separate CFG scale", value=False,
                    elem_id=f"nai_{prefix}_ucfg",
                )
                components[f'{prefix}_cfg'] = gr.Slider(
                    label="ADetailer CFG scale", minimum=0.0, maximum=30.0, step=0.5, value=7.0,
                    elem_id=f"nai_{prefix}_cfg",
                )
            with gr.Row():
                components[f'{prefix}_use_sampler'] = gr.Checkbox(
                    label="Use separate sampler", value=False,
                    elem_id=f"nai_{prefix}_usa",
                )
                components[f'{prefix}_sampler'] = gr.Dropdown(
                    label="ADetailer sampler", choices=_get_sampler_names(),
                    value="DPM++ 2M Karras",
                    elem_id=f"nai_{prefix}_sa",
                )
            with gr.Row():
                components[f'{prefix}_use_noise'] = gr.Checkbox(
                    label="Use separate noise multiplier", value=False,
                    elem_id=f"nai_{prefix}_unm",
                )
                components[f'{prefix}_noise'] = gr.Slider(
                    label="Noise multiplier", minimum=0.5, maximum=1.5, step=0.01, value=1.0,
                    elem_id=f"nai_{prefix}_nm",
                )
            with gr.Row():
                components[f'{prefix}_use_clip'] = gr.Checkbox(
                    label="Use separate CLIP skip", value=False,
                    elem_id=f"nai_{prefix}_ucs",
                )
                components[f'{prefix}_clip'] = gr.Slider(
                    label="CLIP skip", minimum=1, maximum=12, step=1, value=1,
                    elem_id=f"nai_{prefix}_cs",
                )
            components[f'{prefix}_restore'] = gr.Checkbox(
                label="Restore face after ADetailer", value=False,
                elem_id=f"nai_{prefix}_rf",
            )
        with gr.Accordion("ControlNet", open=False):
            components[f'{prefix}_cn_model'] = gr.Dropdown(
                label="ControlNet model", choices=_get_cn_models(), value="None",
                elem_id=f"nai_{prefix}_cnm",
            )
            components[f'{prefix}_cn_weight'] = gr.Slider(
                label="ControlNet weight", minimum=0.0, maximum=1.0, step=0.05, value=1.0,
                elem_id=f"nai_{prefix}_cnw",
            )
            with gr.Row():
                components[f'{prefix}_cn_start'] = gr.Slider(
                    label="Guidance start", minimum=0.0, maximum=1.0, step=0.01, value=0.0,
                    elem_id=f"nai_{prefix}_cns",
                )
                components[f'{prefix}_cn_end'] = gr.Slider(
                    label="Guidance end", minimum=0.0, maximum=1.0, step=0.01, value=1.0,
                    elem_id=f"nai_{prefix}_cne",
                )
    return components


def _on_ui_tabs():
    engine = MultiSubjectEngine.get()

    with gr.Blocks(analytics_enabled=False, elem_id="nai_multi_subject_tab") as tab:
        with gr.Row(elem_id="nai_main_row"):
            with gr.Column(scale=2, elem_id="nai_left_panel"):
                with gr.Accordion("Characters", open=True, elem_id="nai_char_section"):
                    char_prompts = []
                    char_weights = []
                    char_enableds = []
                    char_genders = []

                    for i in range(4):
                        label = f"Character {i + 1}" if i < 2 else f"Character {i + 1} (Optional)"
                        with gr.Accordion(label, open=(i < 2), elem_id=f"nai_char_acc_{i}"):
                            with gr.Row():
                                char_enabled = gr.Checkbox(label="Active", value=(i < 2), elem_id=f"nai_char_en_{i}")
                                char_gender = gr.Radio(
                                    choices=["girl", "boy", "none"],
                                    label="Gender",
                                    value="girl" if i < 2 else "none",
                                    elem_id=f"nai_char_gender_{i}",
                                )
                            char_prompt = gr.Textbox(
                                label="Prompt (comma-separated tags)",
                                placeholder="e.g., 1girl, red hair, white dress, smile",
                                lines=2,
                                elem_id=f"nai_char_p_{i}",
                            )
                            char_weight = gr.Slider(label="Weight", minimum=0.1, maximum=2.0, step=0.05, value=1.0, elem_id=f"nai_char_w_{i}")

                            with gr.Accordion("Tag Picker", open=False, elem_id=f"nai_tag_pick_{i}"):
                                with gr.Tabs(elem_id=f"nai_tag_t_{i}"):
                                    for cat_key, cat_data in TAG_DB.items():
                                        with gr.Tab(f'{cat_data["icon"]} {cat_data["label"]}', elem_id=f"nai_tag_tab_{i}_{cat_key}"):
                                            tag_dd = gr.Dropdown(
                                                choices=cat_data["tags"],
                                                label=f"Select {cat_data['label']}",
                                                multiselect=True,
                                                elem_id=f"nai_tag_dd_{i}_{cat_key}",
                                            )
                                            tag_dd.change(
                                                fn=_make_tag_append_fn(char_prompt),
                                                inputs=[tag_dd, char_prompt],
                                                outputs=[char_prompt],
                                            )

                            char_prompts.append(char_prompt)
                            char_weights.append(char_weight)
                            char_enableds.append(char_enabled)
                            char_genders.append(char_gender)

                with gr.Accordion("Global", open=True, elem_id="nai_global_section"):
                    main_env = gr.Textbox(
                        label="Global Environment / Style",
                        placeholder="e.g., masterpiece, cinematic lighting, forest...",
                        lines=2,
                        value="masterpiece, best quality",
                        elem_id="nai_env",
                    )
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="e.g., lowres, bad anatomy, bad hands...",
                        lines=2,
                        value="lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
                        elem_id="nai_neg",
                    )

            with gr.Column(scale=1, elem_id="nai_center_panel"):
                with gr.Accordion("Generation", open=True, elem_id="nai_gen_section"):
                    checkpoint_name = gr.Dropdown(
                        label="Checkpoint (model)",
                        choices=_get_checkpoint_list(),
                        value="Use same checkpoint",
                        elem_id="nai_checkpoint",
                    )
                    with gr.Row():
                        width = gr.Slider(label="Width", minimum=256, maximum=2048, step=64, value=1024, elem_id="nai_w")
                        height = gr.Slider(label="Height", minimum=256, maximum=2048, step=64, value=1024, elem_id="nai_h")
                    with gr.Row():
                        steps = gr.Slider(label="Steps", minimum=1, maximum=150, step=1, value=28, elem_id="nai_steps")
                        cfg_scale = gr.Slider(label="CFG Scale", minimum=1.0, maximum=30.0, step=0.5, value=7.0, elem_id="nai_cfg")
                    with gr.Row():
                        sampler_name = gr.Dropdown(
                            label="Sampler",
                            choices=_get_sampler_names(),
                            value="Euler a",
                            elem_id="nai_sampler",
                        )
                    with gr.Row():
                        seed = gr.Number(label="Seed (-1 = random)", value=-1, elem_id="nai_seed")
                        batch_size = gr.Slider(label="Batch Size", minimum=1, maximum=8, step=1, value=1, elem_id="nai_batch")

                with gr.Accordion("Blend Mode", open=True, elem_id="nai_blend_section"):
                    blend_mode = gr.Radio(
                        ["Simple AND", "BREAK (Attention)", "Regional Blend (Horizontal)"],
                        label="Blend Mode",
                        value="Simple AND",
                        elem_id="nai_blend",
                    )
                    with gr.Column(visible=False, elem_id="nai_regional_group") as regional_options:
                        with gr.Row():
                            region_ratios = gr.Textbox(label="Region Ratios", placeholder="e.g., 1,1 or 2,1", value="1,1", elem_id="nai_ratios")
                            base_ratio = gr.Slider(label="Base Ratio", minimum=0.0, maximum=1.0, step=0.05, value=0.3, elem_id="nai_base_r")
                        with gr.Row():
                            feather_width = gr.Slider(label="Feather Width", minimum=0, maximum=64, step=1, value=16, elem_id="nai_feather")
                            calc_mode = gr.Radio(["Attention", "Latent"], label="Calc Mode", value="Attention", elem_id="nai_calc")
                        region_preview = gr.HTML(value=_render_region_preview([1.0, 1.0]), elem_id="nai_rprev")

                    def on_blend_change(mode):
                        return gr.Group(visible=mode == "Regional Blend (Horizontal)")

                    blend_mode.change(
                        fn=on_blend_change,
                        inputs=[blend_mode],
                        outputs=[regional_options],
                    )

                    def on_ratios_change(r):
                        return gr.HTML(value=_render_region_preview(engine.parse_ratios(r, 4)))

                    region_ratios.change(
                        fn=on_ratios_change,
                        inputs=[region_ratios],
                        outputs=[region_preview],
                    )

                with gr.Accordion("ADetailer", open=False, elem_id="nai_ad_section"):
                    ad1_enable = gr.Checkbox(
                        label="Enable ADetailer", value=False,
                        elem_id="nai_ad1_en",
                    )
                    with gr.Tabs(elem_id="nai_ad_tabs"):
                        ad1 = _ad_ui_group(1)
                        ad2 = _ad_ui_group(2)
                    ad2_enable = gr.Checkbox(
                        label="Enable 2nd ADetailer instance", value=False,
                        elem_id="nai_ad2_en",
                    )

            with gr.Column(scale=2, elem_id="nai_right_panel"):
                with gr.Row(elem_id="nai_btn_row"):
                    generate_btn = gr.Button("Generate", variant="primary", elem_id="nai_gen_btn")
                    interrupt_btn = gr.Button("Interrupt", variant="stop", visible=False, elem_id="nai_interrupt_btn")
                    skip_btn = gr.Button("Skip", visible=False, elem_id="nai_skip_btn")

                preview_html = gr.HTML(
                    value='<div id="nai_preview_container" style="display:none;text-align:center;padding:8px;"><img id="nai_preview_img" style="max-width:100%;border-radius:8px;" src=""/><p id="nai_preview_progress" style="color:#888;font-size:12px;margin-top:4px;"></p></div>',
                    elem_id="nai_preview_html",
                )

                gallery = gr.Gallery(
                    label="Output",
                    show_label=False,
                    elem_id="nai_gallery",
                )
                info_text = gr.Textbox(label="Generation Info", lines=4, elem_id="nai_info")

        ad1_keys = ['ad1_model', 'ad1_prompt', 'ad1_neg', 'ad1_conf', 'ad1_k', 'ad1_min_r', 'ad1_max_r',
                     'ad1_x_off', 'ad1_y_off', 'ad1_dilate', 'ad1_merge', 'ad1_mask_blur', 'ad1_ds',
                     'ad1_iom', 'ad1_iom_pad', 'ad1_use_wh', 'ad1_iw', 'ad1_ih',
                     'ad1_use_steps', 'ad1_steps', 'ad1_use_cfg', 'ad1_cfg', 'ad1_use_sampler', 'ad1_sampler',
                     'ad1_use_noise', 'ad1_noise', 'ad1_use_clip', 'ad1_clip', 'ad1_restore',
                     'ad1_cn_model', 'ad1_cn_weight', 'ad1_cn_start', 'ad1_cn_end']
        ad2_keys = ['ad2_model', 'ad2_prompt', 'ad2_neg', 'ad2_conf',
                     'ad2_mask_blur', 'ad2_ds', 'ad2_iom', 'ad2_iom_pad', 'ad2_dilate']

        interrupt_btn.click(fn=_interrupt, inputs=[], outputs=[])
        skip_btn.click(fn=_skip, inputs=[], outputs=[])

        gen_inputs = [
            char_prompts[0], char_prompts[1], char_prompts[2], char_prompts[3],
            char_weights[0], char_weights[1], char_weights[2], char_weights[3],
            char_enableds[0], char_enableds[1], char_enableds[2], char_enableds[3],
            char_genders[0], char_genders[1], char_genders[2], char_genders[3],
            main_env, negative_prompt, blend_mode,
            region_ratios, base_ratio, feather_width, calc_mode,
            width, height, steps, cfg_scale, sampler_name, seed, batch_size,
            checkpoint_name,
            ad1_enable,
            *[ad1[k] for k in ad1_keys],
            ad2_enable,
            *[ad2[k] for k in ad2_keys],
        ]

        generate_btn.click(
            fn=_on_generate_start,
            inputs=[],
            outputs=[generate_btn, interrupt_btn, skip_btn],
        ).then(
            fn=_generate,
            inputs=gen_inputs,
            outputs=[gallery, info_text],
        ).then(
            fn=_on_generate_end,
            inputs=[],
            outputs=[generate_btn, interrupt_btn, skip_btn],
        )

    return [(tab, "Multi-Subject", "nai_multi_subject_tab")]


on_ui_tabs(_on_ui_tabs)
