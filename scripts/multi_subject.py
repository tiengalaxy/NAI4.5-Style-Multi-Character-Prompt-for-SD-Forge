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

    def build_final_prompt(self, active_prompts, env, blend_mode):
        combined = []
        for prompt in active_prompts:
            parts = [prompt]
            if env:
                parts.append(env)
            combined.append(" ".join(parts))
        if blend_mode == "Simple AND":
            return " AND ".join(combined)
        elif blend_mode in ("BREAK (Attention)", "Regional Blend (Horizontal)"):
            return " BREAK ".join(combined)
        return " AND ".join(combined)

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


def _get_lora_names():
    try:
        lora_dir = shared.cmd_opts.lora_dir if hasattr(shared.cmd_opts, 'lora_dir') else None
        if not lora_dir:
            return []
        import os
        names = []
        for f in sorted(os.listdir(lora_dir)):
            full = os.path.join(lora_dir, f)
            if os.path.isdir(full):
                names.append(f)
            elif f.endswith(('.safetensors', '.pt', '.ckpt', '.bin')):
                names.append(os.path.splitext(f)[0])
        return names
    except Exception:
        return []


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


def _generate(
    cp0, cp1, cp2, cp3,
    cw0, cw1, cw2, cw3,
    ce0, ce1, ce2, ce3,
    cg0, cg1, cg2, cg3,
    main_env, negative_prompt, blend_mode,
    region_ratios, base_ratio, feather_width, calc_mode,
    width, height, steps, cfg_scale, sampler_name, seed, batch_size,
    ln0, ln1, ln2, ln3,
    lw0, lw1, lw2, lw3,
    le0, le1, le2, le3,
):
    char_prompts = [cp0, cp1, cp2, cp3]
    char_weights = [cw0, cw1, cw2, cw3]
    char_enableds = [ce0, ce1, ce2, ce3]
    char_genders = [cg0, cg1, cg2, cg3]
    lora_names = [ln0, ln1, ln2, ln3]
    lora_weights = [lw0, lw1, lw2, lw3]
    lora_enableds = [le0, le1, le2, le3]

    engine = MultiSubjectEngine.get()
    engine.cleanup()

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

    lora_tags = []
    for i, name in enumerate(lora_names):
        w = lora_weights[i] if i < len(lora_weights) else 1.0
        en = lora_enableds[i] if i < len(lora_enableds) else False
        if en and name and name.strip():
            lora_tags.append(f"<lora:{name.strip()}:{w:.2f}>")
    lora_str = " ".join(lora_tags)

    final_prompt = engine.build_final_prompt(active_prompts, env, blend_mode)

    prompt_parts = []
    if count_str:
        prompt_parts.append(count_str)
    prompt_parts.append(final_prompt)
    final_prompt = ", ".join(prompt_parts)

    if lora_str:
        final_prompt = f"{lora_str} {final_prompt}"

    neg = negative_prompt.strip() if negative_prompt else ""

    if blend_mode == "Regional Blend (Horizontal)":
        ratios = engine.parse_ratios(region_ratios, len(active_prompts))
        engine.setup_regional(active_prompts, env, ratios, base_ratio, feather_width, calc_mode)

    outdir_samples = getattr(shared.cmd_opts, "outdir_txt2img_samples", None) or getattr(shared.cmd_opts, "outdir_txt2img", None) or "outputs/txt2img-images"
    outdir_grids = getattr(shared.cmd_opts, "outdir_txt2img_grids", None) or getattr(shared.cmd_opts, "outdir_grids", None) or "outputs/txt2img-grids"
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

    try:
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
                    with gr.Group(visible=False, elem_id="nai_regional_group") as regional_options:
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

                with gr.Accordion("LoRA", open=False, elem_id="nai_lora_section"):
                    lora_names_list = []
                    lora_weights_list = []
                    lora_enableds_list = []

                    for i in range(4):
                        with gr.Row(elem_id=f"nai_lora_row_{i}"):
                            lora_en = gr.Checkbox(label="On", value=False, elem_id=f"nai_lora_en_{i}")
                            lora_name = gr.Dropdown(
                                choices=_get_lora_names(),
                                label=f"LoRA {i + 1}",
                                value=None,
                                elem_id=f"nai_lora_n_{i}",
                            )
                            lora_wt = gr.Slider(label="Weight", minimum=-2.0, maximum=2.0, step=0.05, value=1.0, elem_id=f"nai_lora_w_{i}")
                            lora_names_list.append(lora_name)
                            lora_weights_list.append(lora_wt)
                            lora_enableds_list.append(lora_en)

                    def refresh_loras():
                        names = _get_lora_names()
                        return [gr.Dropdown(choices=names) for _ in lora_names_list]

                    refresh_btn = gr.Button("Refresh LoRA List", elem_id="nai_lora_refresh")
                    refresh_btn.click(fn=refresh_loras, outputs=lora_names_list)

            with gr.Column(scale=2, elem_id="nai_right_panel"):
                generate_btn = gr.Button("Generate", variant="primary", elem_id="nai_gen_btn")

                gallery = gr.Gallery(
                    label="Output",
                    show_label=False,
                    elem_id="nai_gallery",
                )
                info_text = gr.Textbox(label="Generation Info", lines=4, elem_id="nai_info")

        generate_btn.click(
            fn=_generate,
            inputs=[
                char_prompts[0], char_prompts[1], char_prompts[2], char_prompts[3],
                char_weights[0], char_weights[1], char_weights[2], char_weights[3],
                char_enableds[0], char_enableds[1], char_enableds[2], char_enableds[3],
                char_genders[0], char_genders[1], char_genders[2], char_genders[3],
                main_env, negative_prompt, blend_mode,
                region_ratios, base_ratio, feather_width, calc_mode,
                width, height, steps, cfg_scale, sampler_name, seed, batch_size,
                lora_names_list[0], lora_names_list[1], lora_names_list[2], lora_names_list[3],
                lora_weights_list[0], lora_weights_list[1], lora_weights_list[2], lora_weights_list[3],
                lora_enableds_list[0], lora_enableds_list[1], lora_enableds_list[2], lora_enableds_list[3],
            ],
            outputs=[gallery, info_text],
        )

    return [(tab, "Multi-Subject", "nai_multi_subject_tab")]


on_ui_tabs(_on_ui_tabs)
