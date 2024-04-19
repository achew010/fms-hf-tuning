from types import MethodType
from transformers.utils.import_utils import _is_package_available

from peft.peft_model import PeftModelForCausalLM

has_xformers = _is_package_available("xformers")
has_unsloth = _is_package_available("unsloth")

if has_xformers and has_unsloth:

    # unsloth llama imports
    from unsloth.models.llama import (
        LlamaModel_fast_forward_inference,
        LlamaDecoderLayer_fast_forward, 
        CausalLM_fast_forward,
        LlamaModel_fast_forward,
        LlamaAttention_fast_forward,
    )

    # unsloth mistral imports
    from unsloth.models.mistral import (
        MistralAttention_fast_forward, MistralForCausalLM_fast_forward
    )

    from unsloth.models.mixtral import (
        MistralAttention_fast_forward, 
        MixtralForCausalLM_fast_forward,
        MixtralModel_fast_forward,
        MixtralDecoderLayer_fast_forward,
    )

    # from unsloth.models.gemma import (
    #     GemmaDecoderLayer_fast_forward,
    #     GemmaModel_fast_forward_inference,
    #     GemmaFixedRotaryEmbedding,
    # )

    # unsloth gptq fast_lora imports
    from unsloth.gptq.fast_lora import apply_lora_qkv, apply_lora_o, apply_lora_mlp

    UNSLOTH_FAST_FORWARDS = [
        (
            {'LlamaForCausalLM'}, 
            (
                CausalLM_fast_forward(LlamaModel_fast_forward_inference), 
                ('model', LlamaModel_fast_forward), 
                LlamaDecoderLayer_fast_forward,
                (
                    'self_attn', LlamaAttention_fast_forward,
                    apply_lora_qkv, apply_lora_o, 
                    'mlp', apply_lora_mlp,
                ), 
            )
        ),
        # (
        #     {'GemmaForCausalLM'}, 
        #     (
        #         CausalLM_fast_forward(GemmaModel_fast_forward_inference), 
        #         ('model', LlamaModel_fast_forward), 
        #         GemmaDecoderLayer_fast_forward,
        #         (
        #             'self_attn', LlamaAttention_fast_forward,
        #             apply_lora_qkv, apply_lora_o, 
        #             'mlp', apply_lora_mlp,
        #         ), 
        #     )
        # ),
        (
            {'MistralForCausalLM'}, 
            (
                MistralForCausalLM_fast_forward, 
                ('model', LlamaModel_fast_forward), 
                LlamaDecoderLayer_fast_forward,
                (
                    'self_attn', MistralAttention_fast_forward,
                    apply_lora_qkv, apply_lora_o, 
                    'mlp', apply_lora_mlp,
                ), 
            )
        ),
        (
            {'MixtralForCausalLM'}, 
            (
                MixtralForCausalLM_fast_forward, 
                ('model', MixtralModel_fast_forward), 
                MixtralDecoderLayer_fast_forward,
                (
                    'self_attn', MistralAttention_fast_forward,
                    apply_lora_qkv, apply_lora_o, 
                    'mlp', apply_lora_mlp,
                ), 
            )
        )
    ]

    from typing import List, Set, Tuple, Any
    def _find_arch(architectures: List[str], artifacts: List[Tuple[Set, Any]]):
        for keys, _artifacts in artifacts:
            if any([arch in keys for arch in architectures]):
                return _artifacts

        return None

    def _check_child_modules_for_patch_compatibility(parent, target_child_modules):
        from peft.tuners.lora import LoraLayer
        # 1. ensure the modules_to_check is present in self_attn and are lora layers
        existing_lora_layers_children = [name for name, child in parent.named_children() if isinstance(child, LoraLayer)]
        missing_modules = [name for name in target_child_modules if name not in existing_lora_layers_children]
        if len(missing_modules)>0:
            # print(f'''The following LoraLayers {missing_modules} are not present in {parent.__class__.__name__} module, will skip patching {parent.__class__.__name__} layer''')
            return False

        # 2. ensure no bias and magnitude vector in self_attn children, 
        # if any child has bias or magnitude vector returns False
        for child in parent.children():
            if isinstance(child, LoraLayer):
                bias = getattr(child, "base_layer", child).bias
                has_no_bias = (bias is None or bias.count_nonzero()==0)
                magnitude_vector = getattr(child, "lora_magnitude_vector", None)
                _no_bias_magnitude_attr = (has_no_bias and magnitude_vector is None)        
                if not _no_bias_magnitude_attr:
                    return False
        return True                          

    # add improvements to a PeftModelForCausalLM
    # - fused ops
    # - rms layer norm
    # - RoPE embeddings
    # - causallm cross-entropy loss
    # function to convert the model
    def add_unsloth_improvements(
        model: PeftModelForCausalLM, 
        adapter_name: str = 'default',
    ):

        # some checks
        _is_lora_peft = (
            hasattr(model, "peft_config") and 
            model.peft_config[adapter_name].peft_type.value == 'LORA'
        )

        base_model = model.get_base_model()

        # config
        config = base_model.config
        base_model.max_seq_length = config.max_position_embeddings # the forward needs it

        # fetch artifacts
        artifacts = _find_arch(config.architectures, UNSLOTH_FAST_FORWARDS)
        if artifacts is None:
            raise ValueError(f"No unsloth improvements for any architectures in \'{config.architectures}\'")

        if not hasattr(base_model, '_no_split_modules') or base_model._no_split_modules is None:
            raise ValueError(
                "Only can install unsloth improvements in PreTrainedModels with _no_split_modules"
            )
        
        (
            _causal_f, (_bbname, _backbone_f), _decoder_f, 
            (_attn_name, _attn, _lqkv, _lo, _mlp_name, _lmlp), 
        ) = artifacts
        _no_split_modules = base_model._no_split_modules

        # for layer in base_model._get_no_split_modules():
        for layer in base_model.modules():
            if layer.__class__.__name__ not in _no_split_modules:
                continue

            self_attn = getattr(layer, _attn_name)
            mlp = getattr(layer, _mlp_name)
            self_attn.forward = MethodType(_attn, self_attn)
            if _is_lora_peft:
                # TODO: check if there are adapters 
                if _check_child_modules_for_patch_compatibility(self_attn, target_child_modules=['q_proj', 'k_proj', 'v_proj']):
                    self_attn.apply_qkv = _lqkv
                if _check_child_modules_for_patch_compatibility(self_attn, target_child_modules=['o_proj']):
                    self_attn.apply_o = _lo
                if _check_child_modules_for_patch_compatibility(mlp, target_child_modules=['gate_proj','down_proj','up_proj']):
                    mlp.forward = MethodType(_lmlp, mlp) # simulate no patching for now
            layer.forward = MethodType(_decoder_f, layer)

        backbone = getattr(base_model, _bbname)
        backbone.forward = MethodType(_backbone_f, backbone)
        base_model.forward = MethodType(_causal_f, base_model)
        return model
