

from transformers import TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training

from types import MethodType

import logging

class CheckpointLoadingWarningFilter(logging.Filter):
    '''
    This filter removes a known warning where checkpoint biases are not expected by the model
    This is assumed not to affect the performance.
    '''
    def filter(self, record):
        message_str = record.getMessage()
        return not message_str.startswith('Some weights of the model checkpoint at ')

class QuantLinearExllamaWarningFilter(logging.Filter):
    '''
    This filter removes a known warning where an exllama backend is installed
    the backend will subsequently be replaced regardless with a TritonV2 backend
    '''
    def filter(self, record):
        message_str = record.getMessage()
        prefix = 'QuantLinear with the exllama backend not does support the trainable mode yet, switching to cuda/cuda_old/triton backend'
        return not (prefix in message_str)

logger = logging.getLogger('auto_gptq.utils.accelerate_utils')
logger.addFilter(CheckpointLoadingWarningFilter())
logger = logging.getLogger('auto_gptq.modeling._base')
logger.addFilter(QuantLinearExllamaWarningFilter())

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from auto_gptq.utils.peft_utils import get_gptq_peft_model
from auto_gptq.utils.peft_utils import GPTQLoraModel
from auto_gptq.nn_modules.qlinear.qlinear_tritonv2 import QuantLinear as triton_Qlinear

from peft.tuners.lora.gptq import QuantLinear as LoraLinearGPTQ
from peft.tuners.lora.model import LoraModel

from typing import Tuple, Dict

from .framework_plugin import AccelerationPlugin

from functools import partial
import torch

class AutoGPTQAccelerationPlugin(AccelerationPlugin):
    
    require_packages = ['auto_gptq']

    def __init__(self, configurations: Dict[str, Dict]):
        super().__init__(configurations)

        # just do checking, nothing must to configure at this point
        # if need to configure then do something like this:
        # self.kernel = self._get_config_value("peft.kernel")
        self._check_config_equal(key="peft.quantization", value="auto_gptq")
        self._check_config_equal(key="peft.quantization.auto_gptq.kernel", value="triton_v2")
        self._check_config_equal(key="peft.quantization.auto_gptq.from_quantized", value=True)

    def model_loader(self, model_name: str, **kwargs):

        # guarded imports
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

        # Currently we allow only a quantized checkpoint to be loaded, we do not
        # implement the quantization process here. 
        #
        # The quantization process is used to convert a non-quantized checkpoint
        # (provided in model_name) into a quantized one. This entails
        # 1. providing a BaseQuantizeConfig with the appropriate quantization settings
        # 2. calling BaseGPTQForCausalLM.quantize to run the quantization algorithm (may take time, e.g. hours)
        # 3. calling BaseGPTQForCausalLM.save_pretrained to save a quantized checkpoint
        # 
        # The reasons for not implementing the flow at this point are.
        # 1. The quantization can take very long for large models. As such, it is more appropriate
        #    to run it once outside of training, and save the checkpoint to be used for multiple runs.
        # 2. Requires some API changes to point to where the quantized checkpoint should be saved.
        #    Can be confusing to the user since it will be different from model_name

        # NOTE: there will be a warning that can be ignored
        # "WARNING - QuantLinear with the exllama backend not does support the trainable mode yet, switching to cuda/cuda_old/triton backend."

        # assume model_name points to a quantized checkpoint. Thus we load the quantization
        # config directly from the checkpoint.
        quantize_config = BaseQuantizeConfig.from_pretrained(model_name)

        # get additional parameters
        torch_dtype = kwargs.get('torch_dtype', torch.float32)
        low_cpu_mem_usage = kwargs.get('low_cpu_mem_usage', False)

        model = AutoGPTQForCausalLM.from_quantized(
            model_name, 
            quantize_config=quantize_config,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
            use_marlin=False, # disable, cannot be used for training (no forward+backward)
            disable_exllama=True, # disable, cannot be used for training (no backward)
            warmup_triton=False, # disable for now, because it will try to run the warmup while on CPU
            use_tritonv2=True,
            trainable=True, # only support trainable mode
        )           

        # these will be properly set since it is not loaded using from_pretrained
        # - so, set them here. 
        # - in particular "is_loaded_in_4bit" will be checked in prepare_model_for_kbit_training
        #   and there is a section of code that will be skipped if not set.
        setattr(model, "is_loaded_in_4bit", True)
        setattr(model, "quantization_method", 'gptq')

        return model

    @property
    def requires_custom_loading(self):
        return True

    @property
    def requires_agumentation(self):
        return True

    def augmentation(
        self, 
        model, 
        train_args: TrainingArguments,
        modifiable_args: Tuple[LoraConfig],
    ):
        # guarded imports
        from auto_gptq.utils.peft_utils import get_gptq_peft_model
        from auto_gptq.utils.peft_utils import GPTQLoraModel
        from auto_gptq.nn_modules.qlinear.qlinear_tritonv2 import QuantLinear
        from ..plugin_utils.autogptq_utils import create_new_module_peft, replace_module_peft

        peft_config, = modifiable_args # unpack modifiable args

        # some assertions
        assert peft_config is not None, "need peft_config to install PEFT adapters"
        assert model.dtype == torch.float16 or train_args.fp16,\
             "need to run in fp16 mixed precision or load model in fp16"

        # call the prepare_model_for_kbit_training. This will no longer be called
        # inside SFTTrainer, because we eventually return None for the peft_config.
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=train_args.gradient_checkpointing,
            gradient_checkpointing_kwargs=train_args.gradient_checkpointing_kwargs,
        )

        # These functions need to replaced due to some incompatibliites 
        # with newer PEFT packages.
        # - on augmentation we call auto_gptq.utils.peft_utils.get_gptq_peft_model
        # - this internally calls peft.utils.other.get_peft_model
        # - however the problem is that peft API moves very fast, and there are incompatiblities
        # 
        # During peft wrapping there are two key operations
        # 1. LoraModel._create_new_module is called to create a LoraLinear layer that is
        #    compatible with the base layer. For quantized base layers, the LoraLinear
        #    may be different.
        # 2. GPTQLoraModel._replace_module to replace the existing Linear with the LoraLinear.
        #    Also move to device (which may depend on how base layer is implemented)

        # NOTE: GPTQLoraModel inherits from LoraModel, and the _create_new_module method is called
        # on the parent. Hence _create_new_module is patched on the parent

        # FIXME: 
        # 1. investigate using BaseGPTQForCausalLM.make_sure_compatible_with_peft
        #    to see if we can get around the patching

        _old_create_new_module = LoraModel
        _old_replace_module = GPTQLoraModel._replace_module
        _create_new_module = partial(create_new_module_peft, target_cls=QuantLinear)
        LoraModel._create_new_module = staticmethod(_create_new_module)
        GPTQLoraModel._replace_module = MethodType(replace_module_peft, GPTQLoraModel)

        # Install GPTQ adapters using the AutoGPTQ package (with the above patches)
        model = get_gptq_peft_model(
            model, 
            peft_config=peft_config, 
            auto_find_all_linears=peft_config.target_modules is None,
            train_mode=True, # install adapaters for training
        )
        modifiable_args = (None, ) # return a None for peft_config

        # undo the patching for hygine
        LoraModel._create_new_module = staticmethod(_old_create_new_module)
        GPTQLoraModel._replace_module = MethodType(_old_replace_module, GPTQLoraModel)

        from tuning.acceleration.plugin_utils import unsloth_utils
        model = unsloth_utils.add_unsloth_improvements(model)
        return model, modifiable_args

# register
AccelerationPlugin.register_plugin(
    AutoGPTQAccelerationPlugin,
    configuration_and_paths=["peft.quantization.auto_gptq"], 
)