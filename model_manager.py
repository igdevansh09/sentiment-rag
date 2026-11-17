from hardware_preparation import gpu_preparation
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from hugging_face_authentication import hugging_face_auth
import torch

class ModelManager:
    DEFAULT_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
    
    def __init__(self, model_id: str = DEFAULT_MODEL_ID, quantization_setting = True):
        if gpu_preparation():
            hugging_face_auth()
            self.model_id  = model_id
            self.tokenizer = self.loading_tokenizer()
            self.model = self.loading_model(self.model_id, quantization_setting)
        else:
            raise EnvironmentError("No GPU detected. This model requires a CUDA-compatible GPU to run.")
        
    def loading_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
    
    def quantization_configuration(self):
        return BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                    )
        
    def loading_model(self,model_id,quantization_setting):
        kwargs = {
            "device_map": "auto",
            "dtype": "auto",
            "trust_remote_code": True,
        }

        if quantization_setting:
            kwargs["quantization_config"] = self.quantization_configuration()
        
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    
    def __repr__(self) -> str:
        return (
            f"ModelManager("
            f"model_id='{self.model_id}', "
            f"quantization={self.tokenizer}, "
            f"model={self.model}, "
            f"device='{torch.cuda.get_device_name(0)}')"
        )