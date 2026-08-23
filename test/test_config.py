from transformers import AutoConfig

config = AutoConfig.from_pretrained(
    "shared/ART_Models/attention_mixing_probe",
    trust_remote_code=True,
)

print("Config class:", type(config).__name__)
print(config)