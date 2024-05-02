from transformers import AutoModel
import torch
# Load the model
#model = AutoModel.from_pretrained("/shared/3/projects/spatial-understanding/yijun-ckpts/step_18000")

#torch.save(model, "full_model.pt")
state_dict = torch.load("/home/panyijun/direct-preference-optimization/full_model.pt", map_location='cpu').state_dict()
print(state_dict.keys())
step, metrics = state_dict['step_idx'], state_dict['metrics']