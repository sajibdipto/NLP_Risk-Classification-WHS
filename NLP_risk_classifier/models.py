import torch
from torch import nn
from transformers import RobertaModel

# Risk model

class RobertaRisk(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.pooler_output
        return self.classifier(pooled)


# Multi-task model

class RobertaMultiTask(nn.Module):
    def __init__(self, num_classes_dict, tasks):
        super().__init__()
        self.tasks = tasks
        self.roberta = RobertaModel.from_pretrained("roberta-base")

        self.heads = nn.ModuleDict({
            t: nn.Linear(768, num_classes_dict[t])
            for t in tasks
        })

    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.pooler_output

        return {t: self.heads[t](pooled) for t in self.tasks}
