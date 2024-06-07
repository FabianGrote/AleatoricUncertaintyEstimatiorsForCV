if finetuning:
    # finetuning
    # only optimize last/ classifier layer
    optimizer_params  = self.model.module.classifier.parameters()
    initial_lr= self.CONFIG["train"]["optimizer"]["finetune_lr"]

else:
    # let optimizer change all parameters of the model
    optimizer_params = self.model.parameters()
    initial_lr  =self.CONFIG["train"]["optimizer"]["lr"]