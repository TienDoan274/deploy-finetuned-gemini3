- Finetune notebook: https://www.kaggle.com/code/nhttinon/finetune-gemma3-lab1-mlops

- link demo: https://drive.google.com/drive/folders/12102pA3fDC6P30wnslZUigrDaNTNMRQ5?usp=sharing



After finetuning model, model is saved as gguf file, you will have to put it in a folder call gguf_model:

```
.
├── Dockerfile
├── README.md
├── app.py
├── docker-compose.yaml
├── gguf_model
│   └── gemma3-unsloth-lora.gguf <--
├── inference.py
├── knowledge_base.json
└── requirements.txt
```

Then just run: 

```
docker compose up
```

You can access the model at: http://localhost:7860