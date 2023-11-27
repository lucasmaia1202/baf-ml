# baf-ml

Machine Learning Modeling for Task #1 of Knowledge Discovery on Databases Course on MAPi.


---
## Installation Dependencies

```bash
conda create -n baf-ml python=3.12
conda activate baf-ml
pip install -r requirements.txt
```

---
## Steps of the project

```mermaid
graph LR
    A[Data] --> B[Preprocessing]
    B --> C[Modeling]
    C --> D[Evaluation]
    D --> E[Results]
```