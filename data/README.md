# Data Directory

This directory stores datasets for the **Phishing Message Detector & Overfitting Study**:

- `raw/`: Stores raw dataset files downloaded from Hugging Face or public repositories (`Phishing_Email.csv`).
- `processed/`: Stores cleaned, validated, deduplicated, and split datasets (`train.csv`, `val.csv`, `test.csv`).
- `sample_synthetic.csv`: Small test dataset used solely for automated unit testing and interface smoke testing.

## Dataset Source
- **Repository**: [zefang-liu/phishing-email-dataset](https://huggingface.co/datasets/zefang-liu/phishing-email-dataset)
- **Original Mirror**: Kaggle Phishing Email Detection Dataset
- **License**: Public Domain / Open Research (Apache 2.0 / CC-BY)
- **Original Columns**:
  - `Email Text`: Raw message body and subject
  - `Email Type`: Original labels ("Safe Email", "Phishing Email")
  - `Unnamed: 0`: Raw index
- **Normalized Schema**:
  - `text`: Cleaned message text (non-empty string)
  - `label`: Binary target (1 = Phishing, 0 = Legitimate)
  - `label_name`: "Phishing" or "Legitimate"
