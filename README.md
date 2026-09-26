# Fantasy AI - Predictive ML Web App (POC)

**Description:** A Python-based web application and machine learning pipeline leveraging `nfl_data_py`, `scikit-learn`, and deep learning to forecast weekly and seasonal NFL fantasy points.

This repository contains the Proof of Concept demonstrating the integration of NFL data pipelines with Python-based machine learning libraries, served dynamically through a web interface.

## Architecture & Conceptual Design

### Use Case Diagram
<img width="491" height="378" alt="image" src="https://github.com/user-attachments/assets/f0391d87-fe28-4880-b8ae-ad50d8a42c61" />


### Class Diagram
<img width="493" height="672" alt="image" src="https://github.com/user-attachments/assets/5e77d488-ffa6-45b4-b45a-7c62b0f4749a" />


## Prerequisites & Environment
This project requires an Anaconda environment to resolve package dependencies. The proof of concept was built and tested using **Python 3.10**.

### Required Libraries:
* `nfl_data_py` (NFL play-by-play data fetching)
* `pandas` (Data manipulation)
* `scikit-learn` (Machine learning models)
* `flask` (Web framework for the frontend)
* `tensorflow` / `pytorch` (Included for future deep learning capabilities)

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/MillerAlexander/fantasy-ml.git](https://github.com/MillerAlexander/fantasy-ml.git)
   cd fantasy-ml
