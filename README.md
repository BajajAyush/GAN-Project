# Singing Voice Conversion using CycleGAN-VC
This project is submitted for the partial fulfilment of the requirements for the course CSE4201 - Generative Adversarial Networks to Dr Ankit Jha by-
- Vamsikrishna Konthala (21ucs229)
- Swayam Bhatt (21ucs217)
- Shreshta Gupta (21ucs196)
- Ayush Bajaj (21ucc129)

***
The objective of this project is to use CycleGAN-VC to convert Arijit Singh's voice to Kishore Kumar's. CycleGAN-VC is a novel non-parallel voice conversion method leveraging CycleGANs to learn mappings between source and target speech without relying on parallel data. (View CycleGAN-VC [here](https://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/cyclegan-vc/)).

The steps used in this project are-

1. Collection of songs- We collected 40 songs each for Arijit Singh and Kishore Kumar ([dataset](https://www.kaggle.com/datasets/bajajayush/gan-audio/)).
2. Vocal Separation- Using [spleeter](https://github.com/deezer/spleeter) we isolated the vocals from the songs dataset.
3. Conversion to Spectrogram- The vocal audio files were analyzed using Librosa to generate Mel-Spectrograms.
4. Training - The spectrograms were used to train the model for about 1300 epochs taking about 30 hours of training time.

More details about the project can be found in the [presentation](presentation.pdf)

### File Structure

```bash
.\
├── data_preprocess.ipynb
├── model-to-audio.ipynb
├── presentation.pdf
└── trainining.py
```

The Kaggle notebooks can be viewed here-

1. [data_preprocess.ipynb](https://www.kaggle.com/code/bajajayush/gan-song)
2. [model-to-audio.ipynb](https://www.kaggle.com/code/bajajayush/model-to-audio)

***

## Sample Output

The spectrogram outputs for the song "Ye Fitoor Mera"-

Input-

![image](https://github.com/user-attachments/assets/608be439-f21a-452b-8dae-789a0457e006)

Output at Epoch 294-

![image](https://github.com/user-attachments/assets/043b0b02-d754-4580-98d5-8899ccec7947)

Output at Epoch 1355-

![image](https://github.com/user-attachments/assets/438c4752-dd5a-4357-a424-6a54baea4e10)

Input Song-

![youtube](https://music.youtube.com/watch?v=Zkqhiil2kSo&si=63CAnuotD5DFcyOv)

Output Audio-

![sound cloud](https://on.soundcloud.com/H7JGmhVG9raHtgUu7)

The model has learned to generate lower frequencies correctly but still needs more training to perfectly imitate Kishore Kumar's vocals.
