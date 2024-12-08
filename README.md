# Singing Voice Conversion using CycleGAN-VC
This project is submitted for the partial fulfilment of the requirements for the course CSE4201 - Generative Adversarial Networks to Dr Ankit Jha by-
- Vamsikrishna Konthala (21ucs229)
- Swayam Bhatt (21ucs217)
- Shreshta Gupta (21ucs196)
- Ayush Bajaj (21ucc129)

***
The objective of this project is to use CycleGAN-VC to convert Arijit Singh's voice to Kishore Kumar's. CycleGAN-VC is a novel non-parallel voice conversion method leveraging CycleGANs to learn mappings between source and target speech without relying on parallel data. (View CycleGAN-VC [here](https://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/cyclegan-vc/)).

The steps used in this project are-

1. Collection of songs- We collected about 40 songs each for Arijit Singh and Kishore Kumar.
2. Vocal Separation- Using [spleeter](https://github.com/deezer/spleeter) we isolated the vocals from the songs dataset.
3. Conversion to Spectrogram- The vocal audio files were analyzed using Librosa to generate Mel-Spectrograms.
4. Training - The spectrograms were used to train the model for about 1300 epochs.

More details about the project can be found in the [presentation]()
