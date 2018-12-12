---
layout: default
---

# Abstract

Most video games use static soundtracks, which although can be impressive in their own right, can quickly become overplayed to the gamer. In this project, we built a complete system that can process, learn, and generate new video game music. Our system takes in MIDI files, which are processed and trained as a Markov Chain. Finally, we use our Markov Chain to generate a multi-voice music sample. We optimize our programs by using CUDA techniques in our training algorithm, and multi-threading in the music generation step. Speedup analysis shows that our system performs significantly better than a sequential algorithm. In addition, because our system is fast, we explore the possibility of integrating our music generation method into video games, so that games can have real-time generated music for a more interesting gaming experience.

# Introduction

Through the forms and rules outlined in music theory, humans already follow many rigid guidelines when composing music. Therefore, it seems natural to extend this technique to using an algorithmic composer. In particular, the genre of video game music, which is characterized by the overlapping of multiple melodic lines and a repetitive and structured format, seems ideal as a starting point for parallel algorithmic composers. The many melodic lines offer ample avenues for parallelization, allowing for parallel synthesis of different parts. In addition, the rigid structure makes such music easier to predict and generate with a Markov Model — resulting in higher quality and more harmonious music.

The Markov Model of music generation was chosen for its popularity, good musical results, and ease of implementation, understanding, and parallelization. The model is easy to understand and implement, as it is based on using many probability matrices to determine the next notes of a melodic line. These matrices are easy to store and manipulate in many different programming contexts, including the GPUs we wish to use. Other music generation algorithms introduce too many new challenges, such as the recursive nature of Grammars (which may be difficult as it is not known beforehand how may recursions/threads we would need to spawn), or the poor music production of Structure and Probability techniques (Dannenberg, 2018). Finally, it has been shown that Markov Models produce good musical results, where computer compositions and human compositions had similar rankings in formal listening tests (Elowsson and Friderg, 2012). Therefore, implementing a Markov Model for our project would be the best choice for producing music with integrity and quality.

In a Markov Model, multiple matrices are built from training data, or files of music that fits the patterns and genre of the music we wish to create. Most commonly, this takes the form of counting the number of transitions from one note to another, and storing this count in a matrix. These matrices are then normalized and turned into probability matrices. For each melodic line, the algorithm uses the current (and perhaps some number of past) notes to determine the correct matrix and matrix row to look at. It then uses that matrix section to determine the next note, based on the probabilities in the matrix. Because the matrices are training on music of the same structure and genre, it is likely that the resulting melodies also follow the same structure and sound similar to the training pieces. The resulting melodies are then combined and outputted through MIDI.

# Implementation

Our project consisted of two major portions:

1. Training: preprocessing MIDI files and training a Markov Model based on these MIDI files.
2. Generation: using the Markov Model that was trained to generate music. 

## Training

### Preprocessing

### Markov Model

## Generation

### Server

### Client

# Results

Our speedup results indicate...

# Conclusion

# Future Work

A major goal of video games is to give the gamer an immersive experience in the virtual world. One common feature is realistic graphics and physics in the game. A feature that has been seldom explored is dynamic music that adapts to the context of the game. A possible extension of our project is to integrate our music generation algorithm into a video game, where the algorithm not only uses trained data, but also the current state of the video game to generate music. Such dynamically-generated music would have a similar effect to how soundtracks in movies are synchronized to actions. Such synchronized soundtracks are very effective, because they would allow for the game sound to be a form of live feedback to the gamer.

# References

- Elowsson, A. and Friberg, A. “Algorithmic Composition of Popular Music,” in Proceedings of the 12th International Conference on Music Perception and Cognition and the 8th Triennial Conference of the European Society for the Cognitive Sciences of Music, July 2012, pp 276-285
- Dannenberg, R. (2018). Music Generation and Algorithmic Composition [Powerpoint slides]. Retrieved from http://www.cs.cmu.edu/~./music/cmsip/slides/05-algo-comp.pdf

# Work Distribution