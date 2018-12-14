---
layout: default
---

# Abstract

Most video games use static soundtracks, which although can be impressive in their own right, can quickly become overplayed to the gamer. In this project, we built a complete system that can process, learn, and generate new video game music. Our system takes in MIDI files, which are processed and trained as a Markov Chain. Finally, we use our Markov Chain to generate a multi-voice music sample. We optimize our programs by using CUDA techniques in our training algorithm, and multi-threading in the music generation step. Speedup analysis shows that our system performs significantly better than a sequential algorithm. In addition, because our system is fast, we explore the possibility of integrating our music generation method into video games, so that games can have real-time generated music for a more interesting gaming experience.

# Introduction

Through the forms and rules outlined in music theory, humans already follow many rigid guidelines when composing music. For example, there are basic rules of tonality and music structure that can decide whether a piece of music sounds good or clashing to the year (CITATION about music theory and tonality). Therefore, it seems natural to extend these music theory rules to build an algorithmic composer. In particular, the genre of video game music, which in the early 1970s and 80s was originally designed for auxiliary reasons, such as a "heart rate changer" (NPR), and was limited in technology to 8-bit sound (Collins, 2007), is often characterized by a repetitive and structured format, which is ideal as a starting point for algorithmic composers.

On the other hand, in modern times, video game music has evolved to serve a larger purpose in the gaming experience. For example, Whalen discusses how music in the 2001 horror game Silent Hill has many qualities that are similar to film music. Modern game soundtracks have become more complex, and often features the overlapping of multiple melodic lines as they are composed for multi-instrument ensembles such as an orchestra. Fortunately, these multiple melodic lines offer ample avenues for parallelization, allowing for parallel synthesis of different parts.

In addition, the rigid structure makes such music easier to predict and generate with a Markov Model — resulting in higher quality and more harmonious music.

The Markov Model of music generation was chosen for its popularity, good musical results, and ease of implementation, understanding, and parallelization. The model is easy to understand and implement, as it is based on using many probability matrices to determine the next notes of a melodic line. These matrices are easy to store and manipulate in many different programming contexts, including the GPUs we wish to use. Other music generation algorithms introduce too many new challenges, such as the recursive nature of Grammars (which may be difficult as it is not known beforehand how may recursions/threads we would need to spawn), or the poor music production of Structure and Probability techniques (Dannenberg, 2018). Finally, it has been shown that Markov Models produce good musical results, where computer compositions and human compositions had similar rankings in formal listening tests (Elowsson and Friderg, 2012). Therefore, implementing a Markov Model for our project would be the best choice for producing music with integrity and quality.

In a Markov Model, multiple matrices are built from training data, or files of music that fits the patterns and genre of the music we wish to create. Most commonly, this takes the form of counting the number of transitions from one note to another, and storing this count in a matrix. These matrices are then normalized and turned into probability matrices. For each melodic line, the algorithm uses the current (and perhaps some number of past) notes to determine the correct matrix and matrix row to look at. It then uses that matrix section to determine the next note, based on the probabilities in the matrix. Because the matrices are training on music of the same structure and genre, it is likely that the resulting melodies also follow the same structure and sound similar to the training pieces. The resulting melodies are then combined and outputted through MIDI.

# Implementation

Our project consisted of three major portions:

1. __Training:__ preprocessing MIDI files and training a Markov Model based on these MIDI files.
2. __Generation:__ using the Markov Model that was trained to generate music.
3. __Client:__ Takes in custom user input and requests music generation from the server, which it plays to the user.

We used two machines for our project

1. __AWS Machine:__ 64 Xeon Phi CPUs, 16 NVIDIA Tesla-K80 GPUs, 768 GB of RAM. Used for training and generation. Needed to have CUDA and multi-threading capabilities.
2. __Personal Laptop:__ Dell XPS, used as the client for requesting music from the AWS machine. Did not need any special requirements except for internet, display and keyboard.


## Training

### Preprocessing

Before we can do any training, we need to preprocess our MIDIs into a format our training algorithm can easily parse, since MIDIs themselves are very complicated, and contain many components that we would not want to have to deal with  

### Markov Model

## Generation

### Server

### Client

The client takes in user input to build a music request to the server.

# Results

Our speedup results indicate...

How does our music sound?

# Conclusion

# Future Work

A major goal of video games is to give the gamer an immersive experience in the virtual world. One common feature is realistic graphics and physics in the game. A feature that has been seldom explored is dynamic music that adapts to the context of the game. A possible extension of our project is to integrate our music generation algorithm into a video game, where the algorithm not only uses trained data, but also the current state of the video game to generate music. Such dynamically-generated music would have a similar effect to how soundtracks in movies are synchronized to actions. Such synchronized soundtracks are very effective, because they would allow for the game sound to be a form of live feedback to the gamer.

# References

- Elowsson, A. and Friberg, A. “Algorithmic Composition of Popular Music,” in Proceedings of the 12th International Conference on Music Perception and Cognition and the 8th Triennial Conference of the European Society for the Cognitive Sciences of Music, July 2012, pp 276-285
- Dannenberg, R. (2018). Music Generation and Algorithmic Composition [Powerpoint slides]. Retrieved from http://www.cs.cmu.edu/~./music/cmsip/slides/05-algo-comp.pdf
- Seabrook, Andrea (Host). 13 April, 2008. The Evolution of Video Game Music [All Things Considered]. https://www.npr.org/templates/story/story.php?storyId=89565567
- Collins, Karen. “In the Loop: Creativity and Constraint in 8-Bit Video Game Audio.” Twentieth-Century Music, vol. 4, no. 2, 2007, pp. 209–227., doi:10.1017/S1478572208000510.
- Whalen, Zach. "Case Study: Film Music vs. Video game Music: The case of Silent Hill." https://books.google.com/books?hl=en&lr=&id=j-iqBgAAQBAJ&oi=fnd&pg=PA68&dq=video+game+music&ots=uUmpe1dvJ3&sig=r_mUJw5q0ZgTKLQquVjLM3CNEhY#v=onepage&q=video%20game%20music&f=false
# Work Distribution