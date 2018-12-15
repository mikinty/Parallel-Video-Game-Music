---
layout: default
---

# Abstract/Summary

Most video games use static soundtracks, which although can be impressive in their own right, can quickly become overplayed to the gamer. In this project, we built a complete system that can process, learn, and generate new video game music. Our system takes in MIDI files, which are processed and trained as a Markov Chain in CUDA on multiple GPUs. Finally, we use our Markov Chain to generate a multi-voice music sample through multi-threading. Speedup analysis shows that our system performs significantly better than a sequential algorithm, allowing for a greater possibility of game integration. We demonstrate a working client interface in which the client requests and produces music of different types that is generated (in parallel) by the server, using the previously created training data.

# Background

Through the forms and rules outlined in music theory, humans already follow many rigid guidelines when composing music. For example, there are basic rules of tonality and music structure that can decide whether a piece of music sounds good or clashing to the ear ("Intervals"). Therefore, it seems natural to extend these music theory rules to build an algorithmic composer. In particular, the genre of video game music, which in the early 1970s and 80s was originally designed for auxiliary reasons, such as a "heart rate changer" (NPR), and was limited in technology to 8-bit sound (Collins, 2007), is often characterized by a repetitive and structured format, which is ideal as a starting point for algorithmic composers.

On the other hand, in modern times, video game music has evolved to serve a larger purpose in the gaming experience. For example, Whalen discusses how music in the 2001 horror game Silent Hill has many qualities that are similar to film music. Modern game soundtracks have become more complex, and often features the overlapping of multiple melodic lines as they are composed for multi-instrument ensembles such as an orchestra (Collins 2008). Fortunately, these multiple melodic lines offer ample avenues for parallelization, allowing for parallel synthesis of different parts. In addition, the rigid structure makes such music easier to predict and generate with a Markov Model — resulting in higher quality and more harmonious music (Schulze 2011).

The Markov Model of music generation was chosen for its popularity, good musical results, and ease of implementation, understanding, and parallelization. The model is easy to understand and implement, as it is based on using many probability matrices to determine the next notes of a melodic line. These matrices are easy to store and manipulate in many different programming contexts, including the GPUs we wish to use. Other music generation algorithms introduce too many new challenges, such as the recursive nature of Grammars (which may be difficult as it is not known beforehand how may recursions/threads we would need to spawn), or the poor music production of Structure and Probability techniques (Dannenberg, 2018). Finally, it has been shown that Markov Models produce good musical results, where computer compositions and human compositions had similar rankings in formal listening tests (Elowsson and Friderg, 2012). Therefore, implementing a Markov Model for our project would be the best choice for producing music with integrity and quality.

## Key Data Structures and Operations

In a Markov Model, multiple matrices are built from training data, or files of music that fits the patterns and genre of the music we wish to create. This takes the form of counting the number of transitions from one note to another, and storing this count in a matrix. In our case, we produce a total of 6 matrices - high melodic line, low melodic line, and chord matrices for major and minor keys. For melodic line matrices, notes are stored as a pair of (tone, duration), where the tone represents a typical musical tone ranging from 0 to 73, and the duration is equal to the number of sixteenth notes that this note is held (so a duration of 16 is equal to a whole note). The melodic line matrices are of depth 2, meaning that the rows are indexed by (note 1, note 2) pairs and the columns indexed by (note 3), where a value of x in this matrix cell means that a total of x transitions were counted where note 3 was preceeded by note 1, note 2 in that order. For chord matrices, chord are stored in base 12 (12 notes per octave) where the highest (most significant) note in the chord is in the most significant space. Chords also have no duration marker. The chord matrix has depth 1, meaning rows and columns are both indexed by single notes.

During the training phase, the above 6 matrices are initialized to 0. As threads step through the MIDI data, they perform atomicAdd operations on the proper cells. The major operations on the above matrices are memory loads and stores between the GPU devices and the host, atomicAdds on the GPUs, and the final print and store performed by the CPU to a text file.

For each melodic line, the algorithm uses the current (and perhaps some number of past) notes to determine the correct matrix and matrix row to look at. It then uses that matrix section to determine the next note, based on the probabilities in the matrix. Because the matrices are training on music of the same structure and genre, it is likely that the resulting melodies also follow the same structure and sound similar to the training pieces. The resulting melodies are then combined and outputted through MIDI. Music is stored in nested lists - for every part, there is a list of (tone, duration) pairs representing the music for that part. 

Operations on the generated music include randomization over a given probability distribution, and casing on previous notes. Since the overall operations are simple, we focused on making use of the powerful machines and multiple GPUs and multithreading capabilities on AWS.

## Algorithm Input and Outputs

Inputs to this algorithm are the MIDI files containing instances of music similar to that which we want to generate. After pre-processing, the data came out to about 200MB worth of text files. Output is the sound of generated music from the client, as well as the client interface. Intermediate outputs include the transition matrices and list of encoded music to be outputted. The transition matrices are of size about 2.5GB per melodic line matrix.

## Parallelism Benefits and Dependencies

###Benefits of Parallelism per section:

1. **Matrix generation:**
    - Matrix entries are computed in parallel, as there are no dependencies between entries. Therefore, every entry in the matrix could be compute in parallel. However, to avoid duplicating large matrices, we instead block the matrices between multiple (3) GPUs
    - There are many matrices for varying melodic lines — bass, melody, counter-melody, etc. These matrices can be computed in parallel, and each GPU is assigned only one matrix to work on.
    - The notes file can be parsed in parallel - in particular, different threads on each GPU split the input files into sections where each thread is responsible for one section
2. **Music generation:**
    - Multiple melodic lines can be synthesized in parallel, allowing for faster generation of music through multithreading

### Challenges and Dependencies

Although there are avenues for parallelization as seen above, music has many inherent synchronizational components that make it difficult to parallelize. For example, even if melodic lines can be computed in parallel, they need to sound harmonious when played together. Therefore, we assigned a maximum of 10 differing melodic lines to the music generation algorithm, limiting the amount of parallelization. This also drove our choice to use multithreading instead of CUDA for this section, as there will only ever be a maximum of 10 threads running at a time.

During matrix generation, while we split the input files among many threads, different sections of the file could have the same transitions. As threads are updating the same matrix, this leads to contention. However, because of the size of these matrices (and so large number of possible transitions), these contentions will be infrequent and occur between a minimal number of threads. Finally, when using multiple GPUs, there is the issue of duplication and communication costs. Because of the size of our matrices, we calculated it was inefficent to duplicate any parts of the melodic line. Instead, we duplicate some of the input text parsing work, as it has neligible size compared to the overall matrix.

### Workload Characteristics

- **Memory Accesses:** When we are generating melodic lines, related melodic lines will depend on one another — for example, bass lines will be dependent on each others' values in order to keep chords harmonious. Because of these relationships, there is some degree of locality. In addition, similar lines will access similar probability matrices. While accessing these matrices, since notes that are adjacent on the scale are more likely to be transitioned to, the lines may be accessing adjacent areas of the matrix over time. This is another area of locality. On the other hand, it is unlikely that the melodic line and the bass will need to access similar probability matrices, since they have very different structures and rules.
- **Measure by Measure Generation:**
- **Importance of Casing:** Bad SIMD execution

# Approach/Implementation

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

### Markov Model - CUDA and C++

Memory concerns

Multi-GPU usage

Balance of pinned memory, memory transfer, splitting matrices, etc.

Async queue of kernel operations and memcpy

## Generation

### Server - Multithreading in Python

multi-threading vs cuda, numpy

Reading large matrices into python too slow

### Client

The client takes in user input to build a music request to the server.

# Results

## Experimental Setup
Sizes of inputs, how we measured performance

## Speedup Graphs
Our speedup results indicate...

## Impact of Problem Size

## Speedup Limitations

## Execution Time Breakdown

## Machine Choice

## Musical Results
How does our music sound?

# Conclusion

# Future Work

A major goal of video games is to give the gamer an immersive experience in the virtual world. One common feature is realistic graphics and physics in the game. A feature that has been seldom explored is dynamic music that adapts to the context of the game. A possible extension of our project is to integrate our music generation algorithm into a video game, where the algorithm not only uses trained data, but also the current state of the video game to generate music. Such dynamically-generated music would have a similar effect to how soundtracks in movies are synchronized to actions. Such synchronized soundtracks are very effective, because they would allow for the game sound to be a form of live feedback to the gamer.

# References

- Collins, Karen (2008). Game sound: an introduction to the history, theory, and practice of video game music and sound design. MIT Press. pp. 112–118. ISBN 0-262-03378-X.
- Collins, Karen. “In the Loop: Creativity and Constraint in 8-Bit Video Game Audio.” Twentieth-Century Music, vol. 4, no. 2, 2007, pp. 209–227., doi:10.1017/S1478572208000510.
- Elowsson, A. and Friberg, A. “Algorithmic Composition of Popular Music,” in Proceedings of the 12th International Conference on Music Perception and Cognition and the 8th Triennial Conference of the European Society for the Cognitive Sciences of Music, July 2012, pp 276-285
- Dannenberg, R. (2018). Music Generation and Algorithmic Composition [Powerpoint slides]. Retrieved from http://www.cs.cmu.edu/~./music/cmsip/slides/05-algo-comp.pdf
- “Intervals.” Music Theory, Sibelius Academy, www2.siba.fi/muste1/index.php?id=64&la=en.
- Schulze, Walter, and Brink van der Merwe. “Music Generation With Markov Models.” Computingnow, 2011, www.computer.org/cms/ComputingNow/homepage/2011/1011/W_MM_MusicGenerationwithMarkovModels.pdf.
- Seabrook, Andrea (Host). 13 April, 2008. The Evolution of Video Game Music [All Things Considered]. https://www.npr.org/templates/story/story.php?storyId=89565567
- Whalen, Zach. "Case Study: Film Music vs. Video game Music: The case of Silent Hill." https://books.google.com/books?hl=en&lr=&id=j-iqBgAAQBAJ&oi=fnd&pg=PA68&dq=video+game+music&ots=uUmpe1dvJ3&sig=r_mUJw5q0ZgTKLQquVjLM3CNEhY#v=onepage&q=video%20game%20music&f=false

# Work Distribution
