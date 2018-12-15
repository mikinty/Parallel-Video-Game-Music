---
layout: default
---

# Abstract/Summary

Most video games use static soundtracks, which although can be impressive in their own right, can quickly become overplayed to the gamer. In this project, we built a complete system that can process, learn, and generate new video game music. Our system takes in MIDI files, which are processed and trained as a Markov Chain in CUDA on multiple GPUs. Finally, we use our Markov Chain to generate a multi-voice music sample through multi-threading. Speedup analysis shows that our system performs significantly better than a sequential algorithm, allowing for a greater possibility of game integration. We demonstrate a working client interface in which the client requests and produces music of different types that is generated (in parallel) by the server, using the previously created training data.

# Background

## Musical Background

Through the forms and rules outlined in music theory, humans already follow many rigid guidelines when composing music. For example, there are basic rules of tonality and music structure that can decide whether a piece of music sounds good or clashing to the ear ("Intervals"). Therefore, it seems natural to extend these music theory rules to build an algorithmic composer. In particular, the genre of video game music, which in the early 1970s and 80s was originally designed for auxiliary reasons, such as a "heart rate changer" (NPR), and was limited in technology to 8-bit sound (Collins, 2007), is often characterized by a repetitive and structured format, which is ideal as a starting point for algorithmic composers.

On the other hand, in modern times, video game music has evolved to serve a larger purpose in the gaming experience. For example, Whalen discusses how music in the 2001 horror game Silent Hill has many qualities that are similar to film music. Modern game soundtracks have become more complex, and often features the overlapping of multiple melodic lines as they are composed for multi-instrument ensembles such as an orchestra (Collins 2008). Fortunately, these multiple melodic lines offer ample avenues for parallelization, allowing for parallel synthesis of different parts. In addition, the rigid structure makes such music easier to predict and generate with a Markov Model — resulting in higher quality and more harmonious music (Schulze 2011).

The Markov Model of music generation was chosen for its popularity, good musical results, and ease of implementation, understanding, and parallelization. The model is easy to understand and implement, as it is based on using many probability matrices to determine the next notes of a melodic line. These matrices are easy to store and manipulate in many different programming contexts, including the GPUs we wish to use. Other music generation algorithms introduce too many new challenges, such as the recursive nature of Grammars (which may be difficult as it is not known beforehand how may recursions/threads we would need to spawn), or the poor music production of Structure and Probability techniques (Dannenberg, 2018). Finally, it has been shown that Markov Models produce good musical results, where computer compositions and human compositions had similar rankings in formal listening tests (Elowsson and Friderg, 2012). Therefore, implementing a Markov Model for our project is the best choice for producing music with integrity and quality.

## Key Data Structures and Operations

In a Markov Model, multiple matrices are built from training data, or files of music that fit the patterns and genre of the music we wish to create. This takes the form of counting the number of transitions from one note to another, and storing this count in a matrix. In our case, we produce a total of 6 matrices - high melodic line, low melodic line, and chord matrices for major and minor keys. For melodic line matrices, notes are stored as a pair of (tone, duration), where the tone represents a typical musical tone ranging from 0 to 73, and the duration is equal to the number of sixteenth notes that this note is held (so a duration of 16 is equal to a whole note). The melodic line matrices are of depth 2, meaning that the rows are indexed by (note 1, note 2) pairs and the columns indexed by (note 3), where a value of x in this matrix cell means that a total of x transitions were counted where note 3 was preceeded by note 1, note 2 in that order. During training, the indicies are flattened to be stored in a row-major matrix. Upon output, these matrices are printed to a space-delimited text file.

For chord matrices, chord are stored in base 12 (12 notes per octave) where the highest (most significant) note in the chord is in the most significant space. Chords also have no duration marker. The chord matrix has depth 1, meaning rows and columns are both indexed by single notes. Like melodic line matrices, indicies are flattened to be stored in a row-major matrix, and printed to a text file.

### INSERT FIGURE ABOUT MATRICES

During the training phase, the above 6 matrices are initialized to 0. As threads step through the MIDI data, they perform atomicAdd operations on the proper cells. The major operations on the above matrices are memory loads and stores between the GPU devices and the host, atomicAdds on the GPUs, and the final print and store performed by the CPU to a text file. Because of the size of the melodic line matrices, they are split between 3 GPUs each - where each GPU stores a contiguous line of rows of the larger matrix. When loading to the CPU, these 3 GPUS load in parallel as they load to contiguous lines of CPU memory.

### INSERT FIGURE ABOUT TRANSFERS AND BLOCKING

For each melodic line, the algorithm uses the previous 2 notes to determine the correct matrix and matrix row to look at. It then uses that matrix section to determine the next note, based on the probabilities in the matrix. Because the matrices are training on music of the same structure and genre, it is likely that the resulting melodies also follow the same structure and sound similar to the training pieces. The resulting melodies are then combined and outputted through MIDI. Music is stored in nested lists - for every part, there is a list of (tone, duration) pairs representing the music for that part. 

### INSERT FIGURE ABOUT MUSIC

Each part line in the music is generated in parallel, through step by step casing on the previous notes as mentioned above. This results in many operations for reading the large matrices and normalizing rows. The only operation on the outputted music is stores to this data structure.

## Algorithm Input and Outputs

Inputs to this algorithm are the MIDI files containing instances of music similar to that which we want to generate. After pre-processing, the data came out to about 200MB worth of text files. Output is the sound of generated music from the client, as well as the client interface. Intermediate outputs include the transition matrices and list of encoded music to be outputted. The transition matrices are of size about 2.5GB per melodic line matrix.

## Parallelism Benefits and Dependencies

### Benefits of Parallelism per section:

1. **Matrix generation:**
    - The main computational expense is to parse through the large amounts of data generated by the 1000+ MIDI files, and count every transition. However, each transition is independent of the others, and the correct matrix index for each transition can be computed in parallel. This leads to a huge parallelism benefit, as essentially every line in the input file can be computed on in parallel as opossed to sequentially through each line.
    - In addition, because of our use of 6 different matrices, we also need to determine which matrix to insert the transition into. Because of the large number of GPUs available to us, we can instead assign different GPUs to different matrices, simplifying this computation.
    - Since only the CPU will be reading the input MIDI files, which is sequential and slow, it is expensive to wait for GPU computations and loads/stores. However, we can use asynchronous loads/stores and kernel calls to read in the next notes to count while waiting for GPU computation on the previous notes.
    - Finally, as matrices are very large, it is expensive to load and store from the CPU to the GPU or vice versa. However, as matrice entries are also independent, we can block the matrices into smaller parts and load/store these blocks asynchronously.
2. **Music generation:**
    - Multiple melodic lines can be synthesized in parallel, allowing for faster generation of music through multithreading. The music for each part is computationally expensive to generate, and the many different parts makes the sequential algorithm slow. As parts are mostly independent, we can use parallelization between parts to speed up this process.
    - Since sending and receiving messages has a certain amount of latency between server and client, we may also use asynchronous functions to generate new music and buffer previously generated music.

### Challenges and Dependencies

Because of the confusing formatting of MIDI files and the difficulty in extracting the information we needed, we decided to use outside libraries to pre-process our MIDI data into usable text file formats. However, since we are using outside libraries, this had no parallelization.

During matrix generation, while we split the input files among many threads, different sections of the file could have the same transitions. As threads are updating the same matrix, this leads to contention. However, because of the size of these matrices (and so large number of possible transitions), these contentions will be infrequent and occur between a minimal number of threads. Theoretically, we could parallelize between every single note transition in the input files, and every thread would only need to compute the matrix index for one transition. However the time needed to load matrices onto GPUs and back would vastly dominate any parallelism benefit. Therefore, the amount of parallelism is balanced by the cost of loading both the input music notes and output matrices. Similarly, while matrix entries are independent, every entry must see the entire input note line in order to calculate the correct number of transitions. So, this independency is balanced by the cost of loading and counting the input line. 

Since it is impossible to predict the transitions that appear in the input, there is very low locality in matrix access - there is no way to pre-load the entry we wish to increment. The input line (or any particular section of it) is stepped through in sequential order, so there is some locality there. In addition, a row-major ordering for the matrices was used as during music generation, we must normalize and read from a row at a time. A row-major ordering helps increase locality in that step. Finally, SIMD execution is not helpful, as much of the computation relies on casing (casing on the previous notes, the current note, etc.) and so would be dominated by divergent execution.

Although there are avenues for parallelization as seen above, music has many inherent synchronizational components that make it difficult to parallelize. For example, even if melodic lines can be computed in parallel, they need to sound harmonious when played together. Therefore, we assigned a maximum of 10 differing melodic lines to the music generation algorithm, limiting the amount of parallelization. This also drove our choice to use multithreading instead of CUDA for this section, as there will only ever be a maximum of 10 threads running at a time. Similarly, we generate a few measures of music at a time before synching, both to give time for the transfer to the client and to synchronize the music. In terms of locality, the row major ordering of the matrices allow for better locality, since we need to compute on a row at a time. Also, certain threads will only access certain matrices, as each musical part will only be computing on one particular matrix. However, the random nature of the accesses means there is no way to pre-load any particular row. Again, SIMD execution will be hurt by divergent execution, as much of the note generation relies on casing on previous notes. 

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

Before we can do any training, we need to preprocess our MIDIs into a format our training algorithm can easily parse, since MIDIs themselves are very complicated, and contain many components that we would not want to deal with. Because of the confusing nature of MIDI files, we chose to use outside libraries in order to help with the parsing. Since this was not the focus of our project, we did not implement paralleism here. 

### Markov Model - CUDA and C++

Markov Model training was done on the AWS Machines using C++ and CUDA capabilities. A total of 14 GPUs were used during training. 

Blocking matrices + mapping problem

Async queue of kernel operations and memcpy and pinned memory

Previous iterations attempted to use shared memory between threads in the same block to increase the speed of matrix access.

Another attempted optimization was the use of locks as opposed to atomicAdd.

Another optimization was using a peer-to-peer memory access between GPUs, in order to better parallelize the input file between more threads.

There was no previous code for this algorithm, and the original serial algorithm mapped well to a parallel machine using the techniques above. Since it was very simple (step through every line of an input file and increment a matching matrix entry), there was not much room for change.

## Generation

### Server - Multithreading in Python

In order to write the server-client interface in a more straightforward manner, we decided to use Python. While this would impact our overall performance, we decided the added difficulty in getting a proper server-client link using other langauges, which is irrelvant to parallelism, would be too much extra work. The server and multithreading music generation is done on the AWS machines.

Mapping of problem

multi-threading vs cuda

Reading large matrices into python too slow

Buffering data

There was no previous code for this algorithm, and the original serial algorithm mapped well to a parallel machine using the techniques above. Since it was very simple (step through each part and randomly generate next note), there was not much room for change.

### Client

The client takes in user input to build a music request to the server. This is run on our personal laptops, and is not the focus of parallelism.

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

Total credit should be distributed 50% - 50%.
| Annie's Work | Michael's Work | Both |
| ---  | --- | --- |
| Matrix Training Algorithm | Pre-processing Algorithm | Lit Review + Model |
| Music Generation Algorithm | Matrix Loading | Formatting and Algorithm Structure |
| Training Optimizations | Client Interface | Checkpoint |
| Music Generation Threading | Buffering | Further Reference Research |
| | Making Figures | Poster Creation |
| | | Writeup and Presentation |