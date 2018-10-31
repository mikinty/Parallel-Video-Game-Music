---
layout: default
---

# Proposal

## Summary
We want to generate video game music using Markov chains, with parallel computation through GPUs for real-time music output. We will be analyzing speedup for two portions of the project:

  1. Matrix generation: build the Markov matrices (?) that will be used for music generation
  2. Music generation: parallel synthesis of multiple melodic lines using trained matrices (?) from matrix generation

## Background
Why video games
- many melodic lines
- repetitive and structured (easier to predict/dgenerate msuic w/markov chains)
Why Markov chains -> theory (cite some papers / computer msuic class slides)

### Implementation
Diagram 

The parts of our project that benefit from parallelism are:
- Matrix generation:
    - There are a lot of entries in the matrix
    - Multiple matrices for different melodic lines
    - Generating weights based on grammars require multiple data lookups
    - Training on multiple files can be read/processed in parallel 
- Music generation:
    - There are many melodic lines that happen at the same time
    - Conductor needs to be aware of many lines at once, and also has to perform many operations


Syncing stuff so we can properly communicate

## Challenge
Designing a decent model for generating **GOOD** music. 
Music theory is hard : (

(toipic sentence)
Video game music often has many melodic lines and theme changes that require concurreent operations.

### What is hard:
- **Conductor:** In charge of the overall direction of the music. This includes theme changes, dynamics, tempo, etc. This not only is many operations we have to keep track of in parallel, but also requires synchronization across all the threads being controlled by the conductor. E.g. if we transition from A to B theme, we need all the threads to be aware and transition at the same time.
- **Melodic lines:** There are many melodic lines happenigna tt he same time. Generating them quickly requires paralelism, but also in order to make the music sound good, we need to make sure that the melodic lines communicate with one another to preserve harmony, chord sequences, style.


### Parallel Challenges:
- Memory Accesses: When we are generating melodic lines, related melodic liens will depend on one antoher, which means, for example, that bass lines will be dependnet on each others' values. Because of these relationships, there is some degree of locality.
- Commnuciation between lines: In order to maintain (music theory) and harmony, we need music generation threads to be aware of one another and make sure things are in check.
- Communication from conudctor: The conductor needs to be able to look at what everybody is doing at all times. The conductor will mostly be performing reads on other data, and writes to a global data structure containing values that other people care about and read from. There are challenges associated to reducing contention when the conductor tries to write to its global structure while many other threads are trying to read from it.

### Mapping workload?
- CPU to GPU to CPU: Set up communication between the CPU host to the GPU kernel.
- GPU computation: We need to figure out how to map our Markov model computations into CUDA and also how to convert the results into MIDI.
- Web packets: We have to package our MIDI data into a web packet that can be comprehended by our client.


## Resources
We want to utilize GPUs for faster matrix computation. Therefore, we need access to machines with GPUs.

Our code base is essentially starting from scratch, with some guidance from 15418 asst2, where we used CUDA to proces images. 

We are using past computer music papers as references for building our Markov model for generating music. (provide citations)

We need some sort of device to play the MIDI file. We are most likely going to use our laptop, but we need to figure out how to write a program that can receive web packets and play MIDI in real time.

https://bisqwit.iki.fi/source/adlmidi.html C++ MIDI player


## Goals and Deliverables
We plan to create a video game music generator that produces music in real-time.

We hope to create a Markov model that is more specialized in generating good video game music. We would also like our application to be more widely accessible, which would require setting up a server and communicating to clients, most likely through a web browser.

We might also want to be able to demonstrate the feasibility of using some sort of parallel computation for IRL video games. E.g. many soundtracks in video games are static and built-in, and thus don't have much variation. (video games already use GPUs, interesting to see if music can have a role in using the GPUs in addition to graphics which currently use them)

Our demo...

Analysis: Our algorithm will demonstrate significant speedup from utilizing GPUs, versus just using sequential code. 

## Platform Choice
We are planning to use the Thrust library on top of the C++ CUDA in order to write more efficient code. Thrust has significant performance benefits over C++, and also is easier to develop in, resulting in more productive development.

Using GPUs to perform our computation is a good choice, because:

- **Matrix generation:** There are many elements of the matrix that can be computed in parallel, where each element does not require too much work.
- **Matrix multiplication:** This is a well-known problem that parallelizes well on GPUs. Also, like matrix generation, there are many small problems that can be done in parallel, which works well on GPUs.

## Schedule
| Dates | Goal | Description |
| --- | --- | --- |
| November 4 | Lit Review + Model | Learn about how MIDI is generated, and how Markov does that. We also need to read about how video game music theory works. We want to decide on our model, why it works well, and what algorithm we will use to generate our music. |
| November 11 | Basic Parallel Alg. | Implement a CUDA program that generates matrices for our program. |
| November 18 | Music Generation | Implement the music generation part of our project so we can play generated MIDI files using the matrices we generated. |
| November 19 | Checkpoint | Summarize progress so far on project |
| December 1 | Optimize Music Generation | Optimize the music generation algorithm. |
| December 7 | Optimize Matrix Generation | Optimize the music generation algorithm. |
| December 14 | Wrap-up | Complete final report |
| December 15 | Presentation | Practice presentation |


THRUST