#define NUM_TONES 73
#define NUM_CHORDS 1728
#define NUM_DUR 15
/* All chord numbers are offsetted by +101 */
#define CHORD_OFFSET 101
/* num threads per block */
#define NUM_THREADS 1024
#define ARRAY_LENGTH 10000000000
#define NUM_NOTES (NUM_TONES * NUM_DUR)
/*Constants regarding matrices*/
#define NUM_MATRICES 6
#define NUM_GPU_PER_MATRIX 3 
#define MATRIX_BLOCK_ROWS ((NUM_NOTES * NUM_NOTES + NUM_GPU_PER_MATRIX - 1) / NUM_GPU_PER_MATRIX)
/* CUDA GPU Device Ids */
#define MAJOR_HIGH_DEVICE 0
#define MAJOR_LOW_DEVICE (MAJOR_HIGH_DEVICE + NUM_GPU_PER_MATRIX)
#define MINOR_HIGH_DEVICE (MAJOR_LOW_DEVICE + NUM_GPU_PER_MATRIX)
#define MINOR_LOW_DEVICE (MINOR_HIGH_DEVICE + NUM_GPU_PER_MATRIX)
#define MAJOR_CHORD_DEVICE (MINOR_LOW_DEVICE + NUM_GPU_PER_MATRIX)
#define MINOR_CHORD_DEVICE (MAJOR_CHORD_DEVICE + 1)

//Struct describing a note/chord from the input files
struct sound_t {
  int tone;
  int duration;
};

//Host matrices to be created + outputted to files
extern int* majorHighNotes;
extern int* majorLowNotes;
extern int* majorChords;
extern int* minorHighNotes;
extern int* minorLowNotes;
extern int* minorChords;

//Host note arrays
extern sound_t* majorSoprano;
extern sound_t* majorBass;
extern sound_t* minorSoprano;
extern sound_t* minorBass;

//Device matrices mirroring host matrices above
extern int* deviceMajorHighNotes[NUM_GPU_PER_MATRIX];
extern int* deviceMajorLowNotes[NUM_GPU_PER_MATRIX];
extern int* deviceMajorChords;
extern int* deviceMinorHighNotes[NUM_GPU_PER_MATRIX];
extern int* deviceMinorLowNotes[NUM_GPU_PER_MATRIX];
extern int* deviceMinorChords;

//Device note arrays, with a +1 for the chord GPU
extern sound_t* deviceMajorSoprano[NUM_GPU_PER_MATRIX + 1];
extern sound_t* deviceMajorBass[NUM_GPU_PER_MATRIX + 1];
extern sound_t* deviceMinorSoprano[NUM_GPU_PER_MATRIX + 1];
extern sound_t* deviceMinorBass[NUM_GPU_PER_MATRIX + 1];

//Cuda functions called by main
void initCuda();
void freeCuda();
void countTransitionsCuda(sound_t* soprano, int sLength, sound_t* bass, int bLength, int mood);
void cudaStreamSynch(int mood);
void cudaToHost();
