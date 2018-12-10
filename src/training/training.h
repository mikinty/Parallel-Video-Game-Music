#define NUM_TONES 73
#define NUM_CHORDS 1728
#define NUM_DUR 15
/* All chord numbers are offsetted by +101 */
#define CHORD_OFFSET 101
/* num threads per block */
#define NUM_THREADS 1000
#define INIT_ARRAY_LENGTH 100000 
#define NUM_NOTES NUM_TONES * NUM_DUR
/* CUDA GPU Device Ids */
#define MAJOR_HIGH_DEVICE 0
#define MAJOR_LOW_DEVICE 1
#define MAJOR_CHORD_DEVICE 2
#define MINOR_HIGH_DEVICE 3
#define MINOR_LOW_DEVICE 4
#define MINOR_CHORD_DEVICE 5

//Host matrices to be created + outputted to files
extern int* majorHighNotes;
extern int* majorLowNotes;
extern int* majorChords;
extern int* minorHighNotes;
extern int* minorLowNotes;
extern int* minorChords;

//Device matrices mirroring host matrices above
extern int* deviceMajorHighNotes;
extern int* deviceMajorLowNotes;
extern int* deviceMajorChords;
extern int* deviceMinorHighNotes;
extern int* deviceMinorLowNotes;
extern int* deviceMinorChords;

//Struct describing a note/chord from the input files
struct sound_t {
	int tone;
	int duration;
};

void initCuda();
void freeCuda();
void countTransitionsCuda(sound_t* soprano, int sLength, sound_t* bass, int bLength, std::string mood);
void synchAllCuda();
void cudaToHost();
