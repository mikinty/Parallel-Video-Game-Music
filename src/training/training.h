#define NUM_TONES 73
#define NUM_CHORDS 1728
#define NUM_DUR 15
/* All chord numbers are offsetted by +101 */
#define CHORD_OFFSET 101
/* num threads per block */
#define NUM_THREADS 16
#define INIT_ARRAY_LENGTH 1000
#define NUM_NOTES NUM_TONES * NUM_DUR

//Host matrices to be created + outputted to files
extern float* majorHighNotes;
extern float* majorLowNotes;
extern float* majorChords;
extern float* minorHighNotes;
extern float* minorLowNotes;
extern float* minorChords;

//Device matrices mirroring host matrices above
extern float* deviceMajorHighNotes;
extern float* deviceMajorLowNotes;
extern float* deviceMajorChords;
extern float* deviceMinorHighNotes;
extern float* deviceMinorLowNotes;
extern float* deviceMinorChords;

//Struct describing a note/chord from the input files
struct sound_t {
	int tone;
	int duration;
};

void initCuda();
void freeCuda();
void countTransitionsCuda(sound_t* soprano, int sLength, sound_t* bass, int bLength, std::string mood);
void normalizeCuda();
