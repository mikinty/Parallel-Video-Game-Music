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
float* majorHighNotes;
float* majorLowNotes;
float* majorChords;
float* minorHighNotes;
float* minorLowNotes;
float* minorChords;

//Device matrices mirroring host matrices above
float* deviceMajorHighNotes;
float* deviceMajorLowNotes;
float* deviceMajorChords;
float* deviceMinorHighNotes;
float* deviceMinorLowNotes;
float* deviceMinorChords;

//Struct describing a note/chord from the input files
struct sound_t {
	int tone;
	int duration;
};
