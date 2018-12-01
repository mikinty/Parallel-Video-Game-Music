#define NUM_TONES 96
#define NUM_CHORDS 1728
#define NUM_DUR 15
#define NUM_THREADS 16
#define INIT_ARRAY_LENGTH 1000
#define NUM_NOTES NUM_TONES * NUM_DUR

struct sound_t {
	int tone;
	int duration;
}