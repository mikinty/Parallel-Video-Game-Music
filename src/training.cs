
using Melanchall.DryWetMidi.Smf;

static int Main(string[] args){
	 if (args.Length < 2)
        {
            System.Console.WriteLine("Please enter a MIDI File Path and a Text File Path.");
            return 1;
        }
      ConvertMidiToText(args[0], args[1]);
}

public static int GetTranspose(MidiFile midiFile){
	//Do something to get the Key
	return 0;
}

public static void ConvertMidiToText(string midiFilePath, string textFilePath)
{
    var midiFile = MidiFile.Read(midiFilePath);
    var channelSplit = MidiFile.SplitByChannel(midiFile);

    for (int i = 0; i < channelSplit.Length; i++)
    {
        var channelMidi = channelSplit[i];
        var shift = GetTranspose(channelMidi);
    	var avgNote = channelMidi.GetNotes().Average(n => n.NoteNumber);
    	var count = channelMidi.GetNotes().Length;
    
        textFilePath = textFilePath + $"{i}.txt";
        File.AppendAllText(textFilePath, $"{avgNote} {count}\n");

		File.AppendAllText(textFilePath,
                       	   channelMidi.GetNotes()
                               .Select(n => $"{n.NoteNumber + shift} {n.Time} {n.Length}"));

    }
}