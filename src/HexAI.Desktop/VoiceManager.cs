using System;
using System.Speech.Recognition;
using System.Speech.Synthesis;
using System.Threading.Tasks;

namespace HexAI.Desktop;

public class VoiceManager : IDisposable
{
    private SpeechSynthesizer? _synthesizer;
    private SpeechRecognitionEngine? _recognizer;
    private bool _isListening;
    private TaskCompletionSource<string>? _tcs;

    public bool IsVoiceSupported { get; private set; }
    
    public VoiceManager()
    {
        try
        {
            if (OperatingSystem.IsWindows())
            {
                _synthesizer = new SpeechSynthesizer();
                _synthesizer.SetOutputToDefaultAudioDevice();
                
                _recognizer = new SpeechRecognitionEngine();
                _recognizer.SetInputToDefaultAudioDevice();
                _recognizer.LoadGrammar(new DictationGrammar());
                
                _recognizer.SpeechRecognized += Recognizer_SpeechRecognized;
                _recognizer.SpeechRecognitionRejected += Recognizer_SpeechRecognitionRejected;

                IsVoiceSupported = true;
            }
        }
        catch
        {
            IsVoiceSupported = false;
        }
    }

    public void SpeakAsync(string text)
    {
        if (!IsVoiceSupported || _synthesizer == null || string.IsNullOrWhiteSpace(text)) return;
        string cleanText = CleanForSpeech(text);
        if (!string.IsNullOrWhiteSpace(cleanText))
        {
            _synthesizer.SpeakAsync(cleanText);
        }
    }

    public void StopSpeaking()
    {
        if (IsVoiceSupported && _synthesizer != null)
        {
            _synthesizer.SpeakAsyncCancelAll();
        }
    }

    public async Task<string> ListenAsync()
    {
        if (!IsVoiceSupported || _recognizer == null) return "";
        if (_isListening) return "";

        try
        {
            _isListening = true;
            _tcs = new TaskCompletionSource<string>();
            _recognizer.RecognizeAsync(RecognizeMode.Single);
            return await _tcs.Task;
        }
        catch
        {
            return "";
        }
        finally
        {
            _isListening = false;
        }
    }

    public void StopListening()
    {
        if (IsVoiceSupported && _recognizer != null && _isListening)
        {
            _recognizer.RecognizeAsyncCancel();
            _tcs?.TrySetResult("");
        }
    }

    private void Recognizer_SpeechRecognized(object? sender, SpeechRecognizedEventArgs e)
    {
        if (e.Result.Confidence > 0.5f) _tcs?.TrySetResult(e.Result.Text);
        else _tcs?.TrySetResult("");
    }

    private void Recognizer_SpeechRecognitionRejected(object? sender, SpeechRecognitionRejectedEventArgs e)
    {
        _tcs?.TrySetResult("");
    }

    private static string CleanForSpeech(string input)
    {
        input = System.Text.RegularExpressions.Regex.Replace(input, @"```[\s\S]*?```", " [kod bloğu] ");
        input = input.Replace("**", "").Replace("*", "");
        return input.Trim();
    }

    public void Dispose()
    {
        _synthesizer?.Dispose();
        _recognizer?.Dispose();
    }
}
